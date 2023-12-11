import torch
import torch.utils.data as dutils
import numpy as np
import argparse
from torchmetrics.functional.text import bleu_score, rouge_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List
from tqdm import tqdm
import os
from os.path import join

from config import ModelConfig, EvalConfig
from model import Model, from_config
from dataset import TextDataset, collate

device = torch.device("cuda")


# Taken from DiffuSeq
def get_sentence_bleu(pred: str, gt: str):
    return sentence_bleu([gt.split()], pred.split(), smoothing_function=SmoothingFunction().method4)


def compute_metric(name: str, preds: List[str], gts: List[str], add_special_tokens=False):
    """Computes a given metric on a list of preds and gts."""
    # If add_special_tokens is enabled, start and end 'tokens' are added to all strings.
    # This is **bad practice**, but it is done in DiffuSeq, so I optionally replicate it here for fair comparison.
    if add_special_tokens:
        preds = [f"[CLS] {i} [SEP]" for i in preds]
        gts = [f"[CLS] {i} [SEP]" for i in gts]

    name = name.upper()
    if name == "BLEU":
        return bleu_score(preds, gts).item()
    elif name == "SENTENCE-BLEU":
        scores = [get_sentence_bleu(pred, gt) for pred, gt in zip(preds, gts)]
        return sum(scores) / len(scores)
    elif name == "ROUGE":
        return rouge_score(preds, gts)["rougeL_fmeasure"].item()
    else:
        raise NotImplementedError(f"Unsupported metric '{name}'.")


def _eval_model_on_dataset(model: Model, dataset: TextDataset, config: EvalConfig, batch_size: int = 64):
    """Evaluates `model` on `dataset` entirely, returning two lists of strings: (preds, ground truths)."""
    eval_scheduler = config.get_scheduler(model.timesteps)

    gts = []
    preds = []

    dl = dutils.DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate)

    # Compute all preds
    with torch.no_grad():
        it = tqdm(dl)
        for step, (xs_tok, ys_tok, xs_l, ys_l) in enumerate(it):
            xs_emb = model.embed(xs_tok)
            # TODO: Pass CFG scale
            toks = model.inference(xs_emb, xs_l, ys_l, eval_scheduler, config.nsteps, clamping_trick=config.clamp)

            for ind, pred in enumerate(toks):
                gt = ys_tok[ind][:ys_l[ind]]

                gts.append(model.tokenizer.decode(gt))
                preds.append(model.tokenizer.decode(pred))

    return preds, gts


def eval_model(model: Model, dataset: TextDataset, metric_names: List[str], config: EvalConfig, batch_size: int = 64):
    """Evaluates a model on the provided dataset and computes all requested metrics, which are returned as a dict."""
    preds, gts = _eval_model_on_dataset(model, dataset, config, batch_size)

    out = {}
    for name in metric_names:
        out[name] = compute_metric(name, preds, gts)

    for name in metric_names:
        out[name + "-padded"] = compute_metric(name, preds, gts, add_special_tokens=True)

    return out


def mbr_select(predss: List[List[str]]) -> List[str]:
    """Returns the MBR set given a list of all candidates (predss has shape S x N where S=MBR set size and N=dataset size)."""
    out = []
    s = len(predss)
    if s == 1:
        return predss[0]
    for preds in zip(*predss):
        scores = np.zeros((s, s))
        for i in range(s):
            for j in range(s):
                scores[i, j] = get_sentence_bleu(preds[i], preds[j])
        chosen_ind = np.argmax(np.sum(scores, axis=1)).item()
        out.append(preds[chosen_ind])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", required=True, help="Path to model directory. Must contain "
                                                                 "config.json and train_config.json files.")
    parser.add_argument("-d", "--data-dir", required=True, help="Path to data directory.")
    parser.add_argument("-n", "--nsteps", type=int, default=30, help="Number of steps to use for sampling")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("-c", "--clamping-trick", action="store_true", help="See Diffusion-LM paper for details.")
    parser.add_argument("-s", "--scheduler", type=str, default="DPM++", help="Scheduler (DPM++ or DDIM).")
    parser.add_argument("-S", "--mbr-set-size", type=int, default=1)
    args = parser.parse_args()

    # TODO: Add CFG
    config = EvalConfig(scheduler=args.scheduler, nsteps=args.nsteps, clamp=args.clamping_trick)

    # Load model
    model_config = ModelConfig.load(args.model_dir)
    model = from_config(model_config).to(device)
    model.load(args.model_dir)
    model.eval()

    # Load + tokenizer test set
    dataset = TextDataset(args.data_dir, split="test", tokenizer=model.tokenizer, device=device)

    # Compute all preds / load from files if computed before
    out_dir = join(args.model_dir, "test_gen", config.get_path())
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving/loading results from {out_dir}...")
    gts = [model.tokenizer.decode(dataset[i][1]) for i in range(len(dataset))]
    predss = []
    for i in range(args.mbr_set_size):
        path = join(out_dir, f"{i}.txt")

        if os.path.isfile(path):
            with open(path, "r") as file:
                preds = [i.strip() for i in file.readlines()]
        else:
            print(f"Evaluating {i+1} / {args.mbr_set_size}...")
            preds, gts_ = _eval_model_on_dataset(model, dataset, config, batch_size=args.batch_size)
            with open(path, "w") as file:
                file.write('\n'.join(preds) + '\n')

            assert gts_ == gts  # just a sanity check

        predss.append(preds)

    # Run MBR decoding to obtain a single list of preds
    preds = mbr_select(predss)

    metric_names = ["BLEU", "ROUGE", "sentence-BLEU"]

    # Compute all metrics
    out = {}
    for name in metric_names:
        out[name] = compute_metric(name, preds, gts)
    for name in metric_names:
        out[name + "-padded"] = compute_metric(name, preds, gts, add_special_tokens=True)

    print("Model path:", args.model_dir)
    print("Evaluation config:", config.get_path())
    print()
    print("Evaluation results:")
    print(out)
