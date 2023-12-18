import torch
import torch.utils.data as dutils
import numpy as np
import argparse
# from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sacrebleu import corpus_bleu, sentence_bleu
from rouge import Rouge
from typing import List
from tqdm import tqdm
import os
from os.path import join
import matplotlib.pyplot as plt

from config import ModelConfig, EvalConfig
from model import Model, from_config
from dataset import TextDataset, collate
from length_model import NormalDist

device = torch.device("cuda")


def get_sentence_bleu(pred: str, gt: str):
    return sentence_bleu(pred, [gt]).score

    # DiffuSeq version: (using NLTK sentence_bleu)
    # return sentence_bleu([gt.split()], pred.split(), smoothing_function=SmoothingFunction().method4)


def self_bleu(hyps: List[List[str]]):
    """
    Computes self-BLEU using sacreBLEU. `hyps` should be a list of possible hypotheses for each input, i.e.
     hyps = [[pred1_version1, pred1_version2, ...], [pred2_version1, ...], ...]
     where each version of pred1 was generated from the same condition.
    """
    total = 0
    for versions in hyps:
        refs = []
        for i in range(len(versions)):
            # The references for versions[i] are all the other versions, except i
            refs.append(versions[:i] + versions[i+1:])

        # Currently, refs[i] = (list of refs for version i)
        # but sacre-BLEU expects refs[i] = (ith ref for version 1, ith ref for version 2, ...)
        # to fix this we just transpose:
        refs = list(zip(*refs))

        total += corpus_bleu(versions, refs).score

    return total / len(hyps)


def compute_metric(name: str, preds: List[str], gts: List[str]):
    """Computes a given metric on a list of preds and gts."""
    name = name.upper()
    if name == "BLEU":
        return corpus_bleu(preds, [gts]).score
    # This (sentence-bleu) is not how BLEU should be calculated, but unfortunately I have seen it done.
    # I don't report the results of this, but keep it here for comparison.
    elif name == "SENTENCE-BLEU":
        scores = [get_sentence_bleu(pred, gt) for pred, gt in zip(preds, gts)]
        return sum(scores) / len(scores)
    elif name == "ROUGE":
        return Rouge().get_scores(preds, gts, avg=True)["rouge-l"]["f"]
    else:
        raise NotImplementedError(f"Unsupported metric '{name}'.")


def _eval_model_on_dataset(model: Model, dataset: TextDataset, model_config: ModelConfig, config: EvalConfig, batch_size: int = 64, callback=None):
    """Evaluates `model` on `dataset` entirely, returning two lists of strings: (preds, ground truths)."""
    eval_scheduler = config.get_scheduler(model_config)
    len_model = config.get_length_model()
    len_model.load(dataset.root_path)

    gts = []
    preds = []

    dl = dutils.DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate)

    # Compute all preds
    with torch.no_grad():
        it = tqdm(dl)
        for step, (xs_tok, ys_tok, xs_l, ys_l) in enumerate(it):
            ys_l_pred = len_model.predict(xs_l, ys_l)
            xs_emb = model.embed(xs_tok)
            toks = model.inference(xs_emb, xs_l, ys_l_pred, eval_scheduler, config, callback=callback)

            for ind, pred in enumerate(toks):
                gt = ys_tok[ind][:ys_l[ind]]

                gts.append(model.tokenizer.decode(gt))
                preds.append(model.tokenizer.decode(pred))

    return preds, gts


def eval_model(model: Model, dataset: TextDataset, metric_names: List[str], model_config: ModelConfig, eval_config: EvalConfig, batch_size: int = 64):
    """Evaluates a model on the provided dataset and computes all requested metrics, which are returned as a dict."""
    preds, gts = _eval_model_on_dataset(model, dataset, model_config, eval_config, batch_size)

    out = {}
    for name in metric_names:
        out[name] = compute_metric(name, preds, gts)

    return out


def mbr_select(predss: List[List[str]]) -> List[str]:
    """Returns the MBR set given a list of all candidates (predss has shape S x N where S=MBR set size and N=dataset size)."""
    out = []
    s = len(predss)
    if s == 1:
        return predss[0]
    for preds in zip(*predss):
        # v1
        scores = np.zeros((s, s))
        for i in range(s):
            for j in range(s):
                scores[i, j] = get_sentence_bleu(preds[i], preds[j])
        chosen_ind = np.argmax(np.sum(scores, axis=1)).item()

        # v2
        # scores = np.zeros((s,))
        # for i in range(s):
        #     others = preds[:i] + preds[i+1:]
        #     scores[i] = sentence_bleu([i.split() for i in others], preds[i].split(), smoothing_function=SmoothingFunction().method4)
        # chosen_ind = np.argmax(scores).item()

        out.append(preds[chosen_ind])
    return out


def get_preds(model_dir, model, dataset, model_config: ModelConfig, eval_config: EvalConfig, seed: int = 0,
              batch_size: int = 128):
    out_dir = join(model_dir, "test_gen", eval_config.get_path())
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving/loading results from {out_dir}...")

    path = join(out_dir, f"{seed}.txt")

    if os.path.isfile(path):
        with open(path, "r") as file:
            preds = [i.strip() for i in file.readlines()]
    else:
        torch.manual_seed(seed)
        preds, _ = _eval_model_on_dataset(model, dataset, model_config, eval_config, batch_size=batch_size)
        with open(path, "w") as file:
            file.write('\n'.join(preds) + '\n')

    return preds


def compute_metrics(preds, gts, metric_names):
    out = {}
    for name in metric_names:
        out[name] = compute_metric(name, preds, gts)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", required=True, help="Path to model directory. Must contain "
                                                                 "config.json and train_config.json files.")
    parser.add_argument("-d", "--data-dir", required=True, help="Path to data directory.")
    parser.add_argument("-n", "--nsteps", type=int, default=30, help="Number of steps to use for sampling.")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("-cl", "--clamp-lerp", action="store_true", help="Lerp clamp strength over time")
    parser.add_argument("-cm", "--clamp-mode", type=int, default=0, help="Clamp mode (0, 1, 2, 3).")
    parser.add_argument("-s", "--scheduler", type=str, default="DPM++", help="Scheduler (DPM++ or DDIM).")
    parser.add_argument("-S", "--mbr-set-size", type=int, default=1)
    parser.add_argument("-cfg", "--cfg", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("-cfgm", "--cfg-mode", type=str, default="constant", help="constant | lerp | alpha")
    parser.add_argument("-lm", "--length-model", type=str, default="oracle", help="Length model")
    args = parser.parse_args()

    eval_config = EvalConfig(scheduler=args.scheduler, nsteps=args.nsteps, cfg=args.cfg, cfg_mode=args.cfg_mode,
                             length_model=args.length_model, clamp_lerp=args.clamp_lerp, clamp_mode=args.clamp_mode)

    # Load model
    model_config = ModelConfig.load(args.model_dir)
    model = from_config(model_config).to(device)
    model.load(args.model_dir)
    model.eval()

    # Load + tokenizer test set
    dataset = TextDataset(args.data_dir, split="test", tokenizer=model.tokenizer, device=device)

    # Compute all preds / load from files if computed before
    gts = [model.tokenizer.decode(dataset[i][1]) for i in range(len(dataset))]
    predss = []
    for i in range(args.mbr_set_size):
        preds = get_preds(args.model_dir, model, dataset, model_config, eval_config, seed=i, batch_size=args.batch_size)
        predss.append(preds)
    # Run MBR decoding to obtain a single list of preds
    preds = mbr_select(predss)

    metric_names = ["BLEU", "ROUGE", "sentence-BLEU"]

    out = compute_metrics(preds, gts, metric_names)

    print("Model path:", args.model_dir)
    print("Evaluation config:", eval_config.get_path())
    print()
    print("Evaluation results:")
    print(str(out).replace(", ",",\t"))


