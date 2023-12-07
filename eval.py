import torch
import torch.utils.data as dutils
from os.path import join
from diffusers import DPMSolverMultistepScheduler
import argparse
from torchmetrics.functional.text import bleu_score, rouge_score
from typing import List
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from config import ModelConfig
from model import Model, from_config
from dataset import TextDataset, collate

device = torch.device("cuda")


# Taken from DiffuSeq
def get_sentence_bleu(pred, gt):
    return sentence_bleu([gt.split()], pred.split(), smoothing_function=SmoothingFunction().method4)


def compute_metric(name: str, preds: List[str], gts: List[str]):
    """Computes a given metric on a list of preds and gts."""
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


def eval_metric(model: Model, dataset: TextDataset, metric_names: List[str], eval_scheduler, nsteps: int,
                batch_size: int = 64, clamping_trick: bool = False):
    """Evaluates a model completely on the provided dataset and computes all requested metrics."""
    gts = []
    preds = []

    dl = dutils.DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate)

    # Compute all preds
    with torch.no_grad():
        it = tqdm(dl)
        for step, (xs_tok, ys_tok, xs_l, ys_l) in enumerate(it):
            xs_emb = model.embed(xs_tok)
            toks = model.inference(xs_emb, xs_l, ys_l, eval_scheduler, nsteps, clamping_trick=clamping_trick)

            for ind, t in enumerate(toks):
                gt = ys_tok[ind][:ys_l[ind]]
                pred = toks[ind]

                gts.append(model.tokenizer.decode(gt))
                preds.append(model.tokenizer.decode(pred))

    out = {}
    for name in metric_names:
        out[name] = compute_metric(name, preds, gts)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", required=True, help="Path to model directory. Must contain "
                                                                 "config.json and train_config.json files.")
    parser.add_argument("-d", "--data-dir", required=True, help="Path to data directory.")
    parser.add_argument("-n", "--nsteps", type=int, default=50, help="Number of steps to use for sampling")
    parser.add_argument("-b", "--batch", type=int, default=64, help="Batch size")
    parser.add_argument("-c", "--clamping-trick", action="store_true")
    args = parser.parse_args()

    model_config = ModelConfig.load(args.model_dir)
    model = from_config(model_config).to(device)
    model.load(args.model_dir)

    eval_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=model_config.timesteps, prediction_type="sample")
    dataset = TextDataset(args.data_dir, split="test", tokenizer=model.tokenizer, device=device)

    metrics = ["BLEU", "ROUGE", "sentence-BLEU"]

    out = eval_metric(
        model,
        dataset,
        metrics,
        eval_scheduler,
        nsteps=args.nsteps,
        batch_size=args.batch,
        clamping_trick=args.clamping_trick
    )
    print("Score:", out)
