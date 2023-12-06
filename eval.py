import torch
import torch.utils.data as dutils
from os.path import join
from diffusers import DPMSolverMultistepScheduler
import argparse
from torchmetrics.text import BLEUScore, ROUGEScore
from typing import List

from config import ModelConfig
from model import Model, from_config
from dataset import TextDataset, collate

device = torch.device("cuda")


def eval_metric(model: Model, dataset: TextDataset, metric_names: List[str], eval_scheduler, nsteps: int, batch_size: int = 64):
    def get_metric(name):
        name = name.upper()
        if name == "BLEU":
            return BLEUScore().to(device)
        elif name == "ROUGE":
            m = ROUGEScore().to(device)
            return lambda a, b: m(a, b)["rougeL_fmeasure"]
        else:
            raise NotImplementedError(f"Unsupported metric '{name}'. Must be BLEU or ROUGE.")

    metrics = {name: get_metric(name) for name in metric_names}
    scores = {name: 0 for name in metric_names}
    total = 0

    dl = dutils.DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate)

    with torch.no_grad():
        for step, (xs_tok, ys_tok, xs_l, ys_l) in enumerate(dl):
            xs_emb = model.embed(xs_tok)
            toks = model.inference(xs_emb, xs_l, ys_l, eval_scheduler, nsteps)

            for ind, t in enumerate(toks):
                inp = xs_tok[ind][:xs_l[ind]]
                gt = ys_tok[ind][:ys_l[ind]]
                out = toks[ind]

                inp = model.tokenizer.decode(inp)
                gt = model.tokenizer.decode(gt)
                out = model.tokenizer.decode(out)

                for name in metric_names:
                    scores[name] += metrics[name]([out], [[gt]])
                total += 1

    return {name: float(scores[name] / total) for name in metric_names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", required=True, help="Path to model directory. Must contain "
                                                                 "config.json and train_config.json files.")
    parser.add_argument("-d", "--data-dir", required=True, help="Path to data directory.")
    parser.add_argument("-n", "--nsteps", type=int, default=50, help="Number of steps to use for sampling")
    parser.add_argument("-b", "--batch", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    model_config = ModelConfig.load(args.model_dir)
    model = from_config(model_config).to(device)
    model.load(args.model_dir)

    eval_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=model_config.timesteps, prediction_type="sample")
    dataset = TextDataset(args.data_dir, split="test", tokenizer=model.tokenizer, device=device)

    out = eval_metric(
        model,
        dataset,
        ["BLEU", "ROUGE"],
        eval_scheduler,
        nsteps=50,
        batch_size=64
    )
    print("Score:", out)
