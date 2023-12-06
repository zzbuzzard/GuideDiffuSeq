import torch
import torch.nn.functional as F
import torch.utils.data as dutils
from torch import optim
from torch.optim import lr_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
from os.path import join
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, get_cosine_schedule_with_warmup
from dataclasses import asdict
import wandb
import argparse

from config import TrainingConfig, ModelConfig
from model import Model, from_config
from dataset import TextDataset, collate
from utils import masked_loss, padding_mask, load_state, save_state
from eval import eval_metric

device = torch.device("cuda")


def train_loop(model_dir: str, train_config: TrainingConfig, model_config: ModelConfig, model: Model, optimizer,
               train_dataloader, lr_scheduler, val_dataset):
    noise_scheduler = DDPMScheduler(num_train_timesteps=model_config.timesteps, prediction_type="sample")
    eval_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=model_config.timesteps, prediction_type="sample")

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=train_config.mixed_precision,
        project_dir=join(model_dir, "logs"),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("train")

    start_epoch, train_data = load_state(model_dir, model, optimizer)
    lr_scheduler.last_epoch = start_epoch * len(train_dataloader)

    keys = ["loss", "lr", "BLEU_val", "ROUGE_val"]
    for i in keys:
        if i not in train_data:
            train_data[i] = {}

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = start_epoch * len(train_dataloader)

    for epoch in range(start_epoch, train_config.num_epochs):
        # progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {epoch}")
        print("Epoch", epoch)

        tot_loss = 0
        loss_count = 0

        for step, (xs_tok, ys_tok, xs_l, ys_l) in enumerate(train_dataloader):
            xs_emb = model.embed(xs_tok)
            ys_emb = model.embed(ys_tok)

            noise = torch.randn(ys_emb.shape, device=ys_emb.device)
            bs = xs_emb.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, model_config.timesteps, (bs,), device=ys_emb.device,
                dtype=torch.int64
            )

            # Add noise
            ys_noised = noise_scheduler.add_noise(ys_emb, noise, timesteps)

            with accelerator.accumulate(model):
                ys_pred = model.forward(xs_emb, ys_noised, xs_l, ys_l, timesteps)
                ys_mask = padding_mask(ys_emb, ys_l)
                loss = masked_loss(ys_emb, ys_pred, padding_mask=ys_mask)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            # progress_bar.update(1)
            # progress_bar.set_postfix(**logs)
            global_step += 1

            tot_loss += loss.detach().item()
            loss_count += 1

        wandb.log({"loss": tot_loss / loss_count, "lr": lr_scheduler.get_last_lr()[0], "step": global_step})
        train_data["loss"][epoch] = tot_loss / loss_count
        train_data["lr"][epoch] = lr_scheduler.get_last_lr()[0]

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if epoch % train_config.save_epochs == 0:
                save_state(model_dir, model, optimizer, epoch + 1, train_data)

            # Show example output on final batch
            if epoch % train_config.sample_epochs == 0:
                mx_num = 8
                toks = model.inference(xs_emb[:mx_num], xs_l[:mx_num], ys_l[:mx_num], eval_scheduler,
                                       train_config.eval_nsteps)

                log_tbl = wandb.Table(columns=["Steps", "Input", "Ground Truth", "Output"])

                for ind, t in enumerate(toks):
                    inp = xs_tok[ind][:xs_l[ind]]
                    gt = ys_tok[ind][:ys_l[ind]]
                    out = toks[ind]

                    inp = model.tokenizer.decode(inp)
                    gt = model.tokenizer.decode(gt)
                    out = model.tokenizer.decode(out)

                    log_tbl.add_data(global_step, inp, gt, out)

                    tqdm.write("INPUT: " + inp)
                    tqdm.write("GT: " + gt)
                    tqdm.write("OUTPUT: " + out)
                    tqdm.write("\n")

                wandb.log({"samples": log_tbl})

            if epoch % train_config.eval_epochs == 0:
                print("Evaluating...")
                out = eval_metric(model, val_dataset, ["BLEU", "ROUGE"], eval_scheduler,
                                  nsteps=train_config.eval_nsteps, batch_size=train_config.batch_size)
                print("Metrics:", out)
                train_data["BLEU_val"][epoch] = out["BLEU"]
                train_data["ROUGE_val"][epoch] = out["ROUGE"]

                wandb.log({"BLEU_val": out["BLEU"], "ROUGE_val": out["ROUGE"], "step": global_step})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", required=True, help="Path to model directory. Must contain "
                                                                 "config.json and train_config.json files.")
    args = parser.parse_args()

    model_config = ModelConfig.load(args.model_dir)
    train_config = TrainingConfig.load(args.model_dir)

    wandb.init(
        project="var-len-diffu-seq",
        name=f"{args.model_dir}",
        config=asdict(train_config),
        tags=[]
    )

    wandb.define_metric("step")
    wandb.define_metric("loss", step_metric="step")
    wandb.define_metric("lr", step_metric="step")
    wandb.define_metric("BLEU_val", step_metric="step")
    wandb.define_metric("ROUGE_val", step_metric="step")

    model = from_config(model_config).to(device)

    dataset = TextDataset(train_config.data_dir, split="train", tokenizer=model.tokenizer, device=device)
    val_dataset = TextDataset(train_config.data_dir, split="valid", tokenizer=model.tokenizer, device=device)
    train_dataloader = dutils.DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate,
                                         drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    scheduler = lr_scheduler.LinearLR(optimizer=optimizer,
                                      start_factor=1,
                                      end_factor=0.1,
                                      total_iters=len(train_dataloader) * train_config.num_epochs)

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=config.lr_warmup_steps,
    #     num_training_steps=(len(train_dataloader) * config.num_epochs),
    # )

    train_loop(
        model_dir=args.model_dir,
        train_config=train_config,
        model_config=model_config,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=scheduler,
        val_dataset=val_dataset
    )

    wandb.finish()
