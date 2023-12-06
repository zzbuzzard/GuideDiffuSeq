import torch
import torch.nn.functional as F
import torch.utils.data as dutils
from torch import optim
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, get_cosine_schedule_with_warmup
from dataclasses import asdict
import wandb

from config import TrainingConfig, ModelConfig
from model import Model, from_config
from dataset import TextDataset, collate
from utils import masked_loss, padding_mask

device = torch.device("cuda")

# TODO Load from json or cmd line or something
model_config = ModelConfig(
    layers_decoder=6,
    layers_encoder=6,
    nhead=2
)
config = TrainingConfig(
    model_config,
    output_dir="out",
    data_dir="datasets/QQP",
    batch_size=4
)

wandb.init(
    project="var-len-diffu-seq",
    name="Initial run",
    config=asdict(config),
    tags=["EMBED_MODE"]
)


def train_loop(config, model: Model, optimizer, train_dataloader, lr_scheduler):
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.model.timesteps, prediction_type="sample")
    eval_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=config.model.timesteps, prediction_type="sample")

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (xs_tok, ys_tok, xs_l, ys_l) in enumerate(train_dataloader):
            xs_emb = model.embed(xs_tok)
            ys_emb = model.embed(ys_tok)

            noise = torch.randn(ys_emb.shape, device=ys_emb.device)
            bs = xs_emb.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, config.model.timesteps, (bs,), device=ys_emb.device,
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

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            wandb.log(logs)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if epoch % config.save_model_epochs == 0:
                # TODO Save model
                # print("Not saving")
                pass

            # Show example output on final batch
            if epoch % config.sample_epochs == 0:
                toks = model.inference(xs_emb, xs_l, ys_l, eval_scheduler, config.eval_nsteps)

                log_tbl = wandb.Table(columns=["Steps", "Input", "Ground Truth", "Output"])

                for ind, t in enumerate(toks):
                    inp = xs_tok[ind][:xs_l[ind]]
                    gt = ys_tok[ind][:ys_l[ind]]
                    out = toks[ind]

                    inp = model.tokenizer.decode(inp)
                    gt = model.tokenizer.decode(gt)
                    out = model.tokenizer.decode(out)

                    log_tbl.add_data(global_step, inp, gt, out)

                    # tqdm.write("INPUT: " + inp)
                    # tqdm.write("GT: " + gt)
                    # tqdm.write("OUTPUT: " + out)
                    # tqdm.write("\n")

                wandb.log({"samples": log_tbl})


model = from_config(model_config).to(device)

dataset = TextDataset(config.data_dir, split="test", tokenizer=model.tokenizer, device=device)
train_dataloader = dutils.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate,
                                     drop_last=True)

optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

train_loop(
    config=config,
    model=model,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    lr_scheduler=lr_scheduler
)

wandb.finish()
