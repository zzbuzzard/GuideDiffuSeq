import torch
import torch.utils.data as dutils
from torch import optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
from diffusers import DDPMScheduler
from dataclasses import asdict
import wandb
import argparse
import os
from contextlib import nullcontext

from config import TrainingConfig, ModelConfig, EvalConfig
from model import Model, from_config
from dataset import TextDataset, collate
from utils import masked_loss, masked_loss_batched, padding_mask, load_state, save_state, sqrt_noise_schedule
from eval import eval_model
from importance_sampler import ImportanceSampler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
metrics = ["BLEU", "ROUGE", "sentence-BLEU"]


def train_loop(model_dir: str, train_config: TrainingConfig, model_config: ModelConfig, model: Model, optimizer,
               train_dataloader, lr_scheduler, val_dataset):
    def train(model, scaler, ys_noised, timesteps, xs_emb, ys_emb, xs_l, ys_l):
        ys_pred = model.forward(xs_emb, ys_noised, xs_l, ys_l, timesteps)
        ys_mask = padding_mask(ys_emb, ys_l)

        if train_config.importance_sampling:
            loss_batched = masked_loss_batched(ys_emb, ys_pred, padding_mask=ys_mask, lengths=ys_l)
            not_masked = xs_l > 1
            imp_sampler.register(timesteps[not_masked], loss_batched[not_masked])
            loss_scaled = imp_sampler.scale_losses(timesteps, loss_batched)
            loss = torch.mean(loss_scaled)
        else:
            loss_batched = masked_loss_batched(ys_emb, ys_pred, padding_mask=ys_mask, lengths=ys_l)
            loss = torch.mean(loss_batched)
            # loss = masked_loss(ys_emb, ys_pred, padding_mask=ys_mask)

        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        optimizer.zero_grad()
        return loss.item()

    if train_config.compile:
        train = torch.compile(train, dynamic=True)

    mask_token_id = model.tokenizer.mask_token_id

    mixed_precision = train_config.mixed_precision == "fp16"
    print(f"Mixed precision: {mixed_precision}")

    noise_scheduler = DDPMScheduler(num_train_timesteps=model_config.timesteps,
                                    prediction_type="sample",
                                    trained_betas=sqrt_noise_schedule(model_config.timesteps))  # custom sqrt schedule
    eval_config = EvalConfig(nsteps=train_config.eval_nsteps)
    eval_scheduler = eval_config.get_scheduler(model_config.timesteps)

    start_epoch, train_data = load_state(model_dir, model, optimizer)
    lr_scheduler.last_epoch = start_epoch * len(train_dataloader)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if train_config.importance_sampling:
        imp_sampler = ImportanceSampler(timesteps=model.timesteps, n=10, device=device)

    keys = ["loss", "lr"] + [i+"-val" for i in metrics]
    for i in keys:
        if i not in train_data:
            train_data[i] = {}

    global_step = start_epoch * len(train_dataloader)

    for epoch in range(start_epoch, train_config.num_epochs + 1):
        # progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {epoch}")
        print("Epoch", epoch)

        tot_loss = 0
        loss_count = 0

        for step, (xs_tok, ys_tok, xs_l, ys_l) in enumerate(tqdm(train_dataloader)):
            # Only activate autocast if it is needed
            with torch.autocast(device_type="cuda", dtype=torch.float16) if mixed_precision else nullcontext():
                batch_size = xs_tok.shape[0]

                # Change some conditions to a single [MASK] token to enable CFG
                with torch.no_grad():
                    uncond_xs = torch.zeros_like(xs_tok)
                    uncond_xs[:, 0] = mask_token_id
                    uncond_xs_l = torch.tensor([1] * batch_size, device=device, dtype=torch.long)
                    unconds = torch.rand((batch_size,), device=device) < train_config.uncond_prob

                    xs_tok = torch.where(unconds.unsqueeze(1), uncond_xs, xs_tok)
                    xs_l = torch.where(unconds, uncond_xs_l, xs_l)

                xs_emb = model.embed(xs_tok)
                ys_emb = model.embed(ys_tok)

                noise = torch.randn(ys_emb.shape, device=ys_emb.device)
                bs = xs_emb.shape[0]

                # Sample a random timestep for each image
                if train_config.importance_sampling:
                    timesteps = imp_sampler.sample(batch_size=bs)
                else:
                    timesteps = torch.randint(
                        1, model_config.timesteps, (bs,), device=ys_emb.device,
                        dtype=torch.int64
                    )

                # Add noise
                ys_noised = noise_scheduler.add_noise(ys_emb, noise, timesteps)

                loss = train(model, scaler, ys_noised, timesteps, xs_emb, ys_emb, xs_l, ys_l)

            tot_loss += loss
            global_step += 1
            loss_count += 1

        # if train_config.importance_sampling:
        #     imp_sampler.visualise()

        wandb.log({"loss": tot_loss / loss_count, "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch})
        train_data["loss"][epoch] = tot_loss / loss_count
        train_data["lr"][epoch] = lr_scheduler.get_last_lr()[0]

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if epoch % train_config.save_epochs == 0:
            save_state(model_dir, model, optimizer, epoch + 1, train_data)

        # Show example output on final batch of this epoch
        if epoch % train_config.sample_epochs == 0:
            mx_num = 8

            model.eval()
            toks = model.inference(xs_emb[:mx_num], xs_l[:mx_num], ys_l[:mx_num], eval_scheduler,
                                   train_config.eval_nsteps)
            model.train()

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
            model.eval()
            out = eval_model(model, val_dataset, metrics, config=eval_config, batch_size=train_config.batch_size)
            model.train()
            out = {i+"-val": out[i] for i in out}
            print("Metrics:", out)
            for i in out:
                train_data[i][epoch] = out[i]

            out["epoch"] = epoch

            wandb.log(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-dir", required=True, help="Path to model directory. Must contain "
                                                                 "config.json and train_config.json files.")
    args = parser.parse_args()

    model_config = ModelConfig.load(args.model_dir)
    train_config = TrainingConfig.load(args.model_dir)

    torch.manual_seed(train_config.seed)

    name = os.path.split(args.model_dir)[-1]
    wandb.init(
        project="var-len-diffu-seq-3",
        name=f"{name}",
        config=asdict(train_config),
        tags=[]
    )

    wandb.define_metric("epoch")
    wandb.define_metric("loss", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    for i in metrics:
        wandb.define_metric(i + "-val", step_metric="epoch")

    model = from_config(model_config).to(device)

    dataset = TextDataset(train_config.data_dir, split="train", tokenizer=model.tokenizer, device=device)
    val_dataset = TextDataset(train_config.data_dir, split="valid", tokenizer=model.tokenizer, device=device)
    train_dataloader = dutils.DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate,
                                         drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    scheduler = lr_scheduler.LinearLR(optimizer=optimizer,
                                      start_factor=1,
                                      end_factor=train_config.learning_rate_final_mul,
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
