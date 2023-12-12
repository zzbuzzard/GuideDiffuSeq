import os
from os.path import join
import torch
from torch_cluster import knn
import torch.nn.functional as F
from torch import optim
import math


def padding_mask(xs: torch.Tensor, lengths: torch.LongTensor):
    """
    Given a batch of embedded sequence data of shape (B x S x D) and the lengths (B) of each sequence,
    produces a padding mask of shape (B x S).
    """
    batch_size, max_seq_length, _ = xs.shape
    return torch.arange(max_seq_length, device=lengths.device)[None] >= lengths[:, None]


def clamp_to_vocab(xs: torch.Tensor, vocab: torch.Tensor):
    assert xs.size(-1) == vocab.size(-1), f"Expected predictions (shape {xs.shape}) and vocabulary (shape {vocab.shape})" \
            "to have matching shape in the final dimension."

    # KNN implementation only supports 2D inputs
    if xs.dim() == 3:
        d1, d2, dim = xs.shape
        out = knn(vocab, xs.reshape((d1 * d2, dim)), k=1)[1]
        return out.reshape((d1, d2))

    return knn(vocab, xs, k=1)[1]


def masked_loss(goal: torch.Tensor, pred: torch.Tensor, padding_mask: torch.Tensor):
    """Calculate MSE loss, excluding padding (padding_mask=True indicates padding)"""
    masked_goal = goal[~padding_mask]
    masked_pred = pred[~padding_mask]
    # Calculate MSE loss
    loss = F.mse_loss(masked_pred, masked_goal)
    return loss


def masked_loss_batched(goal: torch.Tensor, pred: torch.Tensor, padding_mask: torch.Tensor, lengths: torch.LongTensor):
    """
    Calculate MSE loss, excluding padding (padding_mask=True indicates padding).
    This version returns the loss per batch, as is necessary for importance sampling.
    """
    goal = torch.where(padding_mask.unsqueeze(2), 0, goal)
    pred = torch.where(padding_mask.unsqueeze(2), 0, pred)
    d = (goal - pred).pow(2)
    d = torch.mean(d, dim=2)  # Average over the embedding dimension
    d = torch.sum(d, dim=1) / lengths  # Average over the sequence - use lengths rather than mean to account for padding
    return d


def save_state(root_path: str, model, opt: optim.Optimizer, epoch: int, train_data: dict):
    train_path = join(root_path, "train_data.pt")

    print(f"Saving to {root_path}...")
    model.save(root_path)

    d = {
        "opt": opt.state_dict(),
        "epoch": epoch,
        "data": train_data
    }
    torch.save(d, train_path)


def load_state(root_path: str, model, opt: optim.Optimizer):
    """Loads the model and optimizer, returns (epoch, train_data)"""
    train_path = join(root_path, "train_data.pt")

    model.load(root_path)

    if not os.path.isfile(train_path):
        print(f"{train_path} not found, not loading training data")
        return 1, {}

    print(f"Loading from {root_path}")
    d = torch.load(train_path)
    opt.load_state_dict(d["opt"])
    return d["epoch"], d["data"]


# Taken from OpenAI/guided-diffusion
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.Tensor(betas)


def sqrt_noise_schedule(num_diffusion_timesteps: int):
    return betas_for_alpha_bar(
        num_diffusion_timesteps,
        lambda t: 1 - math.sqrt(t + 0.0001),
    )
