import os
from os.path import join
import torch
from torch_cluster import knn
import torch.nn.functional as F
from torch import optim
import json
from config import ModelConfig


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
        print(f"Cannot load training data from {train_path}")
        return 0, {}

    print(f"Loading from {root_path}")
    d = torch.load(train_path)
    opt.load_state_dict(d["opt"])
    return d["epoch"], d["data"]
