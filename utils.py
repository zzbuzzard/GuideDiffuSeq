import torch
from torch_cluster import knn


def padding_mask(xs: torch.Tensor, lengths: torch.LongTensor):
    """
    Given a batch of embedded sequence data of shape (B x S x D) and the lengths (B) of each sequence,
    produces a padding mask of shape (B x S).
    """
    batch_size, max_seq_length, _ = xs.shape
    return lengths[:, None] < torch.arange(max_seq_length, device=lengths.device)[None]


def clamp_to_vocab(xs: torch.Tensor, vocab: torch.Tensor):
    assert xs.size(-1) == vocab.size(-1), f"Expected predictions (shape {xs.shape}) and vocabulary (shape {vocab.shape})" \
            "to have matching shape in the final dimension."

    # KNN implementation only supports 2D inputs
    if xs.dim() == 3:
        d1, d2, dim = xs.shape
        out = knn(vocab, xs.reshape((d1 * d2, dim)), k=1)[1]
        return out.reshape((d1, d2))

    return knn(vocab, xs, k=1)[1]
