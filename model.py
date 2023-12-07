import os
from os.path import join
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import math
from diffusers.schedulers import SchedulerMixin
from tqdm import tqdm

from utils import padding_mask, clamp_to_vocab
from config import ModelConfig


class Model(nn.Module):
    def __init__(self, embed_mode: str, dim: int, internal_dim: int, nhead: int, layers_encoder: int,
                 layers_decoder: int, max_len: int):
        super().__init__()

        bert_model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        if dim == internal_dim:
            self.up_proj = self.down_proj = nn.Identity()
        else:
            self.up_proj = nn.Linear(dim, internal_dim)
            self.down_proj = nn.Linear(internal_dim, dim)

        # Load BERT tokenizer and embeddings
        if embed_mode == 'bert':
            assert internal_dim == 768, "When using BERT embeddings, internal_dim must equal 768"
            model = BertModel.from_pretrained(bert_model_name)
            self.embed = model.embeddings.word_embeddings
            del model  # (we do not need the rest of the BERT model)
        elif embed_mode == 'learned':
            # Random embeddings initialised from N(0, 1)
            self.embed = nn.Embedding(num_embeddings=self.tokenizer.vocab_size, embedding_dim=dim)
        else:
            raise NotImplementedError(f"Unknown embed_mode '{embed_mode}'.")

        self.transformer = nn.Transformer(
            d_model=internal_dim,
            nhead=nhead,
            num_decoder_layers=layers_decoder,
            num_encoder_layers=layers_encoder,
            dim_feedforward=dim * 4,
            batch_first=True
        )

        self.dim = dim

        # Generate positional encoding as in Attention Is All You Need
        #  (taken from PyTorch documentation)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def add_positional_encoding(self, seq):
        batch, seqlen, _ = seq.shape
        return seq + self.pe[:, :seqlen]

    def add_time_encoding(self, seq, timesteps):
        # timesteps : vector of length B
        device = seq.device
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device) * (-math.log(10000.0) / self.dim))
        inp = timesteps[:, None, None] * div_term  # shape B x 1 x (dim/2)
        seq[:, :, 0::2] = seq[:, :, 0::2] + torch.sin(inp)
        seq[:, :, 1::2] = seq[:, :, 1::2] + torch.cos(inp)
        return seq

    def forward(self, xs, ys, xs_lengths, ys_lengths, timesteps):
        """
        Applies model to a batch of inputs at given timesteps. Used during training.
        """
        xs_pad = padding_mask(xs, xs_lengths)
        ys_pad = padding_mask(ys, ys_lengths)

        xs = self.add_positional_encoding(xs)

        ys = self.add_positional_encoding(ys)
        ys = self.add_time_encoding(ys, timesteps)

        return self.transformer.forward(
            src=xs,
            tgt=ys,
            src_key_padding_mask=xs_pad,
            tgt_key_padding_mask=ys_pad
        )

    def inference(self, xs, xs_lengths, ys_lengths, scheduler: SchedulerMixin, nsteps: int, clamping_trick: bool = False):
        """
        Carries out the complete denoising inference procedure for the given token inputs `xs`.
        Note: `xs` is expected to be embedded and padded, as during training.
        Requires the lengths `ys_lengths` of the outputs to be specified, then generates `ys`.
        """
        torch.set_grad_enabled(False)
        # Construct ys_T from Gaussian noise
        batch_size, _, dim = xs.shape
        max_ys_len = torch.max(ys_lengths).item()
        ys_shape = (batch_size, max_ys_len, dim)
        ys = torch.randn(ys_shape, device=xs.device)

        # Set step values
        scheduler.set_timesteps(nsteps)

        # Run denoising process
        for t in scheduler.timesteps:
            timesteps = torch.stack([t] * batch_size).to(xs.device)
            model_output = self.forward(xs, ys, xs_lengths, ys_lengths, timesteps)

            if clamping_trick:
                model_output = clamp_to_vocab(model_output, self.embed.weight)
                model_output = self.embed(model_output)

            ys = scheduler.step(model_output, t, ys).prev_sample

        # Clamp ys to the vocab
        tokens = clamp_to_vocab(ys, self.embed.weight)

        torch.set_grad_enabled(True)

        # Trim parts not in the input
        # Note: it is inefficient to compute KNN for the padding, this could be improved
        return [toks[:ys_lengths[i]] for i, toks in enumerate(tokens)]

    def load(self, root_path: str):
        path = join(root_path, "model.pt")
        if not os.path.isfile(path):
            print(f"{path} not found, not loading state.")
        else:
            print("Loading model state from", path)
            self.load_state_dict(torch.load(path))

    def save(self, root_path: str):
        path = join(root_path, "model.pt")
        torch.save(self.state_dict(), path)


def from_config(config: ModelConfig):
    return Model(**vars(config))
