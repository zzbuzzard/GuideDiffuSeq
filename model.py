import os
from os.path import join
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import math
from diffusers.schedulers import SchedulerMixin
from transformers import BertTokenizerFast
from tqdm import tqdm

from utils import padding_mask, clamp_to_vocab
from config import ModelConfig


class Model(nn.Module):
    # Constructor args correspond to the fields in config.ModelConfig
    def __init__(self, embed_mode: str, dim: int, internal_dim: int, nhead: int, layers_encoder: int,
                 layers_decoder: int, max_len: int, timesteps: int, tokenizer_mode: str, pos_embed_mode: str,
                 time_embed_mode: str):
        super().__init__()

        self.timesteps = timesteps
        self.dim = dim
        self.internal_dim = internal_dim
        self.pos_embed_mode = pos_embed_mode
        self.time_embed_mode = time_embed_mode

        if dim == internal_dim:
            self.up_proj = self.down_proj = nn.Identity()
        else:
            self.up_proj = nn.Linear(dim, internal_dim)
            self.down_proj = nn.Linear(internal_dim, dim)

        bert_model_name = 'bert-base-uncased'

        if tokenizer_mode == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        elif os.path.isfile(tokenizer_mode):
            self.tokenizer = BertTokenizerFast(tokenizer_file=tokenizer_mode)
        else:
            raise NotImplementedError(f"Invalid tokenizer_mode '{tokenizer_mode}'. Must be a path or 'bert'.")

        # Load BERT tokenizer and embeddings
        if embed_mode == 'bert':
            assert dim == 768, "When using BERT embeddings, dim must equal 768"
            assert tokenizer_mode == 'bert', "When using BERT embeddings you must use the BERT tokenizer"
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
            dim_feedforward=internal_dim * 4,
            batch_first=True
        )

        if pos_embed_mode == "fixed":
            # Generate positional encoding as in Attention Is All You Need
            #  (taken from PyTorch documentation)
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, internal_dim, 2) * (-math.log(10000.0) / internal_dim))
            pe = torch.zeros(1, max_len, internal_dim)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        elif pos_embed_mode == "learned":
            self.pos_embeds = nn.Embedding(max_len, embedding_dim=internal_dim)
        else:
            raise NotImplementedError(f"Unknown pos_embed_mode '{pos_embed_mode}'.")

        if time_embed_mode == "learned":
            self.time_map = nn.Sequential(
                nn.Linear(internal_dim, internal_dim * 4),
                nn.ReLU(),
                nn.Linear(internal_dim * 4, internal_dim)
            )
        elif time_embed_mode == "fixed":
            self.time_map = nn.Identity()
        else:
            raise NotImplementedError(f"Unknown time_embed_mode '{time_embed_mode}'.")

    def add_positional_encoding(self, seq):
        batch, seqlen, _ = seq.shape
        if self.pos_embed_mode == "fixed":
            return seq + self.pe[:, :seqlen]
        else:
            inds = torch.arange(seqlen, device=seq.device)
            return seq + self.pos_embeds(inds)[None]

    def add_time_encoding(self, seq, timesteps):
        # timesteps : vector of length B
        device = seq.device
        batch_size = seq.shape[0]

        div_term = torch.exp(torch.arange(0, self.internal_dim, 2, device=device) * (-math.log(10000.0) / self.internal_dim))
        inp = timesteps[:, None, None] * div_term  # shape (B x 1 x (internal_dim / 2))
        te = torch.zeros((batch_size, 1, self.internal_dim), device=device)
        te[:, :, 0::2] = torch.sin(inp)
        te[:, :, 1::2] = torch.cos(inp)

        # Apply learned (or identity) map
        te = self.time_map(te)  # shape (B x 1 x internal_dim)

        return seq + te

    def _preprocess_cond(self, xs, xs_lengths):
        """Generates padding mask, applies up_proj and adds positional encoding."""
        xs_pad = padding_mask(xs, xs_lengths)
        xs = self.up_proj(xs)
        xs = self.add_positional_encoding(xs)
        return xs, xs_pad

    def _preprocess_tgt(self, ys, ys_lengths, timesteps):
        """Generates padding mask, applies up_proj and adds positional + time encodings."""
        ys_pad = padding_mask(ys, ys_lengths)

        ys = self.up_proj(ys)
        ys = self.add_positional_encoding(ys)
        ys = self.add_time_encoding(ys, timesteps)

        return ys, ys_pad

    def forward_with_encoder_output(self, encoder_output, xs_pad, ys, ys_pad, timesteps):
        """
        Computes `forward` when the encoder output has been pre-computed. Used for improving CFG efficiency.
        Additionally, takes ys_pad as an output to prevent recomputation at each step of inference.
        """
        ys = self.up_proj(ys)
        ys = self.add_positional_encoding(ys)
        ys = self.add_time_encoding(ys, timesteps)

        out = self.transformer.decoder.forward(
            tgt=ys,
            memory=encoder_output,
            tgt_key_padding_mask=ys_pad,
            memory_key_padding_mask=xs_pad
        )
        return self.down_proj(out)

    def forward(self, xs, ys, xs_lengths, ys_lengths, timesteps):
        """
        Applies model to a batch of inputs at given timesteps. Used during training.
        """
        xs, xs_pad = self._preprocess_cond(xs, xs_lengths)
        ys, ys_pad = self._preprocess_tgt(ys, ys_lengths, timesteps)

        out = self.transformer.forward(
            src=xs,
            tgt=ys,
            src_key_padding_mask=xs_pad,
            tgt_key_padding_mask=ys_pad,
            memory_key_padding_mask=xs_pad
        )
        return self.down_proj(out)

    def inference(self, xs, xs_lengths, ys_lengths, scheduler: SchedulerMixin, nsteps: int, clamping_trick: bool = False,
                  cfg: float = 1, cfg_lerp: bool = False):
        """
        Carries out the complete denoising inference procedure for the given token inputs `xs`.
        Note: `xs` is expected to be embedded and padded, as during training.
        Requires the lengths `ys_lengths` of the outputs to be specified, then generates `ys`.
        """
        with torch.no_grad():
            batch_size, _, dim = xs.shape
            assert dim == self.dim, f"Condition has dimension {dim}, expected {self.dim}."
            max_ys_len = torch.max(ys_lengths).item()
            ys_shape = (batch_size, max_ys_len, dim)
            # Construct ys_T from Gaussian noise
            ys = torch.randn(ys_shape, device=xs.device)

            # The encoder output does not change during inference, so may be precomputed here
            xs, xs_pad = self._preprocess_cond(xs, xs_lengths)
            encoder_out = self.transformer.encoder.forward(
                src=xs,
                src_key_padding_mask=xs_pad
            )

            if cfg != 1:
                # Precompute the encoder output in the masked case too
                uncond_xs = torch.tensor([self.tokenizer.mask_token_id] * batch_size, device=xs.device, dtype=torch.long)[:, None]
                uncond_xs = self.embed(uncond_xs)
                uncond_xs_lengths = torch.tensor([1] * batch_size, device=xs.device, dtype=torch.long)
                uncond_xs, uncond_xs_pad = self._preprocess_cond(uncond_xs, uncond_xs_lengths)
                uncond_encoder_out = self.transformer.encoder.forward(
                    src=uncond_xs,
                    src_key_padding_mask=uncond_xs_pad
                )

            # Finally, precompute ys_pad as this does not change during inference
            ys_pad = padding_mask(ys, ys_lengths)

            # Set step values
            scheduler.set_timesteps(nsteps)

            # Run denoising process
            for t in scheduler.timesteps:
                timesteps = torch.stack([t] * batch_size).to(xs.device)
                # model_output = self.forward(xs, ys, xs_lengths, ys_lengths, timesteps)
                model_output = self.forward_with_encoder_output(encoder_out, xs_pad, ys, ys_pad, timesteps)

                if cfg != 1:
                    # model_output_uncond = self.forward(uncond_xs, ys, uncond_xs_lengths, ys_lengths, timesteps)
                    # Compute unconditional prediction
                    model_output_uncond = self.forward_with_encoder_output(uncond_encoder_out, uncond_xs_pad, ys, ys_pad, timesteps)
                    # Apply CFG
                    scale = cfg if not cfg_lerp else cfg * t / self.timesteps
                    model_output = model_output_uncond + scale * (model_output - model_output_uncond)

                if clamping_trick:
                    model_output = clamp_to_vocab(model_output, self.embed.weight)
                    model_output = self.embed(model_output)

                ys = scheduler.step(model_output, t, ys).prev_sample

            # Clamp ys to the vocab
            tokens = clamp_to_vocab(ys, self.embed.weight)

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
