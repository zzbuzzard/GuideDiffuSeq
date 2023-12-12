import argparse
from transformers import BertTokenizer
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from os.path import join
from abc import ABC, abstractmethod

from dataset import TextDataset


class LengthModel:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def fit(self, xs_lengths, ys_lengths):
        raise NotImplementedError

    @abstractmethod
    def predict(self, xs_lengths, ys_lengths):
        raise NotImplementedError

    def save(self, path):
        d = vars(self)
        pickle.dump(d, open(join(path, self.name), "wb"))

    def load(self, path):
        d = pickle.load(open(join(path, self.name), "rb"))
        for key, val in d.items():
            setattr(self, key, val)


class Oracle(LengthModel):
    """Returns the actual length of y"""
    def __init__(self):
        super().__init__("oracle_length_model")

    def fit(self, xs_lengths, ys_lengths):
        return

    def predict(self, _, ys_lengths: torch.LongTensor) -> torch.LongTensor:
        return ys_lengths

    def save(self, path): return
    def load(self, path): return


class UniformDiff(LengthModel):
    def __init__(self, low: int, hi: int):
        super().__init__(f"uniform_{low}_{hi}")
        self.low = low
        self.hi = hi

    def fit(self, xs_lengths, ys_lengths):
        return

    def predict(self, xs_lengths: torch.LongTensor, _) -> torch.LongTensor:
        min_length = 5
        xs_lengths = xs_lengths + torch.randint_like(xs_lengths, self.low, self.hi + 1)
        xs_lengths[xs_lengths < min_length] = min_length
        return xs_lengths

    def save(self, path): return
    def load(self, path): return


class Fixed(LengthModel):
    def __init__(self, n: int):
        super().__init__(f"fixed_{n}")
        self.n = n

    def fit(self, xs_lengths, ys_lengths):
        return

    def predict(self, xs_lengths: torch.LongTensor, _) -> torch.LongTensor:
        return xs_lengths * 0 + self.n

    def save(self, path): return
    def load(self, path): return


class NormalDist(LengthModel):
    """Models len(y) - len(x) as a Gaussian distribution."""
    def __init__(self):
        super().__init__("gaussian_length_model")
        self.mean = None
        self.std = None

    def fit(self, xs_lengths, ys_lengths):
        ds = np.array(ys_lengths) - np.array(xs_lengths)
        self.mean = np.mean(ds).item()
        self.std = np.std(ds).item()

    def predict(self, xs_lengths: torch.LongTensor, _) -> torch.LongTensor:
        min_length = 5

        y_minus_x = torch.randn(xs_lengths.shape, device=xs_lengths.device) * self.std + self.mean
        ys_lengths = xs_lengths.to(torch.float32) + y_minus_x
        ys_lengths[ys_lengths < min_length] = min_length
        return ys_lengths.to(torch.long)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", required=True, help="Path to data directory.")
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TextDataset(args.data_dir, split="train", tokenizer=tokenizer, device="cpu")
    inps = [tokenizer.decode(dataset[i][0]) for i in range(len(dataset))]
    gts = [tokenizer.decode(dataset[i][1]) for i in range(len(dataset))]

    Il = np.array([len(i) for i in inps])
    Gl = np.array([len(i) for i in gts])

    dist = NormalDist()
    dist.fit(Il, Gl)
    dist.save(args.data_dir)

    diff = Gl - Il
    plt.hist(diff, bins=200)
    plt.show()

    ord = np.argsort(Il)
    xs = np.arange(len(Il))
    plt.scatter(xs, Il[ord])
    plt.scatter(xs, Gl[ord])
    plt.show()

    pass
