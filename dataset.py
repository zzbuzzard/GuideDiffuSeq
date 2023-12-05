import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import os.path
import json
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, root_path: str, split: str, tokenizer, device):
        self.src = []
        self.trg = []

        self.tokenizer = tokenizer
        self.device = device

        path = os.path.join(root_path, f"{split}.jsonl")

        def process(text: str):
            ids = tokenizer(text.strip()).input_ids
            return torch.LongTensor(ids)

        with open(path, 'r') as f_reader:
            for row in tqdm(f_reader, desc=f"Tokenizing {split} split"):
                content = json.loads(row)
                self.src.append(process(content['src']))
                self.trg.append(process(content['trg']))

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        """Returns (src, trg, srclen, trglen)"""
        x = self.src[item].to(self.device)
        y = self.trg[item].to(self.device)
        xl = torch.tensor(x.size(0), dtype=torch.long, device=self.device)
        yl = torch.tensor(y.size(0), dtype=torch.long, device=self.device)
        return x, y, xl, yl


def collate(batch):
    xs, ys, xlens, ylens = zip(*batch)
    xs = pad_sequence(xs, batch_first=True)
    ys = pad_sequence(ys, batch_first=True)
    return xs, ys, torch.stack(xlens), torch.stack(ylens)
