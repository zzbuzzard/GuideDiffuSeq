from dataclasses import dataclass
from os import path
import json


@dataclass
class ModelConfig:
    embed_mode: str = "learned"  # 'bert' or 'learned'
    tokenizer_mode: str = "bert"  # 'bert' or a path to a tokenizer
    pos_embed_mode: str = "fixed"  # 'fixed' or 'learned'

    dim: int = 768
    internal_dim: int = 768
    nhead: int = 12
    layers_encoder: int = 12
    layers_decoder: int = 12
    max_len: int = 128
    timesteps: int = 1000

    @staticmethod
    def load(root_path: str):
        """
        Loads from root_path/config.json.
        """
        jsonpath = path.join(root_path, "config.json")
        assert path.isdir(root_path), f"Model directory '{root_path}' not found!"
        assert path.isfile(jsonpath), f"config.json not found in directory '{root_path}'."

        with open(jsonpath, "r") as file:
            data = json.loads(file.read())

        return ModelConfig(**data)


@dataclass
class TrainingConfig:
    data_dir: str
    batch_size: int
    num_epochs: int

    save_epochs: int
    eval_epochs: int
    sample_epochs: int

    learning_rate: float = 1e-4
    eval_nsteps: int = 30
    seed: int = 0
    mixed_precision: str = "fp16"
    compile: bool = False

    @staticmethod
    def load(root_path: str):
        """
        Loads from root_path/train_config.json.
        """
        jsonpath = path.join(root_path, "train_config.json")
        assert path.isdir(root_path), f"Model directory '{root_path}' not found!"
        assert path.isfile(jsonpath), f"train_config.json not found in directory '{root_path}'."

        with open(jsonpath, "r") as file:
            data = json.loads(file.read())

        return TrainingConfig(**data)

