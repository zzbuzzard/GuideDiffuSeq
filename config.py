from dataclasses import dataclass
from os import path
import json
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler


@dataclass
class ModelConfig:
    embed_mode: str = "learned"  # 'bert' or 'learned'
    tokenizer_mode: str = "bert"  # 'bert' or a path to a tokenizer
    pos_embed_mode: str = "fixed"  # 'fixed' or 'learned'
    time_embed_mode: str = "fixed"  # 'fixed' or 'learned'

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
    learning_rate_final_mul: float = 0.1
    uncond_prob: float = 0.0

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


@dataclass
class EvalConfig:
    scheduler: str = "DPM++"  # DPM++ or DDIM
    nsteps: int = 30
    cfg: float = 1  # CFG=1 is equivalent to not using CFG
    cfg_lerp: bool = False
    clamp: bool = False

    def get_path(self):
        # e.g. 'cfg=3.5_steps=10_clamp_DDIM'
        s = []
        if self.cfg != 1:
            extra = "_lerp" if self.cfg_lerp else ""
            s.append(f"cfg{extra}={self.cfg:.2f}")
        s.append(f"steps={self.nsteps}")
        if self.clamp:
            s.append("clamp")
        if self.scheduler != "DPM++":
            s.append(self.scheduler)
        return "_".join(s)

    def get_scheduler(self, train_timesteps: int):
        if self.scheduler == "DPM++":
            return DPMSolverMultistepScheduler(train_timesteps, prediction_type="sample")
        elif self.scheduler == "DDIM":
            return DDIMScheduler(train_timesteps, prediction_type="sample")
        else:
            raise NotImplementedError(f"Unknown scheduler '{self.scheduler}'. Note: it should be easy to add support"
                                      f"for a new diffusers scheduler by adding it to EvalConfig, so long as that"
                                      f"scheduler has support for sample prediction_type mode.")

