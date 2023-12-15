from dataclasses import dataclass
from os import path
import json
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler
from torch.optim.lr_scheduler import LinearLR

from length_model import LengthModel, Oracle, NormalDist, UniformDiff, Fixed
from utils import get_named_beta_schedule, get_cosine_schedule_with_warmup


@dataclass
class ModelConfig:
    embed_mode: str = "learned"  # 'bert' or 'learned'
    tokenizer_mode: str = "bert"  # 'bert' or a path to a tokenizer
    pos_embed_mode: str = "fixed"  # 'fixed' or 'learned'
    time_embed_mode: str = "fixed"  # 'fixed' or 'learned'

    noise_schedule: str = "linear"

    dim: int = 768
    internal_dim: int = 768
    nhead: int = 12
    layers_encoder: int = 12
    layers_decoder: int = 12
    max_len: int = 128
    timesteps: int = 1000

    def get_betas(self):
        return get_named_beta_schedule(self.noise_schedule, self.timesteps)

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
    lr_schedule: str = "linear"

    importance_sampling: bool = False
    normalise_embeds: bool = False
    anchor_loss: bool = False

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

    def get_lr_scheduler(self, optimizer, total_steps, warmup_steps=2000):
        if self.lr_schedule == "linear":
            return LinearLR(optimizer=optimizer,
                            start_factor=1,
                            end_factor=self.learning_rate_final_mul,
                            total_iters=total_steps)
        elif self.lr_schedule == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            raise NotImplementedError(f"Unknown lr schedule '{self.lr_schedule}'.")


@dataclass
class EvalConfig:
    scheduler: str = "DPM++"  # DPM++ or DDIM
    nsteps: int = 30
    cfg: float = 1  # CFG=1 is equivalent to not using CFG
    cfg_lerp: bool = False
    clamp: bool = False
    clamp_lerp: bool = False
    length_model: str = "oracle"

    def get_path(self):
        # e.g. 'cfg=3.5_steps=10_clamp_DDIM'
        s = []
        if self.cfg != 1:
            extra = "_lerp" if self.cfg_lerp else ""
            s.append(f"cfg{extra}={self.cfg:.2f}")
        s.append(f"steps={self.nsteps}")
        if self.clamp:
            extra = "_lerp" if self.clamp_lerp else ""
            s.append("clamp-lerp" if self.clamp_lerp else "clamp")
        if self.scheduler != "DPM++":
            s.append(self.scheduler)
        if self.length_model != "oracle":
            s.append(self.length_model)
        return "_".join(s)

    def get_scheduler(self, model_config: ModelConfig):
        betas = model_config.get_betas()
        if self.scheduler == "DPM++":
            return DPMSolverMultistepScheduler(model_config.timesteps, prediction_type="sample", trained_betas=betas)
        elif self.scheduler == "DDIM":
            return DDIMScheduler(model_config.timesteps, prediction_type="sample", trained_betas=betas)
        else:
            raise NotImplementedError(f"Unknown scheduler '{self.scheduler}'. Note: it should be easy to add support"
                                      f"for a new diffusers scheduler by adding it to EvalConfig, so long as that"
                                      f"scheduler has support for sample prediction_type mode.")

    def get_length_model(self) -> LengthModel:
        if self.length_model == "oracle":
            return Oracle()
        elif self.length_model == "normal":
            return NormalDist()
        elif self.length_model.startswith("uniform"):
            _, low, hi = self.length_model.split("_")
            return UniformDiff(int(low), int(hi))
        elif self.length_model.startswith("fixed"):
            _, n = self.length_model.split("_")
            return Fixed(int(n))
        else:
            raise NotImplementedError(f"Unknown length model '{self.length_model}'.")
