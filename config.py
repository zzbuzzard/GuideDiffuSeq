from dataclasses import dataclass


@dataclass
class ModelConfig:
    embed_mode: str = "bert"  # 'bert' or 'learned'
    dim: int = 768
    nhead: int = 12
    layers_encoder: int = 12
    layers_decoder: int = 12
    max_len: int = 128
    timesteps: int = 1000


@dataclass
class TrainingConfig:
    model: ModelConfig
    output_dir: str
    data_dir: str
    lr_warmup_steps: int = 100
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4
    save_model_epochs: int = 10
    sample_epochs: int = 1
    mixed_precision: str = "fp16"
    eval_nsteps: int = 50
    seed: int = 0
