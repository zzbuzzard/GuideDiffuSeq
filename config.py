from dataclasses import dataclass


@dataclass
class ModelConfig:
    dim = 768
    nhead = 12
    layers_encoder = 12
    layers_decoder = 12
    max_len = 128
    timesteps = 1000


@dataclass
class TrainingConfig:
    model: ModelConfig
    output_dir: str
    data_dir: str
    lr_warmup_steps = 100
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    save_model_epochs = 10
    sample_epochs = 1
    mixed_precision = "fp16"
    eval_nsteps = 50
    seed = 0
