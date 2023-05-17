from dataclasses import dataclass

@dataclass
class Args:
    # data loading
    dataset: str = "cdminix/libritts-aligned"
    train_split: str = "train"
    eval_split: str = "dev"
    speaker2idx: str = "training/data/speaker2idx.json"
    phone2idx: str = "training/data/phone2idx.json"
    num_workers: int = 96
    prefetch_factor: int = 2
    fit_scalers: bool = True
    fit_scalers_steps: int = 10_000
    # model
    measure_nlayers: int = 8
    dvector_nlayers: int = 4
    depthwise: bool = False
    noise_factor: float = 0.1
    filter_size: int = 256
    kernel_size: int = 3
    dropout: float = 0.1
    # training
    measures: str = "energy,pitch,srmr,snr"
    max_epochs: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 5000
    weight_decay: float = 0.01
    log_every: int = 500
    eval_every: int = 5000
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    gradient_sync_every: int = 100
    bf16: bool = False
    resume_from_checkpoint: str = None
    strict_load: bool = False
    max_grad_norm: float = 1.0
    train_loss_logging_sum_steps: int = 100
    # wandb
    wandb_project: str = "consistency_model"
    wandb_run_name: str = None
    wandb_mode: str = "online"