from dataclasses import dataclass
import json
from torch.utils.data import DataLoader
import torch
from training.dataset import LibriTTSDataModule
from vocex import Vocex
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar

libritts = LibriTTSDataModule(
    "/dev/shm/metts",
    batch_size=16,
    num_workers=16
)
model = Vocex()
wandb_logger = WandbLogger(project="vocex")
trainer = pl.Trainer(
    accelerator="tpu",
    devices=8,
    logger=wandb_logger,
    max_epochs=100,
    callbacks=[TQDMProgressBar(refresh_rate=10)],
)
trainer.fit(model, libritts)