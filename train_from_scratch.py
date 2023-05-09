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
    batch_size=8,
    num_workers=48,
)
model = Vocex(
    measure_nlayers=8,
    dvector_nlayers=4,
    depthwise=True,
    noise_factor=0,
    lr=1e-5,
)
libritts.setup("scalers")
model.fit_scalers(libritts.train_dataloader(), 100)
wandb_logger = WandbLogger(project="consistency_model")
trainer = pl.Trainer(
    accelerator="tpu",
    devices=1,
    logger=wandb_logger,
    max_epochs=10,
    log_every_n_steps=100,
    val_check_interval=10_000,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            "checkpoints",
            every_n_train_steps=1000,
        ),
        TQDMProgressBar(refresh_rate=20),
    ],
)
trainer.fit(model, libritts)