import json
import sys
import random
import os

from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from speech_collator import SpeechCollator
from transformers import HfArgumentParser
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from transformers import get_linear_schedule_with_warmup
from copy import deepcopy

from vocex import Vocex, Vocex2Model
from training.arguments import Vocex2Args

def wave_augmentation_func(wave):
    pass

def mel_augmentation_func(mel):
    pass

def main():
    parser = HfArgumentParser([Vocex2Args])

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
        
    args.measures = args.measures.split(",")

    wandb.init(
        name=args.wandb_run_name,
        project=args.wandb_project,
        mode=args.wandb_mode,
    )
    wandb.config.update(args)

    accelerator = Accelerator()

    with accelerator.main_process_first():
        libritts = load_dataset(args.dataset)

    train_ds = libritts[args.train_split].shuffle(seed=42)
    eval_ds = libritts[args.eval_split]

    speaker2idx = json.load(open(args.speaker2idx))
    phone2idx = json.load(open(args.phone2idx))

    collator = SpeechCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        return_keys=[
            "augmented_mel",
            "mel",
            "speaker",
        ],
        overwrite_max_length=True,
        wave_augmentation_func=wave_augmentation_func,
        mel_augmentation_func=mel_augmentation_func,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    vocex_model = Vocex.from_pretrained(args.pretrained_vocex).model

    target_measures = args.measures.split(",")

    model = Vocex2Model(
        frame_nlayers=args.frame_nlayers,
        utt_nlayers=args.utt_nlayers,
        depthwise=args.depthwise,
        filter_size=args.filter_size,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    num_epochs = args.max_epochs
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps), desc="training", disable=not accelerator.is_local_main_process)

    train_dataloader, eval_dataloader, model, optimizer, lr_scheduler, vocex_model = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer, lr_scheduler, vocex_model
    )

    model.train()

    step = 0

    # categorical cross entropy for speaker classification
    speaker_loss = torch.nn.CrossEntropyLoss()
    # mean squared error for measures
    measure_loss = torch.nn.MSELoss()

    overall_losses = deque(maxlen=100)
    speaker_losses = deque(maxlen=100)
    measure_losses = {
        m: deque(maxlen=100)
        for m in target_measures
    }
    
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            vocex_outputs = vocex_model(
                mel=batch["mel"].to(accelerator.device),
                inference=True,
            )
            measures = {
                m: vocex_outputs["measures"][m]
                for m in target_measures
            }
            speaker = batch["speaker"].to(accelerator.device)
            augmented_mel = batch["augmented_mel"].to(accelerator.device)

            outputs = model(
                mel=augmented_mel,
            )

            speaker_loss_val = speaker_loss(
                outputs["speaker_logits"],
                speaker,
            )

            measure_loss_val = {
                m: measure_loss(
                    outputs["measures"][m],
                    measures[m],
                )
                for m in target_measures
            }

            loss = (speaker_loss_val + sum(measure_loss_val.values())) / (len(measure_loss_val) + 1)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            overall_losses.append(loss.item())
            speaker_losses.append(speaker_loss_val.item())
            for m in target_measures:
                measure_losses[m].append(measure_loss_val[m].item())

            if step % args.log_every == 0:
                loss_dict = {
                    "train/loss": torch.mean(overall_losses).item(),
                    "train/speaker_loss": torch.mean(speaker_losses).item(),
                    **{
                        f"train/{m}_loss": torch.mean(measure_losses[m]).item()
                        for m in target_measures
                    },
                }
                wandb.log(loss_dict, step=step)
                print({
                    k: round(v, 4)
                    for k, v in loss_dict.items()
                })
                overall_losses.clear()
                speaker_losses.clear()
                for m in target_measures:
                    measure_losses[m].clear()
                
            if step % args.eval_every == 0:
                pass

            if step % args.save_every == 0:
                pass

            step += 1
            progress_bar.update(1)


if __name__ == "__main__":
    main()

