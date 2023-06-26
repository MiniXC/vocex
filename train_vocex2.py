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
#import torch_xla.debug.profiler as xp

from vocex import Vocex, Vocex2Model
from training.arguments import Vocex2Args
from vocex.speaker_loss import SpeakerLoss
from .augmentations import wave_augmentation_func, mel_augmentation_func

def save_model(path, accelerator, model):
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(
        unwrapped_model.state_dict(),
        path,
    )

def eval_model(model, eval_dl, speaker_loss, measure_loss, vocex_model, args):
    model.eval()
    loss_dict = {
        "eval/loss": [],
        "eval/speaker_loss": [],
        **{
            f"eval/{m}_loss": []
            for m in args.measures
        },
    }
    speaker_preds = []
    with torch.no_grad():
        for batch in tqdm(eval_dl, desc="evaluating"):
            vocex_outputs = vocex_model(
                mel=batch["mel"],
                inference=True,
            )
            measures = {
                m: vocex_model.scalers[m].transform(vocex_outputs["measures"][m])
                if m != "voice_activity_binary"
                else vocex_outputs["measures"][m]
                for m in args.measures
            }
            speaker = batch["speaker"]
            augmented_mel = batch["augmented_mel"]
            outputs = model(
                mel=augmented_mel,
            )
            speaker_loss_val = speaker_loss(
                outputs["speaker_embedding"],
                speaker,
            )
            measure_loss_val = {
                m: measure_loss(
                    outputs["measures"][m],
                    measures[m],
                )
                for m in args.measures
            }
            loss = (speaker_loss_val + sum(measure_loss_val.values())) / (len(measure_loss_val) + 1)
            loss_dict["eval/loss"].append(loss.item())
            loss_dict["eval/speaker_loss"].append(speaker_loss_val.item())
            for m in args.measures:
                loss_dict[f"eval/{m}_loss"].append(measure_loss_val[m].item())
    model.train()
    wandb.log({
        k: np.mean(v)
        for k, v in loss_dict.items()
    })
    print({
        k: round(np.mean(v), 4)
        for k, v in loss_dict.items()
    })


def main():
    parser = HfArgumentParser([Vocex2Args])

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
        
    args.measures = args.measures.split(",")

    if args.bf16:
        os.environ["ACCELERATE_DOWNCAST_BF16"] = "true"
    accelerator = Accelerator(split_batches=True, mixed_precision="bf16" if args.bf16 else None)

    if accelerator.is_main_process:
        wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_project,
            mode=args.wandb_mode,
        )
        wandb.config.update(args)

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

    vocex_model = Vocex.from_pretrained(args.pretrained_vocex).model


    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator.collate_fn,
        num_workers=args.num_workers,
    )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator.collate_fn,
    )

    model = Vocex2Model(
        nlayers=args.nlayers,
        depthwise=args.depthwise,
        filter_size=args.filter_size,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        speaker_emb_dim=args.speaker_embedding_size,
    )

    print("Model parameters (M):", sum(p.numel() for p in model.parameters()) / 1_000_000)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    num_epochs = args.max_epochs
    num_training_steps = num_epochs * len(train_dl)
    steps_per_epoch = len(train_dl)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    if accelerator.is_main_process:
        progress_bar = tqdm(range(num_training_steps), desc=f"training (rank {accelerator.process_index}, device {accelerator.device})")


    train_dl, eval_dl, model, optimizer, lr_scheduler, vocex_model = accelerator.prepare(
        train_dl, eval_dl, model, optimizer, lr_scheduler, vocex_model
    )

    model.train()

    step = 0

    # categorical cross entropy for speaker classification
    speaker_loss = SpeakerLoss(
        hidden_size=args.speaker_embedding_size,
        num_speakers=len(speaker2idx),
    )
    speaker_loss = accelerator.prepare(speaker_loss)
    # mean squared error for measures
    measure_loss = torch.nn.MSELoss()

    if accelerator.is_main_process:
        overall_losses = deque(maxlen=100)
        speaker_losses = deque(maxlen=100)
        measure_losses = {
            m: deque(maxlen=100)
            for m in args.measures
        }
    
    for epoch in range(num_epochs):
        for batch in train_dl:
            measures = None
            with torch.no_grad():
                vocex_outputs = vocex_model(
                    mel=batch["mel"],
                    inference=True,
                )
                measures = {
                    m: vocex_model.scalers[m].transform(vocex_outputs["measures"][m])
                    if m != "voice_activity_binary"
                    else vocex_outputs["measures"][m]
                    for m in args.measures
                }
            speaker = batch["speaker"]
            augmented_mel = batch["augmented_mel"]

            if (step + 1) % args.gradient_sync_every == 0:
                outputs = model(
                    mel=augmented_mel,
                )
                speaker_loss_val = speaker_loss(
                    outputs["speaker_embedding"],
                    speaker,
                )
                measure_loss_val = {
                    m: measure_loss(
                        outputs["measures"][m],
                        measures[m],
                    )
                    for m in args.measures
                }
                loss = (speaker_loss_val + sum(measure_loss_val.values())) / (len(measure_loss_val) + 1)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            else:
                with accelerator.no_sync(model):
                    outputs = model(
                        mel=augmented_mel,
                    )
                    speaker_loss_val = speaker_loss(
                        outputs["speaker_embedding"],
                        speaker,
                    )
                    measure_loss_val = {
                        m: measure_loss(
                            outputs["measures"][m],
                            measures[m],
                        )
                        for m in args.measures
                    }
                    loss = (speaker_loss_val + sum(measure_loss_val.values())) / (len(measure_loss_val) + 1)
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.is_main_process:
                overall_losses.append(loss)
                speaker_losses.append(speaker_loss_val)
                for m in args.measures:
                    measure_losses[m].append(measure_loss_val[m])

            if (step + 1) % args.log_every == 0 and accelerator.is_main_process:
                loss_dict = {
                    "train/loss": sum(overall_losses).item() / len(overall_losses),
                    "train/speaker_loss": sum(speaker_losses).item() / len(speaker_losses),
                    **{
                        f"train/{m}_loss": sum(measure_losses[m]).item() / len(measure_losses[m])
                        for m in args.measures
                    },
                    "train/epoch": epoch + (step - steps_per_epoch * epoch) / steps_per_epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                wandb.log(loss_dict, step=step)
                print({
                    k: round(v, 4)
                    for k, v in loss_dict.items()
                })
                overall_losses.clear()
                speaker_losses.clear()
                for m in args.measures:
                    measure_losses[m].clear()
                
            if (step + 1) % args.eval_every == 0 and accelerator.is_main_process:
                eval_model(model, eval_dl, speaker_loss, measure_loss, vocex_model, args)

            if (step + 1) % args.save_every == 0:
                if not os.path.exists(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir)
                save_model(
                    os.path.join(args.checkpoint_dir, f"vx2-model-{step+1}.pt"),
                    accelerator,
                    model,
                )

            step += 1
            # update progress bar while using process_index as tqdm desc
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_description(f"training (rank {accelerator.process_index}, device {accelerator.device})")


if __name__ == "__main__":
    main()

