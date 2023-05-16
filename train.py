import json
import sys

from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from speech_collator import SpeechCollator
from speech_collator.measures import EnergyMeasure, PitchMeasure, SRMRMeasure, SNRMeasure
from transformers import HfArgumentParser
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

from vocex import Vocex
from vocex.utils import NoamLR
from training.arguments import Args

MEASURE_DICT = {
    "energy": EnergyMeasure,
    "pitch": PitchMeasure,
    "srmr": SRMRMeasure,
    "snr": SNRMeasure,
}

def eval_loop(accelerator, model, eval_ds, step):
    loss = 0.0
    loss_dict = {}
    i = 0
    progress_bar = tqdm(range(len(eval_ds)), desc="eval")
    for batch in eval_ds:
        outputs = model(**batch, inference=True)
        if i == 0:
            # create a lineplot plot for each scalar in the first batch
            for measure in model.measures:
                fig, ax = plt.subplots()
                pred_vals = outputs["measures"][measure][0]
                true_vals = batch["measures"][measure][0]
                pred_vals, true_vals = accelerator.gather_for_metrics((pred_vals, true_vals))
                pred_vals = model.scalers[measure].transform(pred_vals)
                true_vals = model.scalers[measure].transform(true_vals)
                pred_vals = pred_vals.detach().cpu().numpy()
                true_vals = true_vals.detach().cpu().numpy()
                sns.lineplot(x=range(len(pred_vals)), y=pred_vals, ax=ax, label="pred")
                sns.lineplot(x=range(len(true_vals)), y=true_vals, ax=ax, label="true")
                ax.set_title(measure)
                # log the figure to wandb
                wandb.log({f"eval/{measure}": wandb.Image(fig)}, step=step)
                plt.close(fig)
        i += 1
        loss += outputs["loss"].item()
        for k, v in outputs["compound_losses"].items():
            loss_dict[k] = loss_dict.get(k, 0.0) + v.item()
        progress_bar.update(1)
    wandb.log({"eval/loss": loss / len(eval_ds)}, step=step)
    wandb.log({f"eval/{k}_loss": v / len(eval_ds) for k, v in loss_dict.items()}, step=step)

def main():
    parser = HfArgumentParser([Args])

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

    if not args.bf16:
        accelerator = Accelerator()
    else:
        accelerator = Accelerator(mixed_precision="bf16")

    with accelerator.main_process_first():
        libritts = load_dataset(args.dataset)

    train_ds = libritts[args.train_split]
    eval_ds = libritts[args.eval_split]

    speaker2idx = json.load(open(args.speaker2idx))
    phone2idx = json.load(open(args.phone2idx))

    collator = SpeechCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measures=[MEASURE_DICT[measure]() for measure in args.measures],
        return_keys=[
            "mel",
            "dvector",
            "measures",
        ],
        overwrite_max_length=True
    )

    model = Vocex(
        measure_nlayers=args.measure_nlayers,
        dvector_nlayers=args.dvector_nlayers,
        depthwise=args.depthwise,
        noise_factor=args.noise_factor,
        filter_size=args.filter_size,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )

    if args.resume_from_checkpoint:
        model.load_state_dict(torch.load(args.resume_from_checkpoint))

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator.collate_fn,
        prefetch_factor=args.prefetch_factor,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator.collate_fn,
        prefetch_factor=args.prefetch_factor,
    )

    if args.fit_scalers:
        model.fit_scalers(train_dataloader, args.fit_scalers_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=[0.9, 0.98],
        eps=1e-8,
    )

    lr_scheduler = NoamLR(
        optimizer,
        warmup_steps=args.warmup_steps,
    )

    num_epochs = args.max_epochs
    num_training_steps = num_epochs * len(train_dataloader)

    progress_bar = tqdm(range(num_training_steps), desc="training", disable=not accelerator.is_local_main_process)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    model.train()

    step = 0


    for epoch in range(num_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                step += 1
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), args.max_grad_norm)
                if step % args.gradient_sync_every == 0:
                    outputs = model(**batch)
                    loss = outputs["loss"] / args.gradient_accumulation_steps
                    accelerator.backward(loss)
                else:
                    with accelerator.no_sync(model):
                        outputs = model(**batch)
                        loss = outputs["loss"] / args.gradient_accumulation_steps
                        accelerator.backward(loss)
                lr_scheduler.step()
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                steps_until_logging = args.log_every - (step % args.log_every)
                ## accumulate losses for logging
                if steps_until_logging <= args.train_loss_logging_sum_steps:
                    if steps_until_logging == args.train_loss_logging_sum_steps:
                        loss_dict = {
                            k: v for k, v in outputs["compound_losses"].items()
                        }
                        loss_dict["loss"] = outputs["loss"]
                    else:
                        for k, v in outputs["compound_losses"].items():
                            loss_dict[k] += v
                        loss_dict["loss"] += outputs["loss"]
                ## log losses
                if step % args.log_every == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    wandb.log({"train/loss": loss_dict["loss"]/args.train_loss_logging_sum_steps, "lr": lr}, step=step)
                    wandb.log({f"train/{k}": loss_dict[k]/args.train_loss_logging_sum_steps for k, v in outputs["compound_losses"].items()}, step=step)
                    wandb.log({"train/global_step": step}, step=step)
                ## evaluate
                if step % args.eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        eval_loop(accelerator, model, eval_dataloader, step)
                    model.train()
                ## save checkpoint
                if step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), f"{args.checkpoint_dir}/model_{step}.pt")
                    accelerator.wait_for_everyone()
                progress_bar.update(1)

if __name__ == "__main__":
    main()

