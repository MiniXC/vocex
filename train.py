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
from training.arguments import Args

MEASURE_DICT = {
    "energy": EnergyMeasure,
    "pitch": PitchMeasure,
    "srmr": SRMRMeasure,
    "snr": SNRMeasure,
}

def eval_loop(model, eval_ds, step):
    eval_ds = tqdm(eval_ds, desc="Evaluating", total=len(eval_ds))
    model.eval()
    loss = 0.0
    loss_dict = {}
    i = 0
    for batch in eval_ds:
        outputs = model(**batch)
        if i == 0:
            # create a lineplot plot for each scalar in the first batch
            logits = outputs["logits"]
            for j, measure in enumerate(model.measures):
                fig, ax = plt.subplots()
                pred_vals = logits[0, j]
                pred_vals = model.scalers[measure].inverse_transform(pred_vals).detach().cpu().numpy()
                true_vals = batch["measures"][measure][0].detach().cpu().numpy()
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
    wandb.log({"eval/loss": loss / len(eval_ds)}, step=step)
    wandb.log({f"eval/{k}_loss": v / len(eval_ds) for k, v in loss_dict.items()}, step=step)
    model.train()

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
    )
    wandb.config.update(args)

    libritts = load_dataset(args.dataset)
    train_ds = libritts[args.train_split]
    eval_ds = libritts[args.eval_split]

    if not args.bf16:
        accelerator = Accelerator()
    else:
        accelerator = Accelerator(mixed_precision="bf16")

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

    model.fit_scalers(train_dataloader, args.fit_scalers_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_epochs = args.max_epochs
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps, eta_min=args.learning_rate_min
    )

    progress_bar = tqdm(range(num_training_steps))

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    model.train()

    step = 0

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                step += 1
                outputs = model(**batch)
                loss = outputs["loss"]
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if step % args.log_every == 0:
                    wandb.log({"train/loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=step)
                    wandb.log({f"train/{k}": v.item() for k, v in outputs["compound_losses"].items()}, step=step)
                    wandb.log({"train/global_step": step}, step=step)
                if step % args.eval_every == 0:
                    eval_loop(model, eval_dataloader, step)
                if step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), f"{args.checkpoint_dir}/model_{step}.pt")
                    accelerator.wait_for_everyone()
                progress_bar.update(1)

if __name__ == "__main__":
    main()

