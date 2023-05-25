import json
import sys

from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from speech_collator import SpeechCollator
from speech_collator.measures import EnergyMeasure, PitchMeasure, SRMRMeasure, SNRMeasure, VoiceActivityMeasure
from transformers import HfArgumentParser
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from transformers import get_linear_schedule_with_warmup

from vocex import Vocex
from vocex.utils import NoamLR
from training.arguments import Args

MEASURE_DICT = {
    "energy": EnergyMeasure,
    "pitch": PitchMeasure,
    "srmr": SRMRMeasure,
    "snr": SNRMeasure,
    "voice_activity_binary": VoiceActivityMeasure,
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
                if not measure.endswith("_binary"):
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
            # create an image plot for the dvectors in the first batch
            fig, ax = plt.subplots()
            pred_vals = outputs["dvector"] # (batch_size, dvector_dim)
            true_vals = batch["dvector"]
            pred_vals, true_vals = accelerator.gather_for_metrics((pred_vals, true_vals))
            if not measure.endswith("_binary"):
                pred_vals = model.scalers["dvector"].transform(pred_vals)
                true_vals = model.scalers["dvector"].transform(true_vals)
            # for each dvector, draw as images next to each other
            pred_val = pred_vals.reshape(-1, 16, 16)
            true_val = true_vals.reshape(-1, 16, 16)
            pred_val = pred_val.detach().cpu().numpy()
            true_val = true_val.detach().cpu().numpy()
            # subplots
            fig, axs = plt.subplots(len(pred_val), 3, sharey=True, figsize=(4, 10))
            min_val = np.min([np.min(pred_val), np.min(true_val)])
            max_val = np.max([np.max(pred_val), np.max(true_val)])
            max_error = np.max(np.abs(pred_val-true_val))
            fig.suptitle(f"min={min_val:.2f}, max={max_val:.2f}\nmax_error={max_error:.2f}")
            for i in range(len(pred_val)):
                axs[i, 0].imshow(pred_val[i], interpolation='nearest', vmin=min_val, vmax=max_val)
                axs[i, 1].imshow(true_val[i], interpolation='nearest', vmin=min_val, vmax=max_val)
                axs[i, 2].imshow(
                    np.abs(pred_val[i]-true_val[i]),
                    cmap=sns.color_palette("light:r", as_cmap=True),
                    interpolation='nearest',
                    vmin=0,
                    vmax=max_error
                )
                if i == 0:
                    axs[i, 0].set_title("predicted")
                    axs[i, 1].set_title("ground truth")
                    axs[i, 2].set_title("abs. error")
                axs[i, 0].axis('off')
                axs[i, 1].axis('off')
                axs[i, 2].axis('off')
            plt.tight_layout()
            wandb.log({f"eval/dvector": wandb.Image(fig)}, step=step)
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

    train_ds = libritts[args.train_split].shuffle(seed=42)
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
        measures=args.measures,
        measure_nlayers=args.measure_nlayers,
        dvector_nlayers=args.dvector_nlayers,
        depthwise=args.depthwise,
        noise_factor=args.noise_factor,
        filter_size=args.filter_size,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_softdtw=args.use_softdtw,
        softdtw_gamma=args.softdtw_gamma,
    )

    if args.resume_from_checkpoint:
        try:
            model.load_state_dict(torch.load(args.resume_from_checkpoint), strict=True)
        except RuntimeError as e:
            if args.strict_load:
                raise e
            else:
                print("Could not load model from checkpoint. Trying without strict loading, and removing mismatched keys.")
                current_model_dict = model.state_dict()
                loaded_state_dict = torch.load(args.resume_from_checkpoint)
                new_state_dict={
                    k:v if v.size()==current_model_dict[k].size() 
                    else current_model_dict[k] 
                    for k,v 
                    in zip(current_model_dict.keys(), loaded_state_dict.values())
                }
                model.load_state_dict(new_state_dict, strict=False)

    model.scalers["dvector"].expected_max = torch.tensor(10.0)

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

    train_dataloader, eval_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer, lr_scheduler
    )

    model.train()

    step = 0

    losses = deque(maxlen=100)
    compound_losses = {k: deque(maxlen=100) for k in args.measures + ["dvector"]}

    print(f"number of parameters: {sum(p.numel() for p in model.parameters())}")

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
                ## add to queues
                losses.append(outputs["loss"])
                for k in args.measures + ["dvector"]:
                    compound_losses[k].append(outputs["compound_losses"][k])

                lr_scheduler.step()
                if step % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                ## log losses
                if step % args.log_every == 0:
                    last_lr = lr_scheduler.get_last_lr()[0]
                    log_loss_dict = {
                        f"train/{k}": sum([l.item() for l in compound_losses[k]])/len(compound_losses[k])
                        for k in args.measures + ["dvector"]
                    }
                    log_loss_dict["train/loss"] = sum([l.item() for l in losses])/len(losses)
                    wandb.log(log_loss_dict, step=step)
                    wandb.log({"train/global_step": step}, step=step)
                    print(f"step={step}, lr={last_lr:.8f}:")
                    print({k.split('/')[1]: np.round(v, 4) for k, v in log_loss_dict.items()})
                ## save checkpoint
                if step % args.save_every == 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save({
                        unwrapped_model.cpu().state_dict(),
                    }, f"{args.checkpoint_dir}/checkpoint_{step}.pt")
                ## evaluate
                if step % args.eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        eval_loop(accelerator, model, eval_dataloader, step)
                    #model.verbose = True
                    model.train()
                progress_bar.update(1)
                # set description
                progress_bar.set_description(f"epoch {epoch+1}/{num_epochs}")

if __name__ == "__main__":
    main()

