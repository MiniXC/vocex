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
import numpy as np
from collections import deque
from transformers import get_linear_schedule_with_warmup

from vocex import Vocex
from vocex.utils import NoamLR
from training.arguments import Args
def main():
    parser = HfArgumentParser([Args])

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]


    if not args.bf16:
        accelerator = Accelerator()
    else:
        accelerator = Accelerator(mixed_precision="bf16")

    model = Vocex(
        measure_nlayers=args.measure_nlayers,
        dvector_nlayers=args.dvector_nlayers,
        depthwise=args.depthwise,
        noise_factor=args.noise_factor,
        filter_size=args.filter_size,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )

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

    # save to .pt file on cpu
    model.cpu()
    torch.save(model.state_dict(), args.resume_from_checkpoint.replace(".pt", "_cpu.pt"))

if __name__ == "__main__":
    main()

