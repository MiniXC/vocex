import sys

from transformers import HfArgumentParser
import torch
import torch.nn.utils.prune as prune

from vocex import VocexModel, Vocex
from training.arguments import Args

def main():
    parser = HfArgumentParser([Args])

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
        
    args.measures = args.measures.split(",")
    
    model_args = {
        "measures": args.measures,
        "measure_nlayers": args.measure_nlayers,
        "dvector_nlayers": args.dvector_nlayers,
        "depthwise": args.depthwise,
        "noise_factor": args.noise_factor,
        "filter_size": args.filter_size,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "use_softdtw": args.use_softdtw,
        "softdtw_gamma": args.softdtw_gamma,
    }

    state_dict = "models/vocex_600k.pt"

    model = VocexModel(**model_args)
    model.load_state_dict(torch.load(state_dict))
    
    # change to fp16
    model = model.half()

    vocex = Vocex(model)

    new_checkpoint = "models/checkpoint_half_prune.ckpt"

    vocex.save_checkpoint(new_checkpoint)

    
if __name__ == "__main__":
    main()