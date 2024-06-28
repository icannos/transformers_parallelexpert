import re

import torch
import torch.nn as nn

from accelerate import dispatch_model

from torch.profiler import profile, record_function, ProfilerActivity

from typing import Dict
import argparse

from transformers.models.mixtral import MixtralConfig, MixtralModel

import uuid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forward",
        type=str,
        default="base",
        choices=["base", "parallel", "full_duplication"],
    )
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--intermediate_size", type=int, default=14336)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=4)

    parser.add_argument("--device_map", type=str, default="auto")

    return parser.parse_args()


def make_path_from_args(args):
    value_list = [v for k, v in vars(args).items()]
    return "_".join(map(str, value_list))


def make_device_map(model: nn.Module) -> Dict[str, str]:
    mapping = {}

    expert_re = re.compile(
        r"layers\.(?P<layer_id>\d+)\.block_sparse_moe\.experts\.(?P<expert_id>\d+)"
    )
    for name, m in model.named_modules():
        splitted = name.split(".")

        if name == "":
            continue

        # match the expert name
        match = re.match(expert_re, name)
        if match:
            expert_id = int(match.group("expert_id"))
            mapping[name] = f"cuda:{expert_id % 2}"
        else:
            mapping[name] = "cuda:0"

    return mapping


def auto_device_map(model: nn.Module) -> Dict[str, str]:
    expert_re = re.compile(r"layers\.(?P<layer_id>\d+)\.")
    mapping = {}
    for name, m in model.named_modules():
        if name == "":
            continue

        # match the expert name
        match = re.match(expert_re, name)
        if match:
            layer_id = int(match.group("layer_id"))
            if layer_id >= 4:
                mapping[name] = f"cuda:0"
            else:
                mapping[name] = f"cuda:1"
        else:
            mapping[name] = "cuda:0"

    return mapping


def main():
    args = parse_args()

    unique_id = uuid.uuid4()

    mixtral_config = MixtralConfig(
        forward=args.forward,
        hidden_size=args.hidden_size,
        num_experts=8,
        intermediate_size=args.intermediate_size,
        vocab_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=args.num_hidden_layers,
    )

    mixtral = MixtralModel(mixtral_config).to(torch.float16)

    if args.device_map == "auto":
        device_map = auto_device_map(mixtral)
    elif args.device_map == "parallel":
        device_map = make_device_map(mixtral)
    else:
        raise ValueError(f"Unknown device map: {args.device_map}")

    mixtral = dispatch_model(mixtral, device_map, main_device="cuda:0")

    # fake tokens

    inputs = [
        torch.randint(0, 256, (args.batch_size, args.sequence_length))
        for _ in range(100)
    ]

    schedule = torch.profiler.schedule(
        skip_first=2, wait=1, warmup=1, active=3, repeat=2
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"trace_dir2/{make_path_from_args(args)}_{unique_id}"
        ),
        schedule=schedule,
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        with record_function("forward"):
            with torch.no_grad():
                for i, input_ in enumerate(inputs):
                    with record_function(f"input_{i}"):
                        output = mixtral(input_)
                        prof.step()


if __name__ == "__main__":
    main()
