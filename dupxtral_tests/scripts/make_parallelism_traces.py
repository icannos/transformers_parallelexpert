import re

import torch
import torch.nn as nn

from accelerate import dispatch_model

from torch.profiler import profile, record_function, ProfilerActivity

from typing import Dict, Tuple, List
import argparse

from transformers.models.mixtral import MixtralConfig, MixtralModel
from transformers.models.dupxtral import DupxtralConfig, DupxtralModel


from accelerate import init_empty_weights

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
    parser.add_argument("--num_hidden_layers", type=int, default=2)

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


def make_dupxtral_device_map_and_remapping(
    model: nn.Module, duplicate: List[List[int]]
) -> Tuple[Dict[str, str], Dict[Tuple[int, int], Tuple[int, int]]]:
    mapping = {}
    remapping = {}

    expert_re = re.compile(
        r"layers\.(?P<layer_id>\d+)\.block_sparse_moe\.experts\.(?P<expert_id>\d+)"
    )
    n_experts = 0
    for name, m in model.named_modules():
        splitted = name.split(".")

        if name == "":
            continue

        # match the expert name
        match = re.match(expert_re, name)
        if match:
            layer_id = int(match.group("layer_id"))
            expert_id = int(match.group("expert_id"))

            n_experts = max(n_experts, expert_id + 1)

            if layer_id % 2 == 0:
                mapping[name] = f"cuda:0"
            else:
                mapping[name] = f"cuda:1"

        else:
            mapping[name] = "cuda:0"

    for k in range(len(duplicate)):
        remapping[k] = {}
        for i in range(n_experts):
            for j in range(n_experts):

                new_i = i * 2
                new_j = j * 2

                if i > j:
                    remapping[k][(i, j)] = (new_i, new_j + 1)
                else:
                    remapping[k][(i, j)] = (new_i + 1, new_j)

    return mapping, remapping


def initialize_model(args):

    if args.forward != "full_duplication":
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

    else:
        mixtral_base_config = MixtralConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            vocab_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=args.num_hidden_layers,
        )

        with init_empty_weights():
            mixtral = MixtralModel(mixtral_base_config)

            device_map, remapping = make_dupxtral_device_map_and_remapping(
                mixtral, [[2 for _ in range(8)] for _ in range(args.num_hidden_layers)]
            )

        dupxtral_config = DupxtralConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            vocab_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=args.num_hidden_layers,
            experts_duplicate=[
                [2 for _ in range(8)] for _ in range(args.num_hidden_layers)
            ],
            experts_remapping=remapping,
        )

        mixtral = DupxtralModel(dupxtral_config).to(torch.float16)

    mixtral = dispatch_model(mixtral, device_map, main_device=torch.device("cuda:0"))

    return mixtral


def main():
    args = parse_args()

    unique_id = uuid.uuid4()

    mixtral = initialize_model(args)

    # fake tokens

    inputs = [
        torch.randint(0, 256, (args.batch_size, args.sequence_length))
        for _ in range(100)
    ]

    schedule = torch.profiler.schedule(
        skip_first=2, wait=1, warmup=1, active=10, repeat=8
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
