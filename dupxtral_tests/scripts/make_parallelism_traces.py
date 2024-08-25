import argparse
import json
import uuid
from pathlib import Path
import random

import numpy as np
import torch
from accelerate import dispatch_model
from torch.profiler import profile, record_function, ProfilerActivity

from pairs_proba_random_generation import (
    random_pairs_distribution,
    random_pairs_temp_softmax_distribution,
)
from smart_device_map import (
    SmartAffectation,
    baseline_greedy_device_map,
    make_device_map_from_experts_to_gpu,
    even_experts_to_gpu_0_device_map,
)
from transformers.models.dupxtral import DupxtralConfig, DupxtralModel
from transformers.models.mixtral import MixtralConfig, MixtralModel


class ExperienceConfig:
    def __init__(self, args=None):
        self.from_args(args)

        if "seed" not in vars(args):
            self.seed = 42

    def __str__(self):
        return str(self.__dict__)

    def from_args(self, args):
        if (
            "json_configuration_path" not in vars(args)
            or args.json_configuration_path is None
        ):
            for k, v in vars(args).items():
                setattr(self, k, v)
        else:
            self.from_json(args.json_configuration_path)
            for k, v in vars(args).items():
                setattr(self, k, v)

    def from_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)
            for k, v in data.items():
                setattr(self, k, v)

        self.json_configuration_path = path

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(vars(self), f)

    def args_from_config(self):
        return argparse.Namespace(**vars(self))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_mode",
        type=str,
        default="base",
        choices=["base", "duplication"],
    )

    # w1 4096 x 14336
    # w2 4096 x 14336
    # w2 14336 x 4096

    # Toy model size
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--intermediate_size", type=int, default=14336)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--n_experts", type=int, default=8)

    # Load to process
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=1)

    # Expert probabilities
    parser.add_argument("--experts_pairs_distribution_path", type=str, default=None)
    parser.add_argument(
        "--random_expert_distribution_mode",
        type=str,
        default="uniform",
    )
    parser.add_argument(
        "--random_expert_distribution_temperature", type=float, default=1.0
    )

    # Device map method
    parser.add_argument(
        "--device_map",
        type=str,
        default="",
        choices=["optim_greedy", "random", "naive"],
    )
    parser.add_argument("--experts_per_gpu", type=int, default=8)
    parser.add_argument("--n_gpus", type=int, default=2)

    # Experiments metadata
    parser.add_argument("--expe_name", type=str, default="default")
    parser.add_argument("--json_configuration_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def initialize_model(args):

    # first make pairs distribution
    if args.experts_pairs_distribution_path is not None:
        pair_proba = np.load(args.experts_pairs_distribution_path)
    else:
        if args.random_expert_distribution_mode == "uniform":
            pair_proba = random_pairs_distribution(args.n_experts)
        elif args.random_expert_distribution_mode == "softmax":
            pair_proba = random_pairs_temp_softmax_distribution(
                args.n_experts, args.random_expert_distribution_temperature
            )
        else:
            raise ValueError(
                f"Unknown random expert distribution mode: {args.random_expert_distribution_mode}"
            )

    # pairs are non-zero values
    pairs = torch.nonzero(pair_proba).tolist()

    # Initialize model
    if args.base_mode != "duplication":
        mixtral_config = MixtralConfig(
            forward=args.forward,
            hidden_size=args.hidden_size,
            num_experts=args.n_experts,
            intermediate_size=args.intermediate_size,
            vocab_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=args.num_hidden_layers,
        )

        mixtral = MixtralModel(mixtral_config).to(torch.float16)

        if args.device_map == "auto":
            device_map = baseline_greedy_device_map(mixtral)
        elif args.device_map == "parallel":
            device_map = even_experts_to_gpu_0_device_map(mixtral)
        else:
            raise ValueError(f"Unknown device map: {args.device_map}")

    else:
        affectation_fn = SmartAffectation(
            pair_proba=pair_proba,
            pairs=pairs,
            n_gpus=args.n_gpus,
            max_experts_per_gpu=args.experts_per_gpu,
            n_experts=args.n_experts,
            n_experts_per_tokens=2,
            cost_offloading=1000,
        )

        pairs_mapping, experts_duplications, new_experts_to_gpu = (
            affectation_fn.make_expert_map(method=args.device_map)
        )

        pairs_mapping = [pairs_mapping for _ in range(args.num_hidden_layers)]
        experts_duplications = [
            experts_duplications for _ in range(args.num_hidden_layers)
        ]
        new_experts_to_gpu = [new_experts_to_gpu for _ in range(args.num_hidden_layers)]

        dupxtral_config = DupxtralConfig(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            vocab_size=256,
            num_attention_heads=4,
            num_local_experts=args.n_experts,
            num_key_value_heads=2,
            num_hidden_layers=args.num_hidden_layers,
            experts_duplicate=experts_duplications,
            experts_remapping=pairs_mapping,
            router_distribution=[pair_proba for _ in range(args.num_hidden_layers)],
            experts2gpu=new_experts_to_gpu,
        )

        mixtral = DupxtralModel(dupxtral_config).to(torch.float16)

        device_map = make_device_map_from_experts_to_gpu(
            model=mixtral, experts_to_gpu=new_experts_to_gpu
        )

    mixtral = dispatch_model(mixtral, device_map, main_device=torch.device("cuda:0"))

    return mixtral


def main():
    args = parse_args()
    unique_id = uuid.uuid4()

    config = ExperienceConfig(args)
    args = config.args_from_config()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    mixtral = initialize_model(args)

    # fake tokens

    inputs = [
        torch.randint(0, 256, (args.batch_size, args.sequence_length))
        for _ in range(200)
    ]

    schedule = torch.profiler.schedule(
        skip_first=2, wait=1, warmup=2, active=100, repeat=1
    )

    expertiment_path = Path(f"experiments/{args.expe_name}_{unique_id}")
    expertiment_path.mkdir(parents=True, exist_ok=True)

    # save configuration
    config.to_json(expertiment_path / "config.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(expertiment_path / "trace")
        ),
        schedule=schedule,
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        with record_function("forward"):
            with torch.no_grad():
                for i, input_ in enumerate(inputs):
                    with record_function(f"input_{i}"):
                        output = mixtral(input_)
                        prof.step()


if __name__ == "__main__":
    main()
