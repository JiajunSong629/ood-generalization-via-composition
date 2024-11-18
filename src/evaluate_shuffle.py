import os
import json
import gc
import torch
import numpy as np
from typing import List

from src.models.huggingface_models import HFModel
from src.models.shuffle_models import ShuffleModel
from src.api.result import ICLResult, CopyingResult, GSMResult

from src.tasks.icl.task import ICLTask
from src.tasks.copying.task import CopyingTask
from src.tasks.gsm.task import GSMTask


def save_results(base_model, task_name, component, results, additional_params=None):
    """Utility function to save results"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(cur_dir, "tasks", task_name, "results")
    os.makedirs(result_dir, exist_ok=True)

    params_str = (
        "_" + "_".join(str(v) for v in additional_params) if additional_params else ""
    )
    fname = f"{base_model.model_name}{params_str}_shuffle_{component}.json"
    result_class = type(results[0])
    result_class.save_multiple(results, os.path.join(result_dir, fname))


def evaluate_model_with_shuffling(
    base_model,
    task,
    component: str,
    shuffle_n_heads: int,
    num_samples: int,
    task_seed: int = 42,
    shuffle_seeds=None,
    **task_kwargs,
):
    """Common evaluation logic for all tasks"""
    shuffle_seeds = shuffle_seeds or range(0, 100, 20)

    # Select heads based on component
    induction_heads = base_model.diagonal_induction_heads
    previous_token_heads = base_model.diagonal_previous_token_heads
    layer_head_pairs = (induction_heads if component == "qk" else previous_token_heads)[
        :shuffle_n_heads
    ]

    aggregated_results = []

    # Evaluate original model
    result = task.evaluate_model(
        model=base_model,
        num_samples=num_samples,
        task_random_seed=task_seed,
        **task_kwargs,
    )
    aggregated_results.append(result)
    print("========= Original ==========")
    print_result(result)

    # Evaluate shuffled models
    for shuffle_seed in shuffle_seeds:
        shuffle_model = ShuffleModel(
            base_model,
            layer_head_pairs=layer_head_pairs,
            seed=shuffle_seed,
        )

        # Inside shuffle
        shuffle_model.shuffle_inside(component=component)
        result = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
            **task_kwargs,
        )

        print(f"========= Shuffled Inside Seed {shuffle_seed} ==========")
        print_result(result)
        aggregated_results.append(result)

        shuffle_model.revert()

        # Outside shuffle
        shuffle_model.shuffle_outside(component=component)
        result = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
            **task_kwargs,
        )
        print(f"========= Shuffled Outside Seed {shuffle_seed} ==========")
        print_result(result)
        aggregated_results.append(result)

        shuffle_model.revert()

    return aggregated_results


def print_result(result):
    """Print task-specific results"""
    if isinstance(result, CopyingResult):
        print(f"{1 - result.err:.2f}, {result.prob:.2f}")
    elif isinstance(result, (ICLResult, GSMResult)):
        print(f"{result.accuracy:.2f}")


def main(model_names: str, task_name: str):
    task_configs = {
        "copying": {
            "task_class": CopyingTask,
            "task_kwargs": {
                "seg_len": 25,
                "rep": 3,
                "ignore_segment": 2,
                "ignore_burning": 4,
            },
            "model_kwargs": {"quantize": False},
        },
        "icl": {
            "task_class": ICLTask,
            "task_kwargs": {
                "setting": "symbol",
                "num_shots": 20,
                "balanced_sample": True,
            },
            "model_kwargs": {"quantize": False},
            "additional_params": ["symbol", 20],
        },
        "gsm": {
            "task_class": GSMTask,
            "task_kwargs": {
                "num_shots": 10,
                "max_new_tokens": 128,
            },
            "model_kwargs": {"quantize": False},
            "additional_params": [10],
        },
    }

    config = task_configs[task_name]

    for model_name in model_names.split(","):
        base_model = HFModel(model_name, device="cuda", **config["model_kwargs"])

        task = config["task_class"](**config["task_kwargs"])

        batch_size = {"gemma2-9b": 1, "gpt2": 100, "gpt2-xl": 100}.get(model_name, 8)
        if task_name == "copying":
            more_kwargs = {"batch_size": batch_size}
        else:
            more_kwargs = {}

        for component in ["qk", "ov"]:
            results = evaluate_model_with_shuffling(
                base_model=base_model,
                task=task,
                component=component,
                shuffle_n_heads=10,
                num_samples=100,
                **more_kwargs,
            )

            save_results(
                base_model,
                task_name,
                component,
                results,
                config.get("additional_params"),
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_names",
        type=str,
        default="llama2-7b",
        help="Model names to evaluate, separated by commas",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="copying",
        help="Task name to evaluate",
    )

    args = parser.parse_args()
    main(args.model_names, args.task_name)
