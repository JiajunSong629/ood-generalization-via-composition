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


def eval_copying(
    model_name: str,
    shuffle_n_heads: int = 10,
    batch_size: int = 32,
    seg_len: int = 25,
    rep: int = 3,
    ignore_segment: int = 1,
    ignore_burning: int = 4,
    component: str = "qk",
    num_samples: int = 100,
    device: str = "cuda",
):
    aggregated_results = []
    base_model = HFModel(model_name, device=device)
    induction_heads = base_model.induction_heads
    previous_token_heads = base_model.previous_token_heads

    if component == "qk":
        layer_head_pairs = induction_heads[:shuffle_n_heads]
    elif component == "ov":
        layer_head_pairs = previous_token_heads[:shuffle_n_heads]

    task = CopyingTask(
        seg_len=seg_len,
        rep=rep,
        ignore_segment=ignore_segment,
        ignore_burning=ignore_burning,
    )

    task_seed = 42
    shuffle_seeds = range(0, 100, 20)

    result: CopyingResult = task.evaluate_model(
        model=base_model,
        num_samples=num_samples,
        task_random_seed=task_seed,
        batch_size=batch_size,
    )
    aggregated_results.append(result)

    print("========= Original ==========")
    print(f"{1 - result.err:.2f}, {result.prob:.2f}")

    for shuffle_seed in shuffle_seeds:
        shuffle_model = ShuffleModel(
            base_model,
            layer_head_pairs=layer_head_pairs,
            seed=shuffle_seed,
        )

        shuffle_model.shuffle_inside(component=component)
        result: CopyingResult = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
            batch_size=batch_size,
        )

        print(f"========= Shuffled Inside Seed {shuffle_seed} ==========")
        print(f"{1 - result.err:.2f}, {result.prob:.2f}")
        aggregated_results.append(result)

        shuffle_model.revert()

        result: CopyingResult = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            batch_size=batch_size,
            task_random_seed=task_seed,
        )

        shuffle_model.shuffle_outside(component=component)
        result: CopyingResult = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
            batch_size=batch_size,
        )
        aggregated_results.append(result)

        print(f"========= Shuffled Outside Seed {shuffle_seed} ==========")
        print(f"{1 - result.err:.2f}, {result.prob:.2f}")

        shuffle_model.revert()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(cur_dir, "tasks", "copying", "results")
    os.makedirs(result_dir, exist_ok=True)

    fname = f"{base_model.model_name}_shuffle_{component}.json"
    CopyingResult.save_multiple(aggregated_results, os.path.join(result_dir, fname))


def eval_icl(
    model_name: str,
    shuffle_n_heads: int = 10,
    component: str = "qk",
    setting: str = "symbol",
    num_shots: int = 20,
    balanced_sample: bool = True,
    num_samples: int = 100,
    device: str = "cuda",
):
    aggregated_results = []
    base_model = HFModel(model_name, device=device)
    induction_heads = base_model.induction_heads
    previous_token_heads = base_model.previous_token_heads
    if component == "qk":
        layer_head_pairs = induction_heads[:shuffle_n_heads]
    elif component == "ov":
        layer_head_pairs = previous_token_heads[:shuffle_n_heads]

    task = ICLTask(
        setting=setting,
        num_shots=num_shots,
        balanced_sample=balanced_sample,
    )

    task_seed = 42
    shuffle_seeds = range(0, 100, 20)

    result: ICLResult = task.evaluate_model(
        model=base_model,
        num_samples=num_samples,
        task_random_seed=task_seed,
    )
    aggregated_results.append(result)

    print("========= Original ==========")
    print(f"{result.accuracy:.2f}")

    for shuffle_seed in shuffle_seeds:
        shuffle_model = ShuffleModel(
            base_model,
            layer_head_pairs=layer_head_pairs,
            seed=shuffle_seed,
        )

        shuffle_model.shuffle_inside(component=component)
        result: ICLResult = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
        )

        print(f"========= Shuffled Inside Seed {shuffle_seed} ==========")
        print(f"{result.accuracy:.2f}")
        aggregated_results.append(result)

        shuffle_model.revert()
        result: ICLResult = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
        )

        print("=========== Reverted ==========")
        print(f"{result.accuracy:.2f}")

        shuffle_model.shuffle_outside(component=component)
        result: ICLResult = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
        )

        aggregated_results.append(result)
        print(f"========= Shuffled Outside Seed {shuffle_seed} ==========")
        print(f"{result.accuracy:.2f}")

        shuffle_model.revert()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(cur_dir, "tasks", "icl", "results")
    os.makedirs(result_dir, exist_ok=True)

    fname = f"{model_name}_{setting}_{num_shots}_shuffle_{component}.json"
    ICLResult.save_multiple(aggregated_results, os.path.join(result_dir, fname))


def eval_gsm(
    model_name: str,
    shuffle_n_heads: int = 10,
    component: str = "qk",
    num_shots: int = 10,
    num_samples: int = 100,
    device: str = "cuda",
):
    aggregated_results = []
    base_model = HFModel(model_name, device=device, quantize=True)
    induction_heads = base_model.induction_heads
    previous_token_heads = base_model.previous_token_heads
    if component == "qk":
        layer_head_pairs = induction_heads[:shuffle_n_heads]
    elif component == "ov":
        layer_head_pairs = previous_token_heads[:shuffle_n_heads]

    task = GSMTask(num_shots=num_shots, max_new_tokens=128)

    task_seed = 42
    shuffle_seeds = range(0, 100, 20)

    result: GSMResult = task.evaluate_model(
        model=base_model,
        num_samples=num_samples,
        task_random_seed=task_seed,
        batch_size=1,
    )
    aggregated_results.append(result)

    print("========= Original ==========")
    print(f"{result.accuracy:.2f}")

    for shuffle_seed in shuffle_seeds:
        shuffle_model = ShuffleModel(
            base_model,
            layer_head_pairs=layer_head_pairs,
            seed=shuffle_seed,
        )

        shuffle_model.shuffle_inside(component=component)
        result: GSMResult = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
        )

        print(f"========= Shuffled Inside Seed {shuffle_seed} ==========")
        print(f"{result.accuracy:.2f}")
        aggregated_results.append(result)

        shuffle_model.revert()
        result: GSMResult = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
        )

        shuffle_model.shuffle_outside(component=component)
        result: GSMResult = task.evaluate_model(
            model=shuffle_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
        )

        aggregated_results.append(result)
        print(f"========= Shuffled Outside Seed {shuffle_seed} ==========")
        print(f"{result.accuracy:.2f}")

        shuffle_model.revert()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(cur_dir, "tasks", "gsm", "results")
    os.makedirs(result_dir, exist_ok=True)

    fname = f"{model_name}_{num_shots}_shuffle_{component}.json"
    GSMResult.save_multiple(aggregated_results, os.path.join(result_dir, fname))


def main(model_names: str, task_name: str):
    batch_size_dict = {
        "gemma2-9b": 1,
        "gpt2": 100,
        "gpt2-xl": 100,
    }

    def run(model_name, component):
        if task_name == "copying":
            eval_copying(
                model_name=model_name,
                shuffle_n_heads=10,
                component=component,
                num_samples=100,
                batch_size=batch_size_dict.get(model_name, 8),
            )
        elif task_name == "icl":
            eval_icl(
                model_name=model_name,
                shuffle_n_heads=10,
                component=component,
                setting="symbol",
                num_shots=20,
                num_samples=100,
            )
        elif task_name == "gsm":
            eval_gsm(
                model_name=model_name,
                shuffle_n_heads=10,
                component=component,
                num_shots=10,
                num_samples=100,
            )

    for model_name in model_names.split(","):
        for component in ["qk", "ov"]:
            run(model_name=model_name, component=component)


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
