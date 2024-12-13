import os
from typing import List
import json
import matplotlib.pyplot as plt
import numpy as np
import dataclasses

from src.config import TASK_CONFIGS, REMOVAL_CONFIGS
from src.models.removal_models import HeadRemovalModel
from src.models.huggingface_models import HFModel


def save_results(
    model_name,
    task_name,
    results,
    removal_additional_params=None,
    task_additional_params=None,
):
    """Utility function to save results"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    task_type = task_name.split("-")[
        0
    ]  # for ioi-original like task name, we save it under ioi
    result_dir = os.path.join(cur_dir, "tasks", task_type, "results")
    os.makedirs(result_dir, exist_ok=True)

    task_params_str = (
        "_" + "_".join(str(v) for v in task_additional_params)
        if task_additional_params
        else ""
    )
    removal_params_str = (
        "_" + "_".join(str(v) for v in removal_additional_params)
        if removal_additional_params
        else ""
    )

    fname = f"{model_name}{removal_params_str}{task_params_str}_removal.json"
    result_class = type(results[0])
    result_class.save_multiple(results, os.path.join(result_dir, fname))


def evaluate_model_with_masking(
    base_model,
    task,
    **task_kwargs,
):
    """Common evaluation logic for masking experiments"""
    removal_config = REMOVAL_CONFIGS[base_model.model_name]

    aggregated_results = []

    # Evaluate masked models
    removal_model = HeadRemovalModel(base_model=base_model)
    for n_head in removal_config["n_heads"]:
        # remove top induction heads
        removal_model.mask_top_ih(n_head)
        result = task.evaluate_model(removal_model, **task_kwargs)
        print("======== MASK", n_head, "TOP IH HEADS")
        print(result)
        aggregated_results.append(result)
        removal_model.revert()

        if n_head == 0:
            continue

        # remove random heads
        for seed in REMOVAL_CONFIGS["random_seeds"]:
            removal_model.mask_random(n_head, seed)
            result = task.evaluate_model(removal_model, **task_kwargs)
            print("======== MASK", n_head, "RANDOM HEADS")
            print(result)

            aggregated_results.append(result)
            removal_model.revert()

    return aggregated_results


def main(model_names: str, task_name: str):
    task_config = TASK_CONFIGS[task_name]

    for model_name in model_names.split(","):
        removal_config = REMOVAL_CONFIGS[model_name]
        base_model = HFModel(model_name=model_name, **task_config["model_kwargs"])
        task = task_config["task_class"](**task_config["task_kwargs"])
        results = evaluate_model_with_masking(
            base_model=base_model,
            task=task,
            **task_config["eval_kwargs"],
        )

        save_results(
            model_name=model_name,
            task_name=task_name,
            results=results,
            task_additional_params=task_config.get("additional_params"),
            removal_additional_params=removal_config.get("additional_params"),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_names",
        type=str,
        default="pythia-70m",
        help="Model names to evaluate, separated by commas",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="copying",
        help="Task name to evaluate (copying or icl)",
    )

    args = parser.parse_args()
    main(args.model_names, args.task_name)
