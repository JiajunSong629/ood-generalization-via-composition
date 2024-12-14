import os
import json
import gc
import torch
import numpy as np
from typing import List
import logging
import sys
import datetime

from src.models.huggingface_models import HFModel
from src.models.shuffle_models import ShuffleModel
from src.config import TASK_CONFIGS, SHUFFLE_CONFIGS


def configure_logging():
    # Configure logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/run_{timestamp}.log"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file)

    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


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
    shuffle_seeds: List[int],
    logger: logging.Logger,
    **task_kwargs,
):
    logger.info("========= Evaluating Model with Shuffling ==========")
    logger.info(f"Model: {base_model.model_name}")
    logger.info(f"Task: {task.get_task_details()}")
    logger.info(f"Component: {component}")
    logger.info(f"Shuffle n heads: {shuffle_n_heads}")
    logger.info(f"Shuffle seeds: {shuffle_seeds}")

    """Common evaluation logic for all tasks"""
    # Select heads based on component
    induction_heads = base_model.diagonal_induction_heads
    previous_token_heads = base_model.diagonal_previous_token_heads
    layer_head_pairs = (induction_heads if component == "qk" else previous_token_heads)[
        :shuffle_n_heads
    ]

    aggregated_results = []

    # Evaluate original model
    result = task.evaluate_model(model=base_model, **task_kwargs)
    aggregated_results.append(result)
    logger.info("========= Original ==========")
    logger.info(result)

    # Evaluate shuffled models
    for shuffle_seed in shuffle_seeds:
        shuffle_model = ShuffleModel(
            base_model,
            layer_head_pairs=layer_head_pairs,
            seed=shuffle_seed,
        )

        # Inside shuffle
        shuffle_model.shuffle_inside(component=component)
        result = task.evaluate_model(model=shuffle_model, **task_kwargs)

        logger.info(f"========= Shuffled Inside Seed {shuffle_seed} ==========")
        logger.info(result)
        aggregated_results.append(result)

        shuffle_model.revert()

        # Outside shuffle
        shuffle_model.shuffle_outside(component=component)
        result = task.evaluate_model(model=shuffle_model, **task_kwargs)
        logger.info(f"========= Shuffled Outside Seed {shuffle_seed} ==========")
        logger.info(result)
        aggregated_results.append(result)

        shuffle_model.revert()

    return aggregated_results


def main(model_names: str, task_name: str):
    task_config = TASK_CONFIGS[task_name]

    logger = configure_logging()

    for model_name in model_names.split(","):
        base_model = HFModel(model_name, device="cuda", **task_config["model_kwargs"])
        task = task_config["task_class"](**task_config["task_kwargs"])
        for component in ["qk", "ov"]:
            results = evaluate_model_with_shuffling(
                base_model=base_model,
                task=task,
                component=component,
                shuffle_n_heads=SHUFFLE_CONFIGS["shuffle_n_heads"],
                shuffle_seeds=SHUFFLE_CONFIGS["shuffle_seeds"],
                logger=logger,
                **task_config["eval_kwargs"],
            )

            save_results(
                base_model,
                task_name,
                component,
                results,
                task_config.get("additional_params"),
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
