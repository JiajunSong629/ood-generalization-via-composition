import os
import json
import copy
from typing import List, Dict, Any, Tuple
import dataclasses

from src.models.huggingface_models import HFModel
from src.models.proj_models import ProjectModel
from src.config import PROJECT_CONFIGS, TASK_CONFIGS, MODEL_META


def save_results(
    model_name,
    task_name,
    component,
    results,
    proj_additional_params=None,
    task_additional_params=None,
):
    """Utility function to save results"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(cur_dir, "tasks", task_name, "results")
    os.makedirs(result_dir, exist_ok=True)

    task_params_str = (
        "_" + "_".join(str(v) for v in task_additional_params)
        if task_additional_params
        else ""
    )
    proj_params_str = (
        "_" + "_".join(str(v) for v in proj_additional_params)
        if proj_additional_params
        else ""
    )

    fname = (
        f"{model_name}{proj_params_str}{task_params_str}_projection_{component}.json"
    )
    result_class = type(results[0])
    result_class.save_multiple(results, os.path.join(result_dir, fname))


def get_projection_heads(base_model, proj_config, component: str):
    """Get the appropriate heads for projection based on component"""
    diagonal_induction_heads = base_model.diagonal_induction_heads
    induction_heads = base_model.induction_heads
    previous_token_heads = base_model.previous_token_heads

    model_name = base_model.model_name
    if base_model.model_name in ["mistral-7b", "llama3-8b"]:
        num_key_value_heads = MODEL_META[model_name]["num_key_value_heads"]
        num_hidden_layers = MODEL_META[model_name]["num_layers"]
        num_all_heads = num_key_value_heads * num_hidden_layers
    elif model_name in ["falcon-7b"]:
        num_all_heads = 1 * MODEL_META[model_name]["num_layers"]
    else:
        num_all_heads = len(induction_heads)

    project_n_heads = proj_config["project_n_heads"]
    projected_n_heads = proj_config["projected_n_heads"]
    if projected_n_heads < 1:
        projected_n_heads = int(projected_n_heads * num_all_heads)

    if component == "qk":
        layer_head_pairs = diagonal_induction_heads[:project_n_heads]
        projected_layer_head_pairs = induction_heads[:projected_n_heads]
    elif component == "ov":
        layer_head_pairs = diagonal_induction_heads[:project_n_heads]
        projected_layer_head_pairs = previous_token_heads[:projected_n_heads]

    return layer_head_pairs, projected_layer_head_pairs


def evaluate_model_with_projection(
    base_model,
    task,
    component: str,
    ranks: List[int],
    proj_config,
    **task_kwargs,
):
    print("========= MODEL ==========")
    print(base_model.model_meta)

    """Common evaluation logic for all tasks"""
    layer_head_pairs, projected_layer_head_pairs = get_projection_heads(
        base_model, proj_config, component
    )

    proj_model = ProjectModel(
        base_model,
        layer_head_pairs=layer_head_pairs,
        projected_layer_head_pairs=projected_layer_head_pairs,
    )

    # Evaluate with different ranks and projection settings
    aggregated_results = []
    for project_out in [True, False]:
        for rank in ranks:
            proj_model.project(component, rank, project_out=project_out)
            result = task.evaluate_model(model=proj_model, **task_kwargs)
            aggregated_results.append(result)

            print(f"========= Rank {rank} {project_out} ==========")
            print(result)
            proj_model.revert()

    return aggregated_results


def main(model_names: str, task_name: str):
    task_config = TASK_CONFIGS[task_name]
    proj_config = PROJECT_CONFIGS[task_name]
    for model_name in model_names.split(","):
        base_model = HFModel(model_name, **task_config["model_kwargs"])
        task = task_config["task_class"](**task_config["task_kwargs"])
        proj_ranks = PROJECT_CONFIGS[model_name]["ranks"]

        for component in ["qk", "ov"]:
            results = evaluate_model_with_projection(
                base_model=base_model,
                task=task,
                component=component,
                ranks=proj_ranks,
                proj_config=proj_config,
                **task_config["eval_kwargs"],
            )

            save_results(
                model_name,
                task_name,
                component,
                results,
                proj_additional_params=proj_config.get("additional_params"),
                task_additional_params=task_config.get("additional_params"),
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
