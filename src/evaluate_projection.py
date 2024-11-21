import os
import json
import copy
from typing import List, Dict, Any, Tuple
import dataclasses

from src.models.huggingface_models import HFModel
from src.models.proj_models import ProjectModel
from src.api.result import (
    ICLResult,
    CopyingResult,
    GSMResult,
    FuzzyCopyResult,
    IOIResult,
)

from src.config import PROJECT_CONFIGS, TASK_CONFIGS, MODEL_META


@dataclasses.dataclass
class ProjectionResult:
    model_details: Dict[str, Any]
    task_details: Dict[str, Any]
    num_samples: int
    result: Dict[int, Dict[str, float]]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @staticmethod
    def save_multiple(results: List["ProjectionResult"], path: str):
        results_dict = [dataclasses.asdict(result) for result in results]
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)


def save_results(
    base_model,
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

    fname = f"{base_model.model_name}{proj_params_str}{task_params_str}_projection_{component}.json"
    ProjectionResult.save_multiple(results, os.path.join(result_dir, fname))


def get_projection_heads(base_model, component: str):
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
        num_all_heads = 8 * MODEL_META[model_name]["num_layers"]
    else:
        num_all_heads = len(induction_heads)

    project_n_heads = PROJECT_CONFIGS["project_n_heads"]
    projected_n_heads = PROJECT_CONFIGS["projected_n_heads"]
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
    **task_kwargs,
):
    print("========= MODEL ==========")
    print(base_model.model_meta)

    """Common evaluation logic for all tasks"""
    # Get initial result
    result = task.evaluate_model(model=base_model, **task_kwargs)
    print("========= Original ==========")
    print(result)

    # Setup projection model
    layer_head_pairs, projected_layer_head_pairs = get_projection_heads(
        base_model, component
    )

    proj_model = ProjectModel(
        base_model,
        layer_head_pairs=layer_head_pairs,
        projected_layer_head_pairs=projected_layer_head_pairs,
    )

    # Evaluate with different ranks and projection settings
    aggregated_results = []
    for project_out in [True, False]:
        acc_prob = {}
        K = max(50, int(0.05 * base_model.model_meta["hidden_size"] / 10) * 10)
        ranks = [0, 50, 100, 150, 200, K]

        for rank in ranks:
            proj_model.project(component, rank, project_out=project_out)
            result = task.evaluate_model(model=proj_model, **task_kwargs)

            acc_prob["rank"] = acc_prob.get("rank", []) + [rank]
            # Store results
            if isinstance(result, (CopyingResult)):
                acc_prob["acc"] = acc_prob.get("acc", []) + [1 - result.err]
                acc_prob["prob"] = acc_prob.get("prob", []) + [result.prob]
            elif isinstance(result, (FuzzyCopyResult, ICLResult, IOIResult)):
                acc_prob["acc"] = acc_prob.get("acc", []) + [result.acc]
                acc_prob["prob"] = acc_prob.get("prob", []) + [result.prob]
            elif isinstance(result, (GSMResult)):
                acc_prob["acc"] = acc_prob.get("acc", []) + [result.acc]

            print(f"========= Rank {rank} ==========")
            print(result)
            proj_model.revert()

        # Create projection result
        model_details = copy.deepcopy(base_model.model_meta)
        model_details["projection"] = {
            "component": component,
            "project_out": project_out,
            "layer_head_pairs": layer_head_pairs,
            "projected_layer_head_pairs": projected_layer_head_pairs,
        }

        result = ProjectionResult(
            model_details=model_details,
            task_details=task.get_task_details(),
            num_samples=task_kwargs["num_samples"],
            result=acc_prob,
        )
        aggregated_results.append(result)

    return aggregated_results


def main(model_names: str, task_name: str):
    task_config = TASK_CONFIGS[task_name]
    for model_name in model_names.split(","):
        base_model = HFModel(model_name, device="cuda", **task_config["model_kwargs"])
        task = task_config["task_class"](**task_config["task_kwargs"])

        for component in ["qk", "ov"]:
            results = evaluate_model_with_projection(
                base_model=base_model,
                task=task,
                component=component,
                **task_config["eval_kwargs"],
            )

            save_results(
                base_model,
                task_name,
                component,
                results,
                proj_additional_params=PROJECT_CONFIGS.get("additional_params"),
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
