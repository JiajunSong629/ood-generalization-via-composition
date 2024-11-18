import os
import json
import copy
from typing import List, Dict, Any, Tuple
import dataclasses

from src.models.huggingface_models import HFModel
from src.models.proj_models import ProjectModel
from src.api.result import ICLResult, CopyingResult, GSMResult
from src.tasks.icl.task import ICLTask
from src.tasks.copying.task import CopyingTask
from src.tasks.gsm.task import GSMTask


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


def save_results(base_model, task_name, component, results, additional_params=None):
    """Utility function to save results"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(cur_dir, "tasks", task_name, "results")
    os.makedirs(result_dir, exist_ok=True)

    params_str = (
        "_" + "_".join(str(v) for v in additional_params) if additional_params else ""
    )
    fname = f"{base_model.model_name}{params_str}_projection_{component}.json"
    ProjectionResult.save_multiple(results, os.path.join(result_dir, fname))


def print_result(result):
    """Print task-specific results"""
    if isinstance(result, CopyingResult):
        print(f"Acc: {1 - result.err:.2f}, Prob: {result.prob:.2f}")
    elif isinstance(result, (ICLResult, GSMResult)):
        print(f"Acc: {result.accuracy:.4f}")


def get_projection_heads(base_model, component: str, project_n_heads: int):
    """Get the appropriate heads for projection based on component"""
    diagonal_induction_heads = base_model.diagonal_induction_heads
    induction_heads = base_model.induction_heads
    previous_token_heads = base_model.previous_token_heads

    if component == "qk":
        layer_head_pairs = diagonal_induction_heads[:project_n_heads]
        projected_layer_head_pairs = induction_heads[: int(0.25 * len(induction_heads))]
    elif component == "ov":
        layer_head_pairs = diagonal_induction_heads[:project_n_heads]
        projected_layer_head_pairs = previous_token_heads[
            : int(0.25 * len(previous_token_heads))
        ]

    return layer_head_pairs, projected_layer_head_pairs


def evaluate_model_with_projection(
    base_model,
    task,
    component: str,
    project_n_heads: int,
    num_samples: int,
    task_seed: int = 42,
    **task_kwargs,
):
    print("========= MODEL ==========")
    print(base_model.model_meta)

    """Common evaluation logic for all tasks"""
    # Get initial result
    result = task.evaluate_model(
        model=base_model,
        num_samples=num_samples,
        task_random_seed=task_seed,
        **task_kwargs,
    )
    print("========= Original ==========")
    print_result(result)

    # Setup projection model
    layer_head_pairs, projected_layer_head_pairs = get_projection_heads(
        base_model, component, project_n_heads
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
        ranks = range(0, max(K, 300), 10)

        for rank in ranks:
            proj_model.project(component, rank, project_out=project_out)
            result = task.evaluate_model(
                model=proj_model,
                num_samples=num_samples,
                task_random_seed=task_seed,
                **task_kwargs,
            )

            # Store results
            if isinstance(result, CopyingResult):
                acc_prob["rank"] = acc_prob.get("rank", []) + [rank]
                acc_prob["acc"] = acc_prob.get("acc", []) + [1 - result.err]
                acc_prob["prob"] = acc_prob.get("prob", []) + [result.prob]
            elif isinstance(result, (ICLResult, GSMResult)):
                acc_prob["rank"] = acc_prob.get("rank", []) + [rank]
                acc_prob["acc"] = acc_prob.get("acc", []) + [result.accuracy]

            print(f"========= Rank {rank} ==========")
            print_result(result)
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
            num_samples=num_samples,
            result=acc_prob,
        )
        aggregated_results.append(result)

    return aggregated_results


def main(model_names: str, task_name: str):
    batch_size_dict = {
        "gemma2-9b": 1,
        "gpt2": 100,
        "gpt2-xl": 100,
    }

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
            "additional_params": ["symbol", "20"],
        },
        "gsm": {
            "task_class": GSMTask,
            "task_kwargs": {
                "num_shots": 10,
                "max_new_tokens": 128,
            },
            "model_kwargs": {"quantize": False},
        },
    }

    config = task_configs[task_name]
    for model_name in model_names.split(","):
        base_model = HFModel(model_name, device="cuda", **config["model_kwargs"])
        task = config["task_class"](**config["task_kwargs"])

        if task_name == "copying":
            more_kwargs = {"batch_size": batch_size_dict.get(model_name, 8)}
        else:
            more_kwargs = {}

        for component in ["qk", "ov"]:
            results = evaluate_model_with_projection(
                base_model=base_model,
                task=task,
                component=component,
                project_n_heads=10,
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
