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
    result: Dict[int, Dict[str, float]]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @staticmethod
    def save_multiple(results: List["ProjectionResult"], path: str):
        results_dict = [dataclasses.asdict(result) for result in results]
        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)


def eval_copying(
    model_name: str,
    project_n_heads: int = 10,
    batch_size: int = 32,
    seg_len: int = 25,
    rep: int = 3,
    ignore_segment: int = 2,
    ignore_burning: int = 4,
    component: str = "qk",
    num_samples: int = 100,
    device: str = "cuda",
):
    base_model = HFModel(model_name, device=device)
    diagonal_induction_heads = base_model.diagonal_induction_heads
    diagonal_previous_token_heads = base_model.diagonal_previous_token_heads
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

    task = CopyingTask(
        seg_len=seg_len,
        rep=rep,
        ignore_segment=ignore_segment,
        ignore_burning=ignore_burning,
    )

    task_seed = 42
    result: CopyingResult = task.evaluate_model(
        model=base_model,
        num_samples=num_samples,
        task_random_seed=task_seed,
        batch_size=batch_size,
    )

    print("========= Original ==========")
    print(f"Acc: {1 - result.err:.2f}, Prob: {result.prob:.2f}")

    proj_model = ProjectModel(
        base_model,
        layer_head_pairs=layer_head_pairs,
        projected_layer_head_pairs=projected_layer_head_pairs,
    )

    aggregated_results = []
    for project_out in [True, False]:
        acc_prob = {}
        for rank in range(0, 200, 5):
            proj_model.project(component, rank, project_out=project_out)
            result: CopyingResult = task.evaluate_model(
                model=proj_model,
                num_samples=num_samples,
                task_random_seed=task_seed,
                batch_size=batch_size,
            )
            acc_prob[rank] = {"acc": 1 - result.err, "prob": result.prob}

            print(f"========= Rank {rank} ==========")
            print(f"Acc: {1 - result.err:.2f}, Prob: {result.prob:.2f}")
            proj_model.revert()

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
            result=acc_prob,
        )
        aggregated_results.append(result)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(cur_dir, "tasks", "copying", "results")
    os.makedirs(result_dir, exist_ok=True)

    fname = f"{base_model.model_name}_projection_{component}.json"
    ProjectionResult.save_multiple(aggregated_results, os.path.join(result_dir, fname))


def eval_icl(
    model_name: str,
    project_n_heads: int = 10,
    component: str = "qk",
    setting: str = "symbol",
    num_shots: int = 20,
    balanced_sample: bool = True,
    num_samples: int = 100,
    device: str = "cuda",
):
    base_model = HFModel(model_name, device=device)
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

    task = ICLTask(
        setting=setting,
        num_shots=num_shots,
        balanced_sample=balanced_sample,
    )

    task_seed = 42
    result: ICLResult = task.evaluate_model(
        model=base_model,
        num_samples=num_samples,
        task_random_seed=task_seed,
    )

    print("========= MODEL ==========")
    print(base_model.model_name)
    print("========= Original ==========")
    print(f"Acc: {result.accuracy:.2f}")
    for example in result.examples[:3]:
        print(example.prompt[:30])
        print(example.model_solution)
        print(example.multiple_choice_logprob)

    proj_model = ProjectModel(
        base_model,
        layer_head_pairs=layer_head_pairs,
        projected_layer_head_pairs=projected_layer_head_pairs,
    )

    aggregated_results = []
    for project_out in [True, False]:
        acc_prob = {}
        K = max(50, int(0.05 * base_model.model_meta["hidden_size"] / 10) * 10)
        for rank in set([0, 50, K]):
            proj_model.project(component, rank, project_out=project_out)
            result: ICLResult = task.evaluate_model(
                model=proj_model,
                num_samples=num_samples,
                task_random_seed=task_seed,
            )
            acc_prob[rank] = {"acc": result.accuracy}

            print(f"========= Rank {rank} ==========")
            print(f"Acc: {result.accuracy:.4f}")
            for example in result.examples[:3]:
                print(example.prompt[:30])
                print(example.model_solution)
                print(example.multiple_choice_logprob)

            proj_model.revert()

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
            result=acc_prob,
        )
        aggregated_results.append(result)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(cur_dir, "tasks", "icl", "results")
    os.makedirs(result_dir, exist_ok=True)

    fname = f"{base_model.model_name}_symbol_20_projection_{component}.json"
    ProjectionResult.save_multiple(aggregated_results, os.path.join(result_dir, fname))


def eval_gsm(
    model_name: str,
    project_n_heads: int = 10,
    component: str = "qk",
    num_shots: int = 10,
    num_samples: int = 100,
    device: str = "cuda",
):
    base_model = HFModel(model_name, device=device, quantize=True)
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

    task = GSMTask(num_shots=num_shots, max_new_tokens=128)

    task_seed = 42
    result: GSMResult = task.evaluate_model(
        model=base_model,
        num_samples=num_samples,
        task_random_seed=task_seed,
        batch_size=1,
    )

    print("========= MODEL ==========")
    print(base_model.model_name)
    print("========= Original ==========")
    print(f"Acc: {result.accuracy:.2f}")

    proj_model = ProjectModel(
        base_model,
        layer_head_pairs=layer_head_pairs,
        projected_layer_head_pairs=projected_layer_head_pairs,
    )

    aggregated_results = []
    for project_out in [True, False]:
        acc_prob = {}
        K = max(50, int(0.05 * base_model.model_meta["hidden_size"] / 10) * 10)
        for rank in set([0, 50, K]):
            proj_model.project(component, rank, project_out=project_out)
            result: GSMResult = task.evaluate_model(
                model=proj_model,
                num_samples=num_samples,
                task_random_seed=task_seed,
            )
            acc_prob[rank] = {"acc": result.accuracy}

            print(f"========= Rank {rank} ==========")
            print(f"Acc: {result.accuracy:.4f}")
            proj_model.revert()

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
            result=acc_prob,
        )
        aggregated_results.append(result)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(cur_dir, "tasks", "gsm", "results")
    os.makedirs(result_dir, exist_ok=True)

    fname = f"{base_model.model_name}_projection_{component}.json"
    ProjectionResult.save_multiple(aggregated_results, os.path.join(result_dir, fname))


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
                project_n_heads=10,
                component=component,
                num_samples=100,
                batch_size=batch_size_dict.get(model_name, 8),
            )
        elif task_name == "icl":
            eval_icl(
                model_name=model_name,
                project_n_heads=10,
                component=component,
                num_samples=100,
            )
        elif task_name == "gsm":
            eval_gsm(
                model_name=model_name,
                project_n_heads=10,
                component=component,
                num_samples=3,
            )

    for model_name in model_names.split(","):
        run(model_name=model_name, component="qk")


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
