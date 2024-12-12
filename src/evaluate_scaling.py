import os
from typing import List
import json
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODEL_CLASSES, TASK_CONFIGS, SCALING_CONFIGS
from src.models.removal_models import HeadRemovalModel
from src.models.huggingface_models import HFModel

PYTHIA_MODEL_CLASSES = [m for m in MODEL_CLASSES if "pythia" in m]
SCALING_DIR = "scaling_results"


def evaluate_model_with_masking(
    base_model,
    task,
    num_samples: int = 100,
    task_seed: int = 42,
    **task_kwargs,
):
    """Common evaluation logic for masking experiments"""
    scaling_config = SCALING_CONFIGS[base_model.model_name]
    induction_heads = base_model.induction_heads

    acc_prob = []

    # Evaluate masked models
    removal_model = HeadRemovalModel(base_model=base_model)
    for n_head in scaling_config["n_heads"]:
        removal_model.mask(induction_heads[:n_head])
        result = task.evaluate_model(
            removal_model,
            num_samples=num_samples,
            task_random_seed=task_seed,
            **task_kwargs,
        )
        print("======== N_MASKED_HEADS", n_head)
        print(result)

        acc = 1 - result.err if hasattr(result, "err") else result.acc
        if hasattr(result, "prob"):
            acc_prob.append(
                {
                    "n_head": n_head,
                    "acc": acc,
                    "prob": result.prob,
                }
            )
        else:
            acc_prob.append(
                {
                    "n_head": n_head,
                    "acc": acc,
                }
            )
        removal_model.revert()

    return acc_prob


def plot(results, x_values, task_name: str):
    # Create two figures
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Color map for different model sizes
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # Plot for each model
    for (model_name, model_results), color in zip(results.items(), colors):
        # Extract model size from name for legend
        size = model_name.split("-")[1]

        # Get acc and prob values (first value is baseline, next 4 are masked results)
        acc_values = [r["acc"] for r in model_results][1:]
        prob_values = [r["prob"] for r in model_results][1:]

        # Plot accuracy
        ax1.plot(x_values, acc_values, "-o", label=size, color=color)

        # Plot probability
        ax2.plot(x_values, prob_values, "-o", label=size, color=color)

    # Configure accuracy plot
    ax1.set_xlabel("Number of Heads Masked")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Effect of Head Masking on Model Accuracy")
    ax1.legend(title="Model Size", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True)

    # Configure probability plot
    ax2.set_xlabel("Number of Heads Masked")
    ax2.set_ylabel("Probability")
    ax2.set_title("Effect of Head Masking on Model Probability")
    ax2.legend(title="Model Size", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True)

    # Adjust layout to prevent label cutoff
    fig1.tight_layout()
    fig2.tight_layout()

    os.makedirs(SCALING_DIR, exist_ok=True)
    # Save plots
    fig1.savefig(
        f"{SCALING_DIR}/scaling_removal_acc_{task_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    fig2.savefig(
        f"{SCALING_DIR}/scaling_removal_prob_{task_name}.png",
        bbox_inches="tight",
        dpi=300,
    )


def main(task_name: str):
    config = TASK_CONFIGS[task_name]

    results = {}
    for model_name in PYTHIA_MODEL_CLASSES:
        print(f"Evaluating {model_name}")
        base_model = HFModel(
            model_name=model_name, device="cuda", **config["model_kwargs"]
        )
        task = config["task_class"](**config["task_kwargs"])

        results[model_name] = evaluate_model_with_masking(
            base_model=base_model, task=task, **config["eval_kwargs"]
        )

    # Save results
    os.makedirs(SCALING_DIR, exist_ok=True)
    with open(f"{SCALING_DIR}/scaling_results_{task_name}.json", "w") as f:
        json.dump(results, f)

    # Plot results
    # plot(results, task_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        default="copying",
        help="Task name to evaluate (copying or icl)",
    )

    args = parser.parse_args()
    main(args.task_name)
