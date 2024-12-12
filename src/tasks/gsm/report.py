import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import json


from src.config import MODEL_CLASSES
import src.api.result as result_api


RESULTS_DIR = os.path.join("results")
FIGURES_DIR = os.path.join("figures")
IGNORE_MODELS = ["gpt2", "gpt2-xl"]


os.makedirs(FIGURES_DIR, exist_ok=True)


#######################################################
########### REMOVAL ###################################
#######################################################


def load_removal_results(model_name, type):
    fname = f"{model_name}_10_removal.json"
    result_path = os.path.join(RESULTS_DIR, fname)
    with open(result_path, "r") as f:
        results = json.load(f)

    results = [result_api.GSMResult(**r) for r in results]

    d = defaultdict(dict)
    for result in results:
        if "mask_meta" not in result.model_details:
            d["top_ih"][0] = getattr(result, type)
            continue

        mask_meta = result.model_details["mask_meta"]
        masked_method = mask_meta["masked_method"]
        masked_n_heads = int(mask_meta["masked_n_heads"])
        masked_seed = mask_meta["masked_seed"]

        if masked_method == "random":
            d[f"seed{masked_seed}"][masked_n_heads] = getattr(result, type)
        elif masked_method == "top_ih":
            d["top_ih"][masked_n_heads] = getattr(result, type)

    return d


def plot_removal(d_removal, type):
    sns.set_theme(style="darkgrid")
    num_models = len(d_removal)
    num_experiments = len(d_removal[list(d_removal.keys())[0]][type]) - 1
    fig, axes = plt.subplots(
        num_models,
        num_experiments,
        figsize=(6 * num_experiments, 6 * num_models),
    )

    # trunkate x to smaller than 100
    def trunk(a_list):
        return [x for x in a_list if x < 100]

    for i, (model_name, model_data) in enumerate(d_removal.items()):
        baseline = model_data[type]["top_ih"]
        baseline_x = trunk(list(baseline.keys()))
        baseline_y = [baseline[head] for head in baseline_x]

        ylim_min = min(baseline_y)
        ylim_max = max(baseline_y)

        j = 0
        for seed, seed_data in model_data[type].items():
            if seed == "top_ih":
                continue

            seed_x = trunk(list(seed_data.keys()))
            seed_y = [seed_data[head] for head in seed_x]

            ylim_min = min(ylim_min, min(seed_y))
            ylim_max = max(ylim_max, max(seed_y))

            sns.lineplot(
                x=seed_x,
                y=seed_y,
                ax=axes[i, j],
                color="red",
                label=f"{seed}",
            )
            sns.scatterplot(
                x=seed_x,
                y=seed_y,
                ax=axes[i, j],
                marker="o",
                color="red",
                s=50,
            )
            sns.lineplot(
                x=baseline_x,
                y=baseline_y,
                ax=axes[i, j],
                color="blue",
                label="Top IH",
            )
            sns.scatterplot(
                x=baseline_x,
                y=baseline_y,
                ax=axes[i, j],
                marker="o",
                color="blue",
                s=50,
            )

            axes[i, j].set_xlabel("Mask Head")
            axes[i, j].set_ylabel("Accuracy")
            axes[i, j].set_title(f"Accuracy vs. Mask heads - {model_name}")
            j += 1

        for axi in axes[i]:
            axi.set_ylim(ylim_min - 0.05, ylim_max + 0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "removal.png"))
    plt.close()


def main_removal():
    d_removal = {}
    not_found = []
    for model_name in MODEL_CLASSES:
        try:
            d_removal[model_name] = {}
            for type in ["acc"]:
                d_removal[model_name][type] = load_removal_results(model_name, type)
        except Exception as e:
            d_removal.pop(model_name)
            not_found.append({"model_name": model_name, "error": str(e)})

    with open("removal_results.json", "w") as f:
        json.dump(d_removal, f, indent=4)

    print(not_found)

    plot_removal(d_removal, "acc")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    if args.experiment == "removal":
        main_removal()
