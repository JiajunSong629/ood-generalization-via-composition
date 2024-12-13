import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from src.config import MODEL_CLASSES
import src.api.result as result_api


RESULTS_DIR = os.path.join("results")
FIGURES_DIR = os.path.join("figures")
IGNORE_MODELS = ["gpt2", "gpt2-xl", "gemma-7b"]


os.makedirs(FIGURES_DIR, exist_ok=True)

#####################################################
########### SHUFFLE #################################
#####################################################


def load_shuffle_results(model_name, component, type):
    fname = f"{model_name}_upper_8_shuffle_{component}.json"
    result_path = os.path.join(RESULTS_DIR, fname)
    with open(result_path, "r") as f:
        results = json.load(f)

    results = [result_api.FuzzyCopyResult(**r) for r in results]

    d = {
        "original": [],
        "inside": {},
        "outside": {},
    }

    for r in results:
        if "shuffle_meta" in r.model_details:
            shuffle_meta = r.model_details["shuffle_meta"]
            shuffle_type = shuffle_meta["shuffled_type"]
            shuffle_seed = str(shuffle_meta["seed"])
            d[shuffle_type][shuffle_seed] = getattr(r, type)
        else:
            d["original"] = getattr(r, type)

    return d


def plot_shuffle_results(d_all, type, seed=None):
    d_qk = d_all[type]["qk"]
    d_ov = d_all[type]["ov"]
    cmap = matplotlib.colormaps.get_cmap("brg")
    colors = [cmap(t) for t in np.linspace(0, 0.9, len(d_qk))]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    label_handle = []
    label_names = []

    for i, model_name in enumerate(sorted(d_qk.keys())):
        d_model_qk = d_qk[model_name]
        d_model_ov = d_ov[model_name]

        mu_original = d_model_qk["original"]
        if seed is None:
            mu_inside = np.mean(list(d_model_qk["inside"].values()))
            mu_outside = np.mean(list(d_model_qk["outside"].values()))
        else:
            mu_inside = d_model_qk["inside"][str(seed)]
            mu_outside = d_model_qk["outside"][str(seed)]
        p1 = ax.scatter(
            (mu_original - mu_inside) / mu_original,
            (mu_original - mu_outside) / mu_original,
            marker="+",
            color=colors[i],
            s=100,
            label=model_name + " qk",
        )

        mu_original = d_model_ov["original"]
        if seed is None:
            mu_inside = np.mean(list(d_model_ov["inside"].values()))
            mu_outside = np.mean(list(d_model_ov["outside"].values()))
        else:
            mu_inside = d_model_ov["inside"][str(seed)]
            mu_outside = d_model_ov["outside"][str(seed)]
        p2 = ax.scatter(
            (mu_original - mu_inside) / mu_original,
            (mu_original - mu_outside) / mu_original,
            marker="x",
            color=colors[i],
            s=100,
            label=model_name + " ov",
        )
        label_handle.append((p1, p2))
        label_names.append(model_name + " QK/OV")

    ax.add_line(plt.Line2D([0, 1], [0, 1], color="black", linestyle="--"))
    ax.set_xlim((0 - 0.08, 1))
    ax.set_ylim((0 - 0.08, 1))
    ax.set_aspect("equal")
    l = ax.legend(
        label_handle,
        label_names,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    fig.set_size_inches(6, 4)
    title = f"Reduced {type.upper()} of predicting correct token"
    ax.set_title(title, weight="bold", fontsize=12)
    ax.set_xlabel("Shuffle heads within", weight="bold", fontsize=14)
    ax.set_ylabel("Shuffle heads outside", weight="bold", fontsize=14)
    if seed is None:
        figpath = os.path.join(FIGURES_DIR, f"shuffle_summary_{type}.png")
    else:
        figpath = os.path.join(FIGURES_DIR, f"shuffle_summary_{type}_seed{seed}.png")
    plt.savefig(figpath, bbox_inches="tight")
    plt.close()


def main_shuffle():
    d_all = {}
    for type in ["acc", "prob"]:
        d_qk, d_ov = {}, {}
        for model_name in MODEL_CLASSES:
            if model_name in IGNORE_MODELS:
                continue
            try:
                d_qk[model_name] = load_shuffle_results(model_name, "qk", type)
                d_ov[model_name] = load_shuffle_results(model_name, "ov", type)
            except FileNotFoundError:
                print(f"File not found for {model_name}")
            except Exception as e:
                print(f"Error for {model_name}: {e}")

        d_all[type] = {"qk": d_qk, "ov": d_ov}

    with open("shuffle_results.json", "w") as f:
        json.dump(d_all, f, indent=4)

    for type in ["acc", "prob"]:
        for seed in [0, 20, 40, 60, 80, None]:
            plot_shuffle_results(d_all, type=type, seed=seed)


#####################################################
########### PROJECTION ##############################
#####################################################


def load_projection_results(model_name, component, type):
    fname = f"{model_name}_50_upper_8_projection_{component}.json"
    result_path = os.path.join(RESULTS_DIR, fname)
    with open(result_path, "r") as f:
        results = json.load(f)

    results = [result_api.FuzzyCopyResult(**r) for r in results]
    d_model = results[0].model_details["hidden_size"]
    K = max(50, int(0.05 * d_model / 10) * 10)

    d = {"K": K}
    for result in results:
        proj_meta = result.model_details["project_meta"]
        proj_out = proj_meta["proj_out"]
        rank = proj_meta["rank"]

        if rank == 0:
            d["original"] = getattr(result, type)
        elif proj_out and rank == K:
            d["proj_true_K"] = getattr(result, type)
        elif not proj_out and rank == K:
            d["proj_false_K"] = getattr(result, type)

    return d


def plot_projection_results(d, component, type):
    import matplotlib
    from matplotlib.legend_handler import HandlerTuple

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    label_handle = []
    label_names = []

    dd = d[type][component]
    cmap = matplotlib.colormaps.get_cmap("brg")
    colors = [cmap(t) for t in np.linspace(0, 0.9, len(dd))]

    for i, model_name in enumerate(sorted(dd.keys())):
        baseline = dd[model_name]["original"]
        proj_true = dd[model_name]["proj_true_K"]
        proj_false = dd[model_name]["proj_false_K"]

        p1 = ax.scatter(
            (baseline - proj_true) / baseline,
            (baseline - proj_false) / baseline,
            marker="+",
            s=100,
            color=colors[i],
        )
        label_handle.append((p1))
        label_names.append(model_name)

    ax.add_line(plt.Line2D([0, 1], [0, 1], color="black", linestyle="--"))
    ax.set_xlim((0 - 0.05, 1 + 0.05))
    ax.set_ylim((0 - 0.05, 1 + 0.05))
    ax.set_aspect("equal")
    l = ax.legend(
        label_handle,
        label_names,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    fig.set_size_inches(6, 4)
    title = f"Reduced {type.upper()} of predicting correct token"
    ax.set_title(title, weight="bold", fontsize=13)
    ax.set_xlabel("Remove subspace", weight="bold", fontsize=14)
    ax.set_ylabel("Keep subspace", weight="bold", fontsize=14)
    plt.savefig(
        os.path.join(FIGURES_DIR, f"projection_summary_{component}_{type}.png"),
        bbox_inches="tight",
    )
    plt.close()


def main_projection():
    d_all = {}
    for type in ["acc", "prob"]:
        d_qk, d_ov = {}, {}
        for model_name in MODEL_CLASSES:
            if model_name in IGNORE_MODELS:
                continue
            try:
                d_qk[model_name] = load_projection_results(model_name, "qk", type)
                d_ov[model_name] = load_projection_results(model_name, "ov", type)
            except FileNotFoundError:
                print(f"File not found for {model_name}")
            except Exception as e:
                print(f"Error for {model_name}: {e}")

        d_all[type] = {"qk": d_qk, "ov": d_ov}

    with open("projection_results.json", "w") as f:
        json.dump(d_all, f, indent=4)

    for type in ["prob", "acc"]:
        for component in ["qk"]:
            plot_projection_results(d_all, component=component, type=type)


#######################################################
########### SCALING ###################################
#######################################################


def load_scaling_results(model_name, type):
    fname = f"{model_name}_upper_8_removal.json"
    result_path = os.path.join(RESULTS_DIR, fname)
    with open(result_path, "r") as f:
        results = json.load(f)

    results = [result_api.FuzzyCopyResult(**r) for r in results]

    d = {"top_ih": {}, "random": {}}
    for result in results:
        if "mask_meta" not in result.model_details:
            d["top_ih"][0] = getattr(result, type)
            continue

        mask_meta = result.model_details["mask_meta"]
        masked_method = mask_meta["masked_method"]
        masked_n_heads = mask_meta["masked_n_heads"]
        if masked_method == "random":
            d["random"][masked_n_heads] = d["random"].get(masked_n_heads, []) + [
                getattr(result, type)
            ]
        elif masked_method == "top_ih":
            d["top_ih"][masked_n_heads] = getattr(result, type)

    return d


def plot_scaling_results(results, type):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for (model_name, model_result), color in zip(results.items(), colors):
        # Extract model size from name for legend
        size = (
            model_name.split("-")[1]
            .replace("b", "B")
            .replace("m", "M")
            .replace("_", ".")
        )
        x_values = list(model_result[type]["top_ih"].keys())
        y_values = [model_result[type]["top_ih"][x] for x in x_values]
        ax.plot(x_values, y_values, "-o", label=size, color=color)

    ax.set_xlabel("Number of Heads Masked")
    ax.set_ylabel(type.upper())
    ax.set_title(f"Effect of Head Masking on Model {type.upper()}")
    ax.set_xlim(-5, 60)
    ax.legend(title="Model Size", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)

    # Adjust layout to prevent label cutoff
    fig.tight_layout()
    fig.savefig(
        f"{FIGURES_DIR}/scaling_removal_symbol_{type}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def main_scaling():
    d_scaling_symbol = {}
    for model_name in MODEL_CLASSES:
        if not model_name.startswith("pythia") or model_name == "pythia-14m":
            continue
        d_scaling_symbol[model_name] = {}
        for type in ["acc", "prob"]:
            d_scaling_symbol[model_name][type] = load_scaling_results(model_name, type)

    with open("scaling_results.json", "w") as f:
        json.dump(d_scaling_symbol, f, indent=4)

    for type in ["acc", "prob"]:
        plot_scaling_results(d_scaling_symbol, type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    if args.experiment == "shuffle":
        main_shuffle()
    if args.experiment == "projection":
        main_projection()
    if args.experiment == "scaling":
        main_scaling()

    if args.experiment == "all":
        main_shuffle()
        main_projection()
        main_scaling()
