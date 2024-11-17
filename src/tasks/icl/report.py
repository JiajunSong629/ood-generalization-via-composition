import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from src.config import MODEL_CLASSES
import src.api.result as result_api


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CUR_DIR, "results")


def load_results_shuffle(model_name, component):
    fname = f"{model_name}_symbol_20_shuffle_{component}.json"
    result_path = os.path.join(RESULTS_DIR, fname)
    with open(result_path, "r") as f:
        results = json.load(f)

    results = [result_api.ICLResult(**r) for r in results]

    return results


def load_results_projection(model_name, component):
    fname = f"{model_name}_symbol_20_projection_{component}.json"
    result_path = os.path.join(RESULTS_DIR, fname)
    with open(result_path, "r") as f:
        results = json.load(f)

    return results


def pretty_print_result_shuffle(result: result_api.CopyingResult):
    if "shuffle_meta" in result.model_details:
        shuffle_meta = result.model_details["shuffle_meta"]
        print(
            "Shuffle",
            shuffle_meta["shuffled_type"],
            shuffle_meta["shuffled_component"],
            shuffle_meta["seed"],
        )
    else:
        print("Original")

    print(f"{result.prob:.2f}, {1 - result.err:.2f}")


def summarise_results_shuffle(results):
    d = {}

    for r in results:
        if "shuffle_meta" in r.model_details:
            shuffle_meta = r.model_details["shuffle_meta"]
            shuffle_type = shuffle_meta["shuffled_type"]
            if shuffle_type not in d:
                d[shuffle_type] = {"accuracy": []}
            d[shuffle_type]["accuracy"].append(r.accuracy)

        else:
            d["original"] = {"accuracy": r.accuracy}

    for shuffle_type in d:
        d[shuffle_type] = {
            "mean": np.mean(d[shuffle_type]["accuracy"]),
            "std": np.std(d[shuffle_type]["accuracy"]),
        }

    return d


def collect_all_models():
    d_all_qk = {}
    d_all_ov = {}
    for model_name in MODEL_CLASSES:
        results = load_results_shuffle(model_name, "qk")
        d = summarise_results_shuffle(results)
        d_all_qk[model_name] = d

    for model_name in MODEL_CLASSES:
        results = load_results_shuffle(model_name, "ov")
        d = summarise_results_shuffle(results)
        d_all_ov[model_name] = d

    return d_all_qk, d_all_ov


def make_table(d_qk, d_ov):
    print("\\toprule")
    print("Model & Original & QK Inside & QK Outside & OV Inside & OV Outside \\\\")
    print("\\midrule")

    for model_name in MODEL_CLASSES:
        qk_model = d_qk[model_name]
        ov_model = d_ov[model_name]

        original = qk_model["original"]["mean"]  # Original is same in both QK and OV
        qk_inside = qk_model["inside"]["mean"]
        qk_outside = qk_model["outside"]["mean"]
        ov_inside = ov_model["inside"]["mean"]
        ov_outside = ov_model["outside"]["mean"]

        print(
            f"{model_name} & {original:.3f} & {qk_inside:.3f} & {qk_outside:.3f} & {ov_inside:.3f} & {ov_outside:.3f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")


def aggregate_shuffle_results(d_qk, d_ov):

    for model_name in MODEL_CLASSES:
        qk_model = d_qk[model_name]
        ov_model = d_ov[model_name]

        qk_inside = qk_model["inside"]["mean"]
        qk_outside = qk_model["outside"]["mean"]
        ov_inside = ov_model["inside"]["mean"]
        ov_outside = ov_model["outside"]["mean"]

        print(
            f"{model_name} & {qk_inside:.3f} & {qk_outside:.3f} & {ov_inside:.3f} & {ov_outside:.3f} \\\\"
        )


def aggregate_shuffle_results(d_qk, d_ov):
    cmap = matplotlib.cm.get_cmap("brg")
    colors = [cmap(t) for t in np.linspace(0, 0.9, len(MODEL_CLASSES))]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    label_handle = []
    label_names = []
    for i, model_name in enumerate(MODEL_CLASSES):
        d_model = d_qk[model_name]
        mu_original = d_model["original"]["mean"]
        mu_inside = d_model["inside"]["mean"]
        mu_outside = d_model["outside"]["mean"]
        p1 = ax.scatter(
            mu_original - mu_inside,
            mu_original - mu_outside,
            marker="+",
            color=colors[i],
            s=100,
            label=model_name + " qk",
        )

        d_model = d_ov[model_name]
        mu_original = d_model["original"]["mean"]
        mu_inside = d_model["inside"]["mean"]
        mu_outside = d_model["outside"]["mean"]
        p2 = ax.scatter(
            mu_original - mu_inside,
            mu_original - mu_outside,
            marker="x",
            color=colors[i],
            s=100,
            label=model_name + " ov",
        )
        label_handle.append((p1, p2))
        label_names.append(model_name + " QK/OV")

    ax.set_xlim((0 - 0.1, 1))
    ax.set_ylim((0 - 0.1, 1))
    l = ax.legend(
        label_handle, label_names, handler_map={tuple: HandlerTuple(ndivide=None)}
    )
    title = (
        "Reduced prob of predicting correct token"
        if type == "prob"
        else "Reduced acc of predicting correct token"
    )
    ax.set_title(title, weight="bold", fontsize=12)
    ax.set_xlabel("Shuffle heads within", weight="bold", fontsize=14)
    ax.set_ylabel("Shuffle heads outside", weight="bold", fontsize=14)
    plt.savefig(os.path.join(RESULTS_DIR, "shuffle_summary.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    d_qk, d_ov = collect_all_models()
    make_table(d_qk, d_ov)
    aggregate_shuffle_results(d_qk, d_ov)
