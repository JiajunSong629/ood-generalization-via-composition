import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import fire

from utils import calc_QK_OV, create_folder


def jsonify(scores, IH, PTH, save_to):
    K0, K1 = scores.shape
    idx_sort = np.argsort(scores, axis=None)[::-1]

    arr_json = []
    for idx in idx_sort:
        i, j = idx // K1, idx % K1
        Li, Hi = IH[i]
        Lj, Hj = PTH[j]
        arr_json.append(
            {
                "IH": (int(Li), int(Hi)),
                "PTH": (int(Lj), int(Hj)),
                "score": scores[i, j],
            }
        )

    with open(save_to, "w") as f:
        json.dump(arr_json, f)

    print(f"Saved to {save_to}")


def select_IH_PTH(scores, IH, PTH, threshold=2):
    K0, K1 = scores.shape
    idx_sort = np.argsort(scores, axis=None)[::-1]

    IH_selected, PTH_selected = [], []
    for idx in idx_sort:
        i, j = idx // K1, idx % K1
        if scores[i, j] < threshold:
            break

        IH_selected.append(IH[i])
        PTH_selected.append(PTH[j])

    IH_selected = list(set(IH_selected))
    PTH_selected = list(set(PTH_selected))

    return IH_selected, PTH_selected


def qkov_matching_summary(W_all, IH, PTH):
    K0, K1 = len(IH), len(PTH)

    scores_nml = np.zeros((K0, K1))
    for i0, (Layer0, Head0) in enumerate(IH):
        for i1, (Layer1, Head1) in enumerate(PTH):
            W_qk = calc_QK_OV(W_all, Layer0, Head0, QK=True)
            W_ov = calc_QK_OV(W_all, Layer1, Head1, OV=True)
            W_qkov = (W_qk @ W_ov).numpy(force=True)
            scores_nml[i0, i1] = (np.mean(np.diag(W_qkov)) - np.mean(W_qkov)) / np.std(
                W_qkov
            )

    return scores_nml


def plot(scores_nml, IH, PTH, save_to):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    ax = axs[0]
    yticklabels = [f"L{layerhead[0]}H{layerhead[1]}" for layerhead in IH]
    xticklabels = [f"L{layerhead[0]}H{layerhead[1]}" for layerhead in PTH]
    sns.heatmap(scores_nml, ax=ax, xticklabels=xticklabels, yticklabels=yticklabels)
    ax.set_xlabel("Previous token head", weight="bold")
    ax.set_ylabel("IH head", weight="bold")
    ax.set_title(f"IH-Shifting matching", weight="bold")

    ax = axs[1]
    ax.hist(scores_nml.flatten(), bins=30, edgecolor="white")
    ax.set_title(f"Histogram of IH-PTH matching Z-score", weight="bold")

    plt.savefig(save_to, bbox_inches="tight")
    plt.close()


def main(model_name, K=20):
    # create_folder("Figs")
    # create_folder("Figs/diagonal")
    create_folder(f"out/{model_name}/Figs")

    W_all = torch.load(f"checkpoints/{model_name}/W_all.pt")
    IH = torch.load(f"checkpoints/{model_name}/IH.pt")[:K]
    PTH = torch.load(f"checkpoints/{model_name}/PTH.pt")[:K]

    scores = qkov_matching_summary(W_all, IH, PTH)

    jsonify(scores, IH, PTH, save_to=f"out/{model_name}/diagonal.json")
    plot(
        scores,
        IH,
        PTH,
        save_to=f"out/{model_name}/Figs/diagonal_matching_{K}.png",
    )

    # IH_selected, PTH_selected = select_IH_PTH(scores, IH, PTH, threshold=2)


if __name__ == "__main__":
    fire.Fire(main)
