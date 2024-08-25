import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import json
import fire
from utils import calc_QK_OV, create_folder, custom_svd, svdAB


def subspace_matching_IH(
    W_all,
    IH=None,
    rank=10,
    num_samp=50,
    method="largest",
):
    num_layer, num_heads, _, d_model, d_head = W_all.shape
    match_baseline = np.zeros(num_samp)
    for i in range(num_samp):
        U1, s1, Vt1 = svdAB(
            np.random.randn(d_model, d_head), np.random.randn(d_model, d_head)
        )
        U2, s2, Vt2 = svdAB(
            np.random.randn(d_model, d_head), np.random.randn(d_model, d_head)
        )
        _, s_match_u, _ = custom_svd(Vt1[:rank, :] @ Vt2[:rank, :].T)
        match_baseline[i] = (
            s_match_u[0] if method == "largest" else np.sqrt(np.mean(s_match_u**2))
        )

    K0, K1 = len(IH), len(IH)
    s_match = np.zeros((K0, K1))
    for i0 in range(K0):
        for i1 in range(K1):
            L0, H0 = IH[i0]
            L1, H1 = IH[i1]
            U_0, s_0, Vt_0 = svdAB(
                W_all[L0, H0, 0].numpy(force=True), W_all[L0, H0, 1].numpy(force=True)
            )
            U_1, s_1, Vt_1 = svdAB(
                W_all[L1, H1, 0].numpy(force=True), W_all[L1, H1, 1].numpy(force=True)
            )

            A0 = Vt_0.T
            A1 = Vt_1.T
            _, s, _ = custom_svd(A0[:, :rank].T @ A1[:, :rank])
            s_match[i0, i1] = s[0] if method == "largest" else np.sqrt(np.mean(s**2))

    return s_match, match_baseline


def subspace_matching_PTH(
    W_all,
    PTH=None,
    rank=10,
    num_samp=50,
    method="largest",
):
    num_layer, num_heads, _, d_model, d_head = W_all.shape
    match_baseline = np.zeros(num_samp)
    for i in range(num_samp):
        U1, s1, Vt1 = svdAB(
            np.random.randn(d_model, d_head), np.random.randn(d_model, d_head)
        )
        U2, s2, Vt2 = svdAB(
            np.random.randn(d_model, d_head), np.random.randn(d_model, d_head)
        )
        _, s_match_u, _ = custom_svd(Vt1[:rank, :] @ Vt2[:rank, :].T)
        match_baseline[i] = (
            s_match_u[0] if method == "largest" else np.sqrt(np.mean(s_match_u**2))
        )

    K0, K1 = len(PTH), len(PTH)
    s_match = np.zeros((K0, K1))
    for i0 in range(K0):
        for i1 in range(K1):
            L0, H0 = PTH[i0]
            L1, H1 = PTH[i1]
            U_0, s_0, Vt_0 = svdAB(
                W_all[L0, H0, 3].numpy(force=True), W_all[L0, H0, 2].numpy(force=True)
            )
            U_1, s_1, Vt_1 = svdAB(
                W_all[L1, H1, 3].numpy(force=True), W_all[L1, H1, 2].numpy(force=True)
            )
            A0 = U_0
            A1 = U_1
            _, s, _ = custom_svd(A0[:, :rank].T @ A1[:, :rank])
            s_match[i0, i1] = s[0] if method == "largest" else np.sqrt(np.mean(s**2))

    return s_match, match_baseline


def subspace_matching_IH_PTH(
    W_all,
    IH=None,
    PTH=None,
    rank=10,
    num_samp=50,
    method="largest",
):
    num_layer, num_heads, _, d_model, d_head = W_all.shape
    match_baseline = np.zeros(num_samp)
    for i in range(num_samp):
        U1, s1, Vt1 = svdAB(
            np.random.randn(d_model, d_head), np.random.randn(d_model, d_head)
        )
        U2, s2, Vt2 = svdAB(
            np.random.randn(d_model, d_head), np.random.randn(d_model, d_head)
        )
        _, s_match_u, _ = custom_svd(Vt1[:rank, :] @ Vt2[:rank, :].T)
        match_baseline[i] = (
            s_match_u[0] if method == "largest" else np.sqrt(np.mean(s_match_u**2))
        )

    K0, K1 = len(IH), len(PTH)
    s_match = np.zeros((K0, K1))
    for i0 in range(K0):
        for i1 in range(K1):
            L0, H0 = IH[i0]
            L1, H1 = PTH[i1]
            U_0, s_0, Vt_0 = svdAB(
                W_all[L0, H0, 0].numpy(force=True), W_all[L0, H0, 1].numpy(force=True)
            )
            U_1, s_1, Vt_1 = svdAB(
                W_all[L1, H1, 3].numpy(force=True), W_all[L1, H1, 2].numpy(force=True)
            )
            A0 = Vt_0.T
            A1 = U_1
            _, s, _ = custom_svd(A0[:, :rank].T @ A1[:, :rank])
            s_match[i0, i1] = s[0] if method == "largest" else np.sqrt(np.mean(s**2))

    return s_match, match_baseline


def jsonify(
    s_match,
    match_baseline,
    LayerHeadPair0,
    LayerHeadPair1,
    save_to,
):
    arr = [{"baseline": np.mean(match_baseline)}]

    K0, K1 = s_match.shape
    idx_sort = np.argsort(s_match, axis=None)[::-1]
    for idx in idx_sort:
        i, j = idx // K1, idx % K1
        L0, H0 = LayerHeadPair0[i]
        L1, H1 = LayerHeadPair1[j]
        arr.append(
            {
                "LH0": (int(L0), int(H0)),
                "LH1": (int(L1), int(H1)),
                "score": s_match[i, j],
            }
        )

    with open(save_to, "w") as f:
        json.dump(arr, f)

    print(f"Saved to {save_to}")


def plot(
    s_match,
    match_baseline,
    LayerHeadPair0,
    LayerHeadPair1,
    save_to,
):
    yticklabels = [f"L{layerhead[0]}H{layerhead[1]}" for layerhead in LayerHeadPair0]
    xticklabels = [f"L{layerhead[0]}H{layerhead[1]}" for layerhead in LayerHeadPair1]
    baseline = np.mean(match_baseline)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    sns.heatmap(
        s_match,
        ax=axs[0],
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )
    axs[0].set_title(f"Baseline: {baseline:.3f}")

    axs[1].hist(s_match.flatten(), bins=30, edgecolor="white")
    axs[1].axvline(x=baseline, color="red", linestyle="dashed")
    axs[1].set_title(f"Baseline: {baseline:.3f}")

    plt.savefig(save_to, bbox_inches="tight")
    plt.close()


def IH_IH(
    model_name,
    W_all,
    K=10,
    rank=10,
    IH=None,
    method="largest",
):
    s_match, match_baseline = subspace_matching_IH(W_all, IH=IH, rank=rank)
    jsonify(
        s_match,
        match_baseline,
        IH,
        IH,
        save_to=f"out/{model_name}/subspace_IH_K{K}_{method}.json",
    )
    plot(
        s_match,
        match_baseline,
        IH,
        IH,
        save_to=f"out/{model_name}/Figs/subspace_IH_K{K}_{method}.png",
    )


def PTH_PTH(
    model_name,
    W_all,
    K=10,
    rank=10,
    PTH=None,
    method="largest",
):
    s_match, match_baseline = subspace_matching_PTH(W_all, PTH=PTH, rank=rank)
    jsonify(
        s_match,
        match_baseline,
        PTH,
        PTH,
        save_to=f"out/{model_name}/subspace_PTH_K{K}_{method}.json",
    )
    plot(
        s_match,
        match_baseline,
        PTH,
        PTH,
        save_to=f"out/{model_name}/Figs/subspace_PTH_K{K}_{method}.png",
    )


def IH_PTH(
    model_name,
    W_all,
    K=10,
    rank=10,
    IH=None,
    PTH=None,
    method="largest",
):
    s_match, match_baseline = subspace_matching_IH_PTH(W_all, IH=IH, PTH=PTH, rank=rank)
    jsonify(
        s_match,
        match_baseline,
        IH,
        PTH,
        save_to=f"out/{model_name}/subspace_IH_PTH_K{K}_{method}.json",
    )
    plot(
        s_match,
        match_baseline,
        IH,
        PTH,
        save_to=f"out/{model_name}/Figs/subspace_IH_PTH_K{K}_{method}.png",
    )


def main(
    model_name,
    K=10,
    rank=10,
    method="largest",
):
    W_all = torch.load(f"checkpoints/{model_name}/W_all.pt")
    IH = torch.load(f"checkpoints/{model_name}/IH.pt")[:K]
    PTH = torch.load(f"checkpoints/{model_name}/PTH.pt")[:K]

    IH_IH(model_name, W_all, K, rank, IH, method)
    PTH_PTH(model_name, W_all, K, rank, PTH, method)
    IH_PTH(model_name, W_all, K, rank, IH, PTH, method)


if __name__ == "__main__":
    fire.Fire(main)
