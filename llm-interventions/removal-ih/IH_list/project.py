import copy
import torch
import gc
import json
import fire
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    get_qkov_weight,
    get_config,
    load_model,
    inference_probs_and_errs,
    make_input_ids,
    custom_svd,
    calc_rotary_R_mat,
    set_seed,
)


def projection_edit(
    model,
    model_name,
    layer_head_pairs,
    P,
    component,
):
    if model_name.startswith("gpt"):
        P_tensor = torch.tensor(P, device="cuda").float()
    else:
        P_tensor = torch.tensor(P, device="cuda", dtype=torch.bfloat16)

    config = get_config(model_name)
    for layer, head in layer_head_pairs:
        if component == "QK":
            W = get_qkov_weight(model, model_name, config, layer, head, "k")
            W.copy_(P_tensor @ copy.deepcopy(W))

        elif component == "OV":
            W = get_qkov_weight(model, model_name, config, layer, head, "o")
            W.copy_(P_tensor @ copy.deepcopy(W))

    return model


def plot(
    result,
    save_to,
):
    # make plots
    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))

    ranks, probs, errs = [], [], []
    for rank in result:
        prob, err = result[rank]["prob"], result[rank]["err"]
        avg_prob, avg_err = np.mean(prob), np.mean(err)
        if rank == 0:
            baseline_prob = avg_prob
            baseline_err = avg_err
        else:
            ranks.append(rank)
            probs.append(avg_prob)
            errs.append(avg_err)

    axs[0].plot(ranks, probs, "-o", label="projected")
    axs[1].plot(ranks, errs, "-o", label="projected")
    axs[0].axhline(y=baseline_prob, linestyle="dashed", label="original")
    axs[1].axhline(y=baseline_err, linestyle="dashed", label="original")

    for j in range(2):
        titles = [f"Pred {a} under projection" for a in ["probs", "errs"]]
        axs[j].set_xlabel("Subspace rank", weight="bold")
        axs[j].set_ylabel("Target token pred probs/errs", weight="bold")
        axs[j].set_ylim(0, 1)
        axs[j].set_title(titles[j], weight="bold")

    axs[0].legend()
    axs[1].legend()
    plt.savefig(save_to, bbox_inches="tight")
    plt.close()


def calc_V(W_all, IH, use_R=False, max_rel_dist=0):
    num_layer, num_heads, _, d_model, d_head = W_all.shape
    K = len(IH)

    W_qk_all = np.zeros((K, d_model, d_model))
    for i in range(K):
        Layer, Head = IH[i][0], IH[i][1]
        if use_R:
            R = calc_rotary_R_mat(
                d_head=d_head,
                max_seq_len=60,
                max_rel_dist=max_rel_dist,
            )[-1]
            W_qk = W_all[Layer, Head, 0] @ R @ W_all[Layer, Head, 1].T
        else:
            W_qk = W_all[Layer, Head, 0] @ W_all[Layer, Head, 1].T

        W_qk_all[i] = W_qk.numpy(force=True)

    U, S, Vt_common = custom_svd(W_qk_all.reshape(-1, d_model))
    return Vt_common


def Vt_to_projection(Vt, rank, project_out):
    if rank == 0:
        return np.eye(Vt.shape[1])

    V = Vt[:rank, :].T
    P = V @ V.T
    P = np.eye(P.shape[0]) - P if project_out else P
    return P


def proj_exp(
    model_name,
    batch_size,
    seg_len,
    rep,
    ignore_segment,
    ignore_burning,
    IH_proj,
    layer_head_pairs,
    component,
    proj_out,
    rank_max,
    rank_step,
):
    W_all = torch.load(f"checkpoints/{model_name}/W_all.pt")
    T_range = range(seg_len * ignore_segment + ignore_burning, rep * seg_len - 1)
    input_ids = make_input_ids(
        batch_size=batch_size,
        seg_len=seg_len,
        rep=rep,
        vocab_size=get_config(model_name).vocab_size,
        prepend_bos=model_name in ["gemma-7b", "llama2-7b", "mistral-7b"],
        bos={"llama2-7b": 1, "gemma-7b": 2, "mistral-7b": 1}.get(model_name, None),
    )

    Vt = calc_V(
        W_all,
        IH_proj,
        use_R=model_name not in ["gpt2-xl", "gpt2"],
        max_rel_dist=seg_len,
    )

    result = {}
    for rank in range(0, rank_max + 1, rank_step):
        P = Vt_to_projection(Vt, rank, proj_out)
        model = load_model(model_name)
        model_edit = projection_edit(
            model=model,
            model_name=model_name,
            layer_head_pairs=layer_head_pairs,
            component=component,
            P=P,
        )

        prob, err = inference_probs_and_errs(model_edit, input_ids)
        result[rank] = {
            "prob": prob[:, T_range],
            "err": err[:, T_range],
        }

        del model_edit, model
        torch.cuda.empty_cache()
        gc.collect()

    return result


def convert_np_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_np_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_list(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)

    return obj


def jsonify(result: dict, save_to):
    result_json = convert_np_to_list(result)
    with open(save_to, "w") as f:
        json.dump(result_json, f)

    print(f"Saved to {save_to}")


def main(
    model_name,
    component,
    proj_out,
    batch_size=50,
    seg_len=25,
    rep=3,
    ignore_segment=1,
    ignore_burning=4,
    K0=10,
    K1=30,
    rank_max=100,
    rank_step=5,
    method=None,
    seed=2024,
):

    set_seed(seed)

    if method is None:
        IH = torch.load(f"checkpoints/{model_name}/IH.pt")[:K1]
        PTH = torch.load(f"checkpoints/{model_name}/PTH.pt")[:K1]
        IH_proj = torch.load(f"checkpoints/{model_name}/IH.pt")[:K0]
        method = ""
    elif method == "subset":
        IH = torch.load(f"checkpoints/{model_name}/IH_subset.pt")
        PTH = torch.load(f"checkpoints/{model_name}/PTH_subset.pt")
        IH_proj = torch.load(f"checkpoints/{model_name}/IH_subset.pt")

    if component == "QK":
        layer_head_pairs = IH
    elif component == "OV":
        layer_head_pairs = PTH
    K0, K1 = len(IH_proj), len(layer_head_pairs)

    result = proj_exp(
        model_name=model_name,
        batch_size=batch_size,
        rep=rep,
        seg_len=seg_len,
        ignore_segment=ignore_segment,
        ignore_burning=ignore_burning,
        IH_proj=IH_proj,
        layer_head_pairs=layer_head_pairs,
        proj_out=proj_out,
        component=component,
        rank_max=rank_max,
        rank_step=rank_step,
    )

    jsonify(
        result,
        save_to=f"out/{model_name}/proj_{component}_proj_{proj_out}_{K0}_{K1}_{method}.json",
    )
    plot(
        result,
        save_to=f"out/{model_name}/Figs/{component}_proj_{proj_out}_{K0}_{K1}_{method}.png",
    )


if __name__ == "__main__":
    fire.Fire(main)
