import copy
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import gc
import json
import fire
from utils import (
    make_input_ids,
    inference_probs_and_errs,
    load_model,
    get_config,
    get_qkov_weight,
    set_seed,
)


def collect_random_layer_head_pairs(model_name, layer_head_pairs):
    config = get_config(model_name)
    num_head = config.num_attention_heads

    LH = []
    for layer, head in layer_head_pairs:
        flag = True
        while flag:
            head_rand = np.random.randint(low=0, high=num_head)
            flag = head_rand == head
        LH.append([layer, head_rand])

    np.random.shuffle(LH)
    return LH


# (5, 10), (8, 20)
# (5, 13), (8, 14)

# (5, 10) <-> (8, 14)
# (8, 20) <-> (5, 13)


def collect_components_to_copy(model, model_name, layer_head_pairs):
    config = get_config(model_name)
    components_copy = {}
    for ilayer, ihead in layer_head_pairs:
        for name in ["Q", "K", "O", "V"]:
            component_name = f"L_{ilayer}_H_{ihead}_{name}_weight"
            components_copy[component_name] = copy.deepcopy(
                get_qkov_weight(
                    model, model_name, config, ilayer, ihead, name.lower()
                ).data
            )

    return components_copy


def exchange_edit(
    model,
    model_name,
    layer_head_pairs,
    component="QK",
    type="original",
):

    K = len(layer_head_pairs)
    if type == "original":
        return model, []
    elif type == "random baseline":
        shuffle_layer_head_pairs = []
        to_copy = []
        for lh1, lh2 in zip(
            layer_head_pairs,
            collect_random_layer_head_pairs(model_name, layer_head_pairs),
        ):
            shuffle_layer_head_pairs.append([lh1, lh2])
            shuffle_layer_head_pairs.append([lh2, lh1])
            to_copy += [lh1, lh2]

    elif type == "shuffle":
        perm = torch.randperm(K)
        shuffle_layer_head_pairs = [
            [lh1, lh2]
            for lh1, lh2 in zip(
                layer_head_pairs, [layer_head_pairs[perm[j]] for j in range(K)]
            )
        ]
        to_copy = layer_head_pairs.copy()

    components_copy = collect_components_to_copy(
        model=model,
        model_name=model_name,
        layer_head_pairs=to_copy,
    )

    for (layer, head), (layer_perm, head_perm) in shuffle_layer_head_pairs:
        for name in list(component):
            component_name = f"L_{layer_perm}_H_{head_perm}_{name}_weight"
            w = get_qkov_weight(
                model=model,
                model_name=model_name,
                config=get_config(model_name),
                ilayer=layer,
                ihead=head,
                component=name.lower(),
            )
            w.copy_(components_copy[component_name])

    return model, shuffle_layer_head_pairs


def plot(result, save_to):
    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))
    for exp_id in result:
        prob, err = result[exp_id]["prob"], result[exp_id]["err"]
        avg_prob, avg_err = np.mean(prob, axis=0), np.mean(err, axis=0)
        axs[0].plot(range(len(avg_prob)), avg_prob, "-o", label=exp_id)
        axs[1].plot(range(len(avg_err)), avg_err, "-o", label=exp_id)

    titles = [f"Pred {a} under shuffling" for a in ["probs", "errs"]]
    for j in range(2):
        axs[j].set_xlabel("Token position", weight="bold")
        axs[j].set_ylabel("Target token pred probs/errs", weight="bold")
        axs[j].set_title(titles[j], weight="bold")
        axs[j].legend()

    plt.savefig(save_to, bbox_inches="tight")
    plt.close()


def shuffle_exp(
    model_name,
    batch_size,
    seg_len,
    rep,
    ignore_segment,
    ignore_burning,
    layer_head_pairs,
    component,
    n_exp,
):
    T_range = range(seg_len * ignore_segment + ignore_burning, rep * seg_len - 1)
    result = {}

    input_ids = make_input_ids(
        batch_size=batch_size,
        seg_len=seg_len,
        rep=rep,
        vocab_size=get_config(model_name).vocab_size,
        prepend_bos=model_name in ["gemma-7b", "llama2-7b", "mixtral-7b"],
        bos={"llama2-7b": 1, "gemma-7b": 2, "mistral-7b": 1}.get(model_name, None),
    )

    exp_ids = (
        ["original"]
        + [f"random baseline {i}" for i in range(1, n_exp + 1)]
        + [f"shuffle {i}" for i in range(1, n_exp + 1)]
    )

    for exp_id in exp_ids:
        if exp_id.startswith("original"):
            type = "original"
        elif exp_id.startswith("random baseline"):
            type = "random baseline"
        elif exp_id.startswith("shuffle"):
            type = "shuffle"

        model = load_model(model_name)
        model_edit, layer_head_pairs_map = exchange_edit(
            model=model,
            model_name=model_name,
            layer_head_pairs=layer_head_pairs,
            component=component,
            type=type,
        )

        prob, err = inference_probs_and_errs(model_edit, input_ids)
        result[exp_id] = {
            "prob": prob[:, T_range],
            "err": err[:, T_range],
            "shuffled_layer_head_map": layer_head_pairs_map,
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
    K=5,
    n_exp=1,
    batch_size=50,
    seg_len=25,
    rep=3,
    ignore_segment=1,
    ignore_burning=4,
    seed=2024,
    method=None,
):
    set_seed(seed)

    if method is None:
        IH = torch.load(f"checkpoints/{model_name}/IH.pt")[:K]
        PTH = torch.load(f"checkpoints/{model_name}/PTH.pt")[:K]
        method = ""
    elif method == "subset":
        IH = torch.load(f"checkpoints/{model_name}/IH_subset.pt")
        PTH = torch.load(f"checkpoints/{model_name}/PTH_subset.pt")
        method = "_subset"

    result = shuffle_exp(
        model_name=model_name,
        batch_size=batch_size,
        seg_len=seg_len,
        rep=rep,
        n_exp=n_exp,
        layer_head_pairs=IH,
        component="QK",
        ignore_burning=ignore_burning,
        ignore_segment=ignore_segment,
    )
    K = len(IH)
    jsonify(result, save_to=f"out/{model_name}/shuffle_result_QK_{K}{method}.json")
    plot(result, save_to=f"out/{model_name}/Figs/shuffle_QK_{K}{method}.png")

    result = shuffle_exp(
        model_name=model_name,
        batch_size=batch_size,
        seg_len=seg_len,
        rep=rep,
        layer_head_pairs=PTH,
        component="OV",
        n_exp=n_exp,
        ignore_burning=ignore_burning,
        ignore_segment=ignore_segment,
    )
    K = len(PTH)
    jsonify(result, save_to=f"out/{model_name}/shuffle_result_OV_{K}{method}.json")
    plot(result, save_to=f"out/{model_name}/Figs/shuffle_OV_{K}{method}.png")


if __name__ == "__main__":
    fire.Fire(main)
