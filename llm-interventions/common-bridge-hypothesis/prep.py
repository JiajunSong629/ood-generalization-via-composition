import json
import warnings

import fire
import numpy as np
import torch

from utils import (
    create_folder,
    get_config,
    get_qkov_weight,
    load_model,
    make_input_ids,
    set_seed,
)

warnings.filterwarnings("ignore")

EPSILON = 1e-6

np.random.seed(2024)


def get_attentions(model, input_ids):
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            cur_batch_input_ids = input_ids[i : i + 1]
            cur_attentions = model(cur_batch_input_ids).attentions
            cur_attentions = np.array(
                [a.float().numpy(force=True).mean(0) for a in cur_attentions]
            )[np.newaxis, :]
            if i == 0:
                attentions = cur_attentions
            else:
                attentions = np.vstack([attentions, cur_attentions])

    return attentions


def measure_IH_PTH(attentions, seg_len, is_IH):
    sample_size, num_layer, num_head, T, _ = attentions.shape
    scores = np.zeros((num_layer, num_head))
    offset = -(seg_len - 1) if is_IH else -1

    for layer in range(num_layer):
        for head in range(num_head):
            A = attentions[:, layer, head]
            A_adjusted = np.zeros((sample_size, T, T))
            A_adjusted[:, 1:, 1:] = A[:, 1:, 1:] / np.sum(
                A[:, 1:, 1:] + EPSILON, axis=2, keepdims=True
            )
            scores[layer, head] = np.mean(
                np.array(
                    [
                        np.mean(np.diag(A_adjusted[i], offset)[1:])
                        for i in range(sample_size)
                    ]
                )
            )

    idx_sort = np.argsort(scores, axis=None)[::-1]
    head_list = [
        [idx_sort[j] // num_head, idx_sort[j] % num_head] for j in range(len(idx_sort))
    ]

    return head_list, scores


def get_W_all(model, model_name):
    config = get_config(model_name)
    num_layer = config.num_hidden_layers
    num_head = config.num_attention_heads
    d_model = config.hidden_size
    if "head_dim" in vars(config):
        d_head = vars(config)["head_dim"]
    else:
        d_head = d_model // num_head

    W_all = torch.zeros(num_layer, num_head, 4, d_model, d_head)
    for ilayer in range(num_layer):
        for ihead in range(num_head):
            for k, component in enumerate(list("qkvo")):
                data: torch.Tensor = get_qkov_weight(
                    model, model_name, config, ilayer, ihead, component
                )
                if data.is_cuda:
                    W_all[ilayer, ihead, k] = data.cpu()
                else:
                    W_all[ilayer, ihead, k] = data

    return W_all


def jsonify(head_list, scores, save_to):
    arr = []
    for i, (L, H) in enumerate(head_list):
        arr.append({i: (int(L), int(H), scores[L, H])})

    with open(save_to, "w") as f:
        json.dump(arr, f)


def main(
    model_name,
    batch_size=50,
    seg_len=25,
    rep=2,
    seed=2024,
    fetch_w_all=True,
):
    set_seed(seed)

    model = load_model(model_name)

    input_ids = make_input_ids(
        batch_size=batch_size,
        seg_len=seg_len,
        rep=rep,
        vocab_size=get_config(model_name).vocab_size,
        prepend_bos=model_name in ["gemma-7b", "llama2-7b", "mixtral-7b", "gemma2-9b"],
        bos={"llama2-7b": 1, "gemma-7b": 2, "mistral-7b": 1, "gemma2-9b": 2}.get(
            model_name, None
        ),
    )

    attentions = get_attentions(model, input_ids)
    IH, scores_IH = measure_IH_PTH(attentions=attentions, seg_len=seg_len, is_IH=True)
    PTH, scores_PTH = measure_IH_PTH(
        attentions=attentions, seg_len=seg_len, is_IH=False
    )

    save_dir = f"checkpoints/{model_name}"
    create_folder(save_dir)
    create_folder(f"out/{model_name}")

    jsonify(IH, scores_IH, save_to=f"out/{model_name}/IH.json")
    jsonify(PTH, scores_PTH, save_to=f"out/{model_name}/PTH.json")

    torch.save(IH, f"{save_dir}/IH.pt")
    torch.save(PTH, f"{save_dir}/PTH.pt")

    if fetch_w_all:
        W_all = get_W_all(model=model, model_name=model_name)
        torch.save(W_all, f"{save_dir}/W_all.pt")


if __name__ == "__main__":
    fire.Fire(main)
