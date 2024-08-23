import torch
import torch.nn.functional as F
import os
import numpy as np
import random
from transformers import (
    GPT2LMHeadModel,
    GemmaForCausalLM,
    LlamaForCausalLM,
    FalconForCausalLM,
    MistralForCausalLM,
    OlmoForCausalLM,
    AutoModelForCausalLM,
    AutoConfig,
)


def create_folder(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.


def calc_QK_OV(W_all, layer, head, QK=False, OV=False, return_original=False):
    """
    calc_QK_OV returns the W_qk or W_ov matrix at specified layer and head
    Arguments:
        W_all is a tensor of shape (num_layer, num_heads, 4, d_model, d_head) extracted from pretrained models; see definition in related function
        layer is the specified layer
        head is the specified head
    Returns:
        A 2-d torch array of size (d_model, d_model)
    """
    assert QK + OV == 1, "Only one of QK and OV should be True, the other is false"
    if QK:
        if return_original:
            return W_all[layer, head, 0], W_all[layer, head, 1]
        W_qk = W_all[layer, head, 0] @ W_all[layer, head, 1].T
        return W_qk
    else:
        if return_original:
            return W_all[layer, head, 3], W_all[layer, head, 2]
        W_ov = W_all[layer, head, 3] @ W_all[layer, head, 2].T
        return W_ov


def custom_svd(M):
    """
    assume M = (n, p) and n << p
    U @ S @ V.t = M

    1. M @ M.t = U @ S^2 @ U.t
    2. V.t = inv(S) @ U.t @ M
    3. V = M.t @ U @ inv(S)
    returns U, S, V.T
    """
    n, p = M.shape
    if n > p:
        V, S, Ut = custom_svd(M.T)
        return Ut.T, S, V.T

    G = M @ M.T  # n, n
    S2, U = np.linalg.eigh(G)
    S = np.sqrt(np.abs(S2))

    sorted_indices = np.argsort(S)[::-1]
    S = S[sorted_indices]
    U = U[:, sorted_indices]

    V = M.T @ U @ np.diag(1 / S)

    return U, S, V.T


def svdAB(A, B):
    """
    returns the SVD of A@B.T

    UaSaVa = A, UbSbVb = B.T
    C = SaVaUbSb

    UcScVc = C

    returns UaUc, Sc, VcVb
    """
    Ua, Sa, Vat = custom_svd(A)
    Ub, Sb, Vbt = custom_svd(B.T)
    C = np.diag(Sa) @ Vat @ Ub @ np.diag(Sb)
    Uc, Sc, Vct = custom_svd(C)
    return Ua @ Uc, Sc, Vct @ Vbt


def print_gpu_mem_usage():
    torch.cuda.reset_peak_memory_stats(device=None)
    used = torch.cuda.max_memory_allocated(device=None) / np.power(2, 30)
    print(f"gpu used {used: .3f}G memory")


def load_model(model_name):
    def load_gpt2():
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2", output_attentions=True, device_map="cuda"
        )
        return model

    def load_gpt2xl():
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2-xl",
            output_attentions=True,
            device_map="cuda",
        )
        return model

    def load_llama2_7b():
        llama = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return llama
    
    def load_llama3_8b():
        llama = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return llama
    
    def load_llama2_70b():
        llama = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-70b-hf",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        return llama

    def load_gemma_7b():
        gemma = GemmaForCausalLM.from_pretrained(
            "google/gemma-7b",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return gemma

    def load_gemma2_9b():
        gemma = GemmaForCausalLM.from_pretrained(
            "google/gemma-2-9b",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return gemma

    def load_falcon_7b():
        falcon = FalconForCausalLM.from_pretrained(
            "tiiuae/falcon-7b",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return falcon
    
    def load_falcon2_11b():
        falcon = FalconForCausalLM.from_pretrained(
            "tiiuae/falcon-11B",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return falcon

    def load_mistral_7b():
        mistral = MistralForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return mistral

    def load_olmo_7b():
        olmo = OlmoForCausalLM.from_pretrained(
            "allenai/OLMo-1.7-7B-hf",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return olmo

    def load_pythia_7b():
        pythia = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-6.9b",
            output_attentions=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        return pythia

    return {
        "gpt2": load_gpt2,
        "gpt2-xl": load_gpt2xl,
        "llama2-7b": load_llama2_7b,
        "llama3-8b": load_llama3_8b,
        "llama2-70b": load_llama2_70b,
        "gemma-7b": load_gemma_7b,
        "gemma-2-9b": load_gemma2_9b,
        "falcon-7b": load_falcon_7b,
        "falcon2-11b": load_falcon2_11b,
        "mistral-7b": load_mistral_7b,
        "olmo-7b": load_olmo_7b,
        "pythia-7b": load_pythia_7b,
    }[model_name]()


def get_config(model_name):
    model_name_hf = {
        "gpt2": "gpt2",
        "gpt2-xl": "gpt2-xl",
        "llama2-7b": "meta-llama/Llama-2-7b-hf",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama2-70b": "meta-llama/Llama-2-70b-hf",
        "gemma-7b": "google/gemma-7b",
        "gemma-2-9b": "google/gemma-2-9b",
        "falcon-7b": "tiiuae/falcon-7b",
        "falcon2-11b": "tiiuae/falcon-11B",
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "olmo-7b": "allenai/OLMo-1.7-7B-hf",
        "pythia-7b": "EleutherAI/pythia-6.9b",
    }[model_name]
    config = AutoConfig.from_pretrained(model_name_hf)

    return config


def get_qkov_weight(model, model_name, config, ilayer, ihead, component):
    num_head = config.num_attention_heads
    d_model = config.hidden_size
    d_head = d_model // num_head

    if model_name in ["gpt2", "gpt2-xl"]:
        attn = model.transformer.h[ilayer].attn.c_attn
        proj = model.transformer.h[ilayer].attn.c_proj
        return {
            "q": attn.weight[:, ihead * d_head : ihead * d_head + d_head],
            "k": attn.weight[
                :, d_model + ihead * d_head : d_model + ihead * d_head + d_head
            ],
            "v": attn.weight[
                :, d_model * 2 + ihead * d_head : d_model * 2 + ihead * d_head + d_head
            ],
            "o": proj.weight[ihead * d_head : ihead * d_head + d_head, :].T,
        }[component].data

    elif model_name in ["llama2-7b", "gemma-7b", "gemma-2-9b", "mistral-7b", "olmo-7b"]:
        if model_name == "mistral-7b":
            ikvhead = ihead * config.num_key_value_heads // num_head
        else:
            ikvhead = ihead

        attn = model.model.layers[ilayer].self_attn
        return {
            "q": attn.q_proj.weight.T[:, ihead * d_head : ihead * d_head + d_head],
            "k": attn.k_proj.weight.T[:, ikvhead * d_head : ikvhead * d_head + d_head],
            "v": attn.v_proj.weight.T[:, ikvhead * d_head : ikvhead * d_head + d_head],
            "o": attn.o_proj.weight.T[ihead * d_head : ihead * d_head + d_head, :].T,
        }[component].data
    elif model_name in ["falcon-7b"]:
        attn = model.transformer.h[ilayer].self_attention
        splited = attn.query_key_value.weight.view(2 + num_head, d_head, d_model)
        q, k, v = splited[:-2], splited[-2], splited[-1]
        return {
            "q": q[ihead].T,
            "k": k.T,
            "v": v.T,
            "o": attn.dense.weight.T[ihead * d_head : ihead * d_head + d_head, :].T,
        }[component].data


def make_input_ids(
    batch_size,
    seg_len,
    rep,
    vocab_size,
    prepend_bos=False,
    bos=None,
):

    # draw batch of random tokens and make repetitions
    sample_int = np.random.randint(
        low=0, high=vocab_size, size=batch_size * seg_len
    ).reshape(batch_size, seg_len)
    sample_int = np.concatenate(tuple([sample_int] * rep), axis=1)

    if prepend_bos:
        sample_int = np.hstack([bos * np.ones((batch_size, 1), dtype=int), sample_int])

    input_ids = torch.Tensor(sample_int).long()

    return input_ids.cuda()


def inference_probs_and_errs(model, input_ids):
    with torch.no_grad():
        for i in range(input_ids.size(0)):
            cur_batch_input_ids = input_ids[i : i + 1]
            cur_logits = model(cur_batch_input_ids).logits.cpu()
            if i == 0:
                logits = cur_logits
            else:
                logits = torch.concat([logits, cur_logits])

    probs = F.softmax(logits.float(), dim=-1)
    _, pred_next_token_ids = torch.topk(probs, dim=-1, k=1)

    err = (input_ids[:, 1:].cpu() != pred_next_token_ids[:, :-1, 0]).numpy(force=True)

    probs_on_correct = np.zeros_like(input_ids[:, 1:].numpy(force=True))
    batch_size, seq_len, n_vocab = probs.shape
    probs_on_correct = np.zeros((batch_size, seq_len - 1))
    for b in range(batch_size):
        for s in range(seq_len - 1):
            probs_on_correct[b, s] = probs[b, s, input_ids[b, s + 1]]

    return probs_on_correct, err


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Rotary embedding helper function"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (freqs_cis.shape, x.shape)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def calc_rotary_R_mat(
    d_head: int,
    max_seq_len: int,
    theta: float = 10000.0,
    max_rel_dist: int = 10,
):
    """
    calc_rotary_R_mat calculates R matrices for computing QK. The formula of QK computation is Q R K^T where
        R is a (d_head, d_head) shaped matrix that depends on the relative distance between two tokens.
        We are interested in precomputing each R matrix for relative distance in {0, 1, ... max_rel_dist-1}
    Args:
        d_head: head dimension, note that R matrix does not depend on d_model, only depend on d_head
        max_seq_len: the maximum sequence length a Transformer accepts
        theta: a frequency related hyperparameter used for computing R matrices
        max_rel_dist: the maximum relative distance we are interested in precompuing
    Returns:
        Rotary_matrices: a 3-order tensor containing every R matrix with relative distance in {0, 1, ... max_rel_dist-1}
    Usage:
        R_all = calc_rotary_R_mat(d_head, 512) gives 10 R matrices
        sns.heatmap(R_all[9].numpy(), vmin=-1, vmax=1, cmap='bwr') shows heatmap visualization;
            most of them are sparse, very similar to identity matrix
        Given W_q of shape (d_model, d_head) and W_k of shape (d_model, d_head),
        W_q @ R[t] W_k.T gives rotated QK for relative distance t
    """
    freqs_cis = precompute_freqs_cis(d_head, max_seq_len * 2, theta)

    Rotary_matrices = torch.zeros(max_rel_dist, d_head, d_head)
    for j in range(d_head):
        for k in range(d_head):
            # construct canonical basis e_j, e_k
            E1 = torch.zeros(max_rel_dist, d_head)
            E1[:, j] = 1
            E2 = torch.zeros(max_rel_dist, d_head)
            E2[:, k] = 1
            E1, E2 = apply_rotary_emb(
                E1.unsqueeze(1).unsqueeze(0),
                E2.unsqueeze(1).unsqueeze(0),
                freqs_cis=freqs_cis[:max_rel_dist],
            )
            R12 = E1.squeeze() @ E2.squeeze().T
            # first row of R12 contains (j,k) entry of R matrix for positions (0, t), where 0<=t<max_rel_dist
            Rotary_matrices[:, j, k] = R12[0]

    return Rotary_matrices
