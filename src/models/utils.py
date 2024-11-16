import torch
import torch.nn.functional as F
import os
import numpy as np
import random


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
