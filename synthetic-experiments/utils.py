# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import random


#####################################################
################# Simple utitlity function  ###################
#####################################################


def create_folder(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)


def fix_random_seed(seed, reproduce=False):
    # cudnn.enabled = True
    # cudnn.benchmark = True

    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: uncomment for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ## NOTE: uncomment for pytorch >= 1.8
        # torch.use_deterministic_algorithms(True)

    # os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)

    return rng


#####################################################


def gen_tran_mat(vocab_size, order, sig=1, sparsity=None):
    mat = torch.exp(sig * torch.randn(tuple([vocab_size] * (order + 1))))
    mat = mat / mat.sum(dim=-1, keepdim=True)
    if sparsity is not None:
        cutoff = torch.quantile(mat.flatten(), 1 - sparsity)
        mat[mat < cutoff] = 0
        mat = mat / mat.sum(dim=-1, keepdim=True)
    return mat


def calc_opt_err(mat):
    """
    Given a transition probability matrix in a Markov chain, calculate the optimal achievable error
    Args:
        mat: the transition probability matrix, will check its validity
    Returns:
        err_opt: scalar, the optimal error under equilibrium distribution
        pi: 1d array, equilibrium distribution
    """

    m = mat.size(0)
    m1, m2 = mat.shape
    assert m1 == m2, "Incorrect input dimension of transition matrix"
    assert torch.all(mat >= 0) and torch.all(
        torch.abs(mat.sum(dim=1) - 1) < 1e-6
    ), "Incorrect input of transition matrix"

    vals, vecs = np.linalg.eig(mat.numpy().T)
    idx = np.argsort(vals)
    pi = np.real(vecs[:, idx[-1]])  # equilibrium distribution
    pi = pi / np.sum(pi)  # don't forget to normalize so that it sums to one
    err_opt = np.dot(pi, 1 - mat.max(dim=1)[0].numpy())

    return err_opt, pi


def get_mat_full(mat, order=2):
    """
    For second-order or third-order markov chains, get_mat_full will
    1) if order=2, convert the transition matrix from the tensor form K*K*K to the matrix form (K^2) * (K^2)
    2) if order=3, convert the transition matrix from the tensor form K*K*K*K to the matrix form (K^3) * (K^3)
    Args:
        mat: the transition probability tensor, will check its validity
    Returns:
        mat_full:  transition probability matrix
    """
    if order == 2:
        m1, m2, m3 = mat.shape
        vocab_size = m1
        assert (m1 == m2) and (
            m1 == m3
        ), "Incorrect input dimension of transition matrix"
        assert torch.all(mat >= 0) and torch.all(
            torch.abs(mat.sum(dim=2) - 1) < 1e-6
        ), "Incorrect input of transition matrix"
        mat_full = torch.zeros(vocab_size**2, vocab_size**2).float()
        for k1 in range(vocab_size):
            for k2 in range(vocab_size):
                k = k1 * vocab_size + k2
                k_out = k2 * vocab_size + torch.arange(vocab_size).long()
                mat_full[k, k_out] = mat[k1, k2, :]
    elif order == 3:
        m1, m2, m3, m4 = mat.shape
        vocab_size = m1
        assert (
            (m1 == m2) and (m1 == m3) and (m1 == m4)
        ), "Incorrect input dimension of transition matrix"
        assert torch.all(mat >= 0) and torch.all(
            torch.abs(mat.sum(dim=3) - 1) < 1e-6
        ), "Incorrect input of transition matrix"
        mat_full = torch.zeros(vocab_size**3, vocab_size**3).float()
        for k1 in range(vocab_size):
            for k2 in range(vocab_size):
                for k3 in range(vocab_size):
                    k = k1 * (vocab_size**2) + k2 * vocab_size + k3
                    k_out = (
                        k2 * (vocab_size**2)
                        + k3 * vocab_size
                        + torch.arange(vocab_size).long()
                    )
                    mat_full[k, k_out] = mat[k1, k2, k3, :]
    else:
        warnings.warn("The order argument receives an incorrect input.")

    return mat_full


def mask_get_along_axis(shape, indices):
    assert shape[0] == len(
        indices
    ), "length of indices show match number of rows in shape"
    mask = np.zeros(shape)
    return np.array(
        [
            np.concatenate((np.zeros(indices[i]), np.ones(shape[1] - indices[i])))
            for i in range(len(indices))
        ]
    )


def mask_get_given_starts(shape, lens, starts, ignore_segment=0, ignore_burning=0):
    n, L = shape[0], shape[1]
    n1 = len(lens)
    n2, rep = starts.shape
    assert n == n1 and n == n2, "Wrong input shapes"
    mask = np.zeros((n, L), dtype=int)
    for i in range(n):
        for j in range(ignore_segment, rep):
            mask[i, (starts[i, j] + ignore_burning) : (starts[i, j] + lens[i])] = 1
    return mask


#####################################################
##################### Making plots ######################
#####################################################


def plot_err_curve(
    err_arr,
    setting_params=None,
    fig_name=None,
    save_dir=None,
    opt_err=None,
    plot_ood=False,
    plot_train=True,
    log_training_time=False,
):
    """
    A simple function to make plots based on err_arr, optionally saving plots in the specified folder
    Args:
        err_arr: a numpy array of size num_epoch-by-6, containing train/test/ood-test loss/errors
        setting_params: a dictionary containing setting parameters such as vocab_size, max_seq_len
        opt_err: a optional 1d array showing the optimal achievable error
        plot_ood: if true, plots the loss/error curces for ood test data
    """
    num_epoch = (
        setting_params["num_epoch"] if setting_params is not None else err_arr.shape[0]
    )
    if fig_name is not None:
        if save_dir is None:
            if not os.path.isdir("Figs"):
                os.mkdir("Figs")
            save_path = os.path.join("Figs", fig_name)
        else:
            save_path = os.path.join(save_dir, fig_name)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    if plot_train:
        axs[0].plot(
            np.arange(num_epoch, dtype=int),
            err_arr[:, 0],
            linewidth=2,
            label="train loss",
        )
    axs[0].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 2], linewidth=2, label="test loss"
    )
    if plot_ood:
        axs[0].plot(
            np.arange(num_epoch, dtype=int),
            err_arr[:, 4],
            linewidth=2,
            label="ood test loss",
        )
    axs[0].set_yscale("log")
    axs[0].set_title(
        f"Train/test loss, last/best test epoch {err_arr[-1,1]:.3f}, {np.min(err_arr[:,1]):.3f}",
        weight="bold",
    )
    axs[0].set_xlabel("Epochs", weight="bold")
    if plot_train:
        axs[1].plot(
            np.arange(num_epoch, dtype=int),
            err_arr[:, 1],
            linewidth=2,
            label="train err",
        )
    axs[1].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 3], linewidth=2, label="test err"
    )
    if plot_ood:
        axs[1].plot(
            np.arange(num_epoch, dtype=int),
            err_arr[:, 5],
            linewidth=2,
            label="ood test err",
        )
    if opt_err is not None:
        axs[1].plot(
            np.arange(num_epoch, dtype=int),
            np.repeat(opt_err, num_epoch),
            linestyle="dashed",
            label="optimal err",
        )
    axs[1].legend()
    axs[1].set_title(
        f"Train/test error, last/best test epoch {err_arr[-1,3]:.3f}, {np.min(err_arr[:,3]):.3f}",
        weight="bold",
    )
    axs[1].set_xlabel("Epochs", weight="bold")
    # axs[1].axhline(y=0.5, xmin=0, xmax=num_epoch, linestyle="dashed", c="black")

    if log_training_time:
        axs[0].set_xscale("log")
        axs[1].set_xscale("log")

    if fig_name is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")


def plot_err_curve_hmm(
    err_arr,
    setting_params=None,
    fig_name=None,
    save_dir=None,
    opt_err=None,
    plot_ood=False,
):
    """
    A simple function to make plots based on err_arr. Similar to plot_err_curve, but also plots errors based on hmm
    Args:
        err_arr: a numpy array of size num_epoch-by-9, containing train/test/ood-test loss/errors, and IOI train/test/ood-test errors
        setting_params: a dictionary containing setting parameters such as vocab_size, max_seq_len
        opt_err: a optional 1d array showing the optimal achievable error
        plot_ood: if true, plots the loss/error curces for ood test data
    """
    num_epoch = (
        setting_params["num_epoch"] if setting_params is not None else err_arr.shape[0]
    )
    if fig_name is not None:
        if save_dir is None:
            if not os.path.isdir("Figs"):
                os.mkdir("Figs")
            save_path = os.path.join("Figs", fig_name)
        else:
            save_path = os.path.join(save_dir, fig_name)

    fig, axs = plt.subplots(2, 3, figsize=(3 * 6, 2 * 6))
    axs[0, 0].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 0], linewidth=2, label="train loss"
    )
    axs[0, 0].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 2], linewidth=2, label="test loss"
    )
    if plot_ood:
        axs[0, 0].plot(
            np.arange(num_epoch, dtype=int),
            err_arr[:, 4],
            linewidth=2,
            label="ood test loss",
        )
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_title(
        f"Train/test loss, last/best test epoch {err_arr[-1,1]:.3f}, {np.min(err_arr[:,1]):.3f}",
        weight="bold",
    )
    axs[0, 0].set_xlabel("Epochs", weight="bold")
    axs[0, 1].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 1], linewidth=2, label="train err"
    )
    axs[0, 1].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 3], linewidth=2, label="test err"
    )
    if plot_ood:
        axs[0, 1].plot(
            np.arange(num_epoch, dtype=int),
            err_arr[:, 5],
            linewidth=2,
            label="ood test err",
        )
    if opt_err is not None:
        axs[0, 1].plot(
            np.arange(num_epoch, dtype=int),
            np.repeat(opt_err, num_epoch),
            linestyle="dashed",
            label="optimal err",
        )
    axs[0, 1].legend()
    axs[0, 1].set_title(
        f"Train/test error, last/best test epoch {err_arr[-1,3]:.3f}, {np.min(err_arr[:,3]):.3f}",
        weight="bold",
    )
    axs[0, 1].set_xlabel("Epochs", weight="bold")
    axs[0, 2].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 6], linewidth=2, label="train err"
    )
    axs[0, 2].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 7], linewidth=2, label="test err"
    )
    if plot_ood:
        axs[0, 2].plot(
            np.arange(num_epoch, dtype=int),
            err_arr[:, 8],
            linewidth=2,
            label="ood test loss",
        )
    axs[0, 2].set_yscale("log")
    axs[0, 2].set_title(
        f"Train/test loss only for IOI, last/best test epoch {err_arr[-1,7]:.3f}, {np.min(err_arr[:,7]):.3f}",
        weight="bold",
    )
    axs[0, 2].set_xlabel("Epochs", weight="bold")
    axs[1, 0].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 9], linewidth=2, label="train err"
    )
    axs[1, 0].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 10], linewidth=2, label="test err"
    )
    if plot_ood:
        axs[1, 0].plot(
            np.arange(num_epoch, dtype=int),
            err_arr[:, 11],
            linewidth=2,
            label="ood test loss",
        )
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_title(
        f"Train/test state prediction errors, last/best test epoch {err_arr[-1,10]:.3f}, {np.min(err_arr[:,10]):.3f}",
        weight="bold",
    )
    axs[1, 0].set_xlabel("Epochs", weight="bold")
    axs[1, 1].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 12], linewidth=2, label="train err"
    )
    axs[1, 1].plot(
        np.arange(num_epoch, dtype=int), err_arr[:, 13], linewidth=2, label="test err"
    )
    if plot_ood:
        axs[1, 1].plot(
            np.arange(num_epoch, dtype=int),
            err_arr[:, 14],
            linewidth=2,
            label="ood test loss",
        )
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title(
        f"Train/test tran matrix prediction errors, last/best test epoch {err_arr[-1,13]:.3f}, {np.min(err_arr[:,13]):.3f}",
        weight="bold",
    )
    axs[1, 1].set_xlabel("Epochs", weight="bold")
    if fig_name is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")


def plot_attention(
    model,
    tokens,
    fig_name,
    norm=True,
    pos=None,
    savefig_dir="Figs",
    use_mask=True,
    num_heads=1,
    layer=0,
):
    """
    This function makes two plots, namely attention plot and QK value heatmap
    Args:
        model: the simpleT model we use for the simulations
        tokens: a sequence of tokens, where each token is any element of type torch.long in the vocabulary
        fig_name: name of figure when saving plots
        is_mask: if True, use a mask when calculating QK and attention for next-token prediction
        num_heads: number of attention heads in the model
    Returns:
        QK_vals: the pre-softmax QK values, torch.Tensor 2-d array
        attn: attentions, numpy 2-d array, normalized to sum 1

    """
    model.eval()
    seq = (
        model.pos_embed(model.embed(tokens.unsqueeze(0)))
        if pos not in ["rotary", "relative"]
        else model.embed(tokens.unsqueeze(0))
    )
    _, seq_len, d_model = seq.size()
    d_k = d_model // num_heads
    if d_model % num_heads != 0:
        warnings.warn("d_model is not divisible by num_heads!")
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(model.device)

    h = seq
    for layer0 in range(layer):
        h = model.h[layer0](h, mask=mask)
    h = model.h[layer].ln_1(h) if norm else h
    queries = model.h[layer].mha.W_q(h)
    keys = model.h[layer].mha.W_k(h)
    values = model.h[layer].mha.W_v(h)
    queries = queries.view(1, seq_len, num_heads, d_k).transpose(1, 2)
    keys = keys.view(1, seq_len, num_heads, d_k).transpose(1, 2)
    values = values.view(1, seq_len, num_heads, d_k).transpose(1, 2)

    _, QKval = model.h[layer].mha.scaled_dot_product_attention(
        queries, keys, values, mask=mask
    )
    attn, QK_vals = QKval[0].squeeze(dim=0).numpy(force=True), QKval[1].squeeze(
        dim=0
    ).numpy(force=True)

    ## making plots now
    fig, axs = plt.subplots(num_heads, 3, figsize=(3 * 9, num_heads * 9))
    width = 1
    example_sep = 2
    word_height = 1
    pad = 0.1
    yoffset = 1
    xoffset = 0

    for head in range(num_heads):
        plot_idx = (head, 0) if num_heads > 1 else 0
        """
        for position, token in enumerate(tokens.numpy()):
            axs[plot_idx].text(xoffset + 0,
                     yoffset - position * word_height,
                     token,
                     ha="right",
                     va="center")
            axs[plot_idx].text(xoffset + width,
                     yoffset - position * word_height,
                     token,
                     ha="left",
                     va="center")
        axs[plot_idx].text(xoffset + 0.5 * width,
                 3,
                 "",
                 ha="center",
                 va="top",
                 weight="bold")
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                axs[plot_idx].plot(
                    [xoffset + pad, xoffset + width - pad],
                    [yoffset - word_height * i, yoffset - word_height * j],
                    color="blue",
                    linewidth=1,
                    alpha=attn[head, i, j])
                axs[plot_idx].set_title(f'Post-softmax attentions: head {head}',weight="bold",fontsize=25)
        """

        plot_idx = (head, 1) if num_heads > 1 else 1
        pcm = axs[plot_idx].imshow(QK_vals[head, :, :])
        axs[plot_idx].set_title(
            f"Pre-softmax QK values: head {head}", weight="bold", fontsize=25
        )
        fig.colorbar(pcm, ax=axs[plot_idx], shrink=0.8)

        plot_idx = (head, 2) if num_heads > 1 else 2
        pcm = axs[plot_idx].imshow(attn[head, :, :])
        axs[plot_idx].set_title(
            f"Attention values: head {head}", weight="bold", fontsize=25
        )
        fig.colorbar(pcm, ax=axs[plot_idx], shrink=0.8)
    plt.savefig(os.path.join(savefig_dir, fig_name), bbox_inches="tight")
    plt.close()
    return attn, QK_vals


def plot_incoh_heatmat(save_dir, model, setting_params, remove_firstlast=True):
    train_embed_str = "trainEmbed_" if setting_params["train_embed"] else ""
    add_embed_str = "addEmbed_" if setting_params["add_embed"] else ""
    MLP_str = "MLP_" if setting_params["use_MLP"] else ""
    sig = setting_params["sig"]
    plt_save_name = MLP_str + train_embed_str + add_embed_str + f"sig_{sig}_incoh"

    fig, ax = plt.subplots(2, 2, figsize=(16, 16))

    W_e = model.embed.embed.weight.detach().numpy()
    Gram_e = W_e @ W_e.T
    pcm = ax[0, 0].imshow(Gram_e)
    fig.colorbar(pcm, ax=ax[0, 0], shrink=0.8)
    ax[0, 0].set_title("Gram matrix of static embed matrix", weight="bold")

    W_p = model.pos_embed.pe.weight.detach().numpy()
    W_p = W_p[1:-1] if remove_firstlast else W_p
    Gram_p = W_p @ W_p.T
    pcm = ax[0, 1].imshow(Gram_p)
    fig.colorbar(pcm, ax=ax[0, 1], shrink=0.8)
    ax[0, 1].set_title("Gram matrix of positional embed matrix", weight="bold")

    u, s, vt = np.linalg.svd(W_e)
    ax[1, 0].plot(s)
    ax[1, 0].set_xlabel("index")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("Spectrum of the static embed matrix", weight="bold")

    u, s, vt = np.linalg.svd(W_p)
    ax[1, 1].plot(s)
    ax[1, 1].set_xlabel("index")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("Spectrum of the positional embed matrix", weight="bold")

    plt.savefig(os.path.join(save_dir, plt_save_name), bbox_inches="tight")


def plot_err_over_pos(
    model,
    src_list,
    vocab_size,
    fig_name,
    criterion=nn.CrossEntropyLoss(reduction="none"),
    lens=None,
    src_labels=None,
    starts=None,
    return_predictions=False,
    ignore_Aa=False,
    save_dir="Figs",
):
    """
    This function makes a plot to show the error of next-token prediction at each position
    Args:
        model: the simple TF model we use for the simulations
        src_list: a list of datasets to test the model, 2D Tensor
        vocab_size: size of vocabulary
        fig_name: name of figure when saving plots
        lens: a list/array of number of positions to ignore when calculting errors (useful when we want to exclude tokens not yet repeated)
        return_predictions: if True, return predictions for each seq in src_list
        ignore_Aa: (useful in Aa setting) if True, then case sensitivitity is ignored when calculating errors
    Returns:
        loss_list:  a list of arrays of length T-1 (max_seq_len-1), average loss at each position
        err_list: a list of arrays of length T-1 (max_seq_len-1), average error at each position
        pred_list: if return_predictions is True, a list of array of length T-1

    """
    # if fig_name is not None:
    #     if save_dir is None:
    #         if not os.path.isdir("Figs"):
    #             os.mkdir("Figs")
    #         save_path = os.path.join("Figs", fig_name)
    #     else:
    #         save_path = os.path.join(save_dir, fig_name)
    # src_labels = [None] * len(src_list) if src_labels is None else src_labels
    # lens = [None] * len(src_list) if lens is None else lens
    # vocab_halfsize = vocab_size // 2
    # eps = 1e-6
    # model.eval()
    # loss_list = []
    # err_list = []
    # pred_list = []

    # for i, src in enumerate(src_list):
    #     N, T = src.size()
    #     M = torch.zeros(src.shape)
    #     if lens[i] is not None:
    #         M = torch.tensor(mask_get_along_axis(src.shape, lens[i]), device=src.device)
    #     with torch.no_grad():
    #         output = model(src)
    #         loss = (
    #             criterion(
    #                 output[:, :-1].contiguous().view(-1, vocab_size),
    #                 src[:, 1:].contiguous().view(-1),
    #             )
    #             .reshape(N, T - 1)
    #             .mean(dim=0)
    #         )
    #         pred = output.argmax(dim=2)[:, :-1]
    #         if ignore_Aa:
    #             tmp = (
    #                 (pred == src[:, 1:])
    #                 | (pred == src[:, 1:] + vocab_halfsize)
    #                 | (pred == src[:, 1:] - vocab_halfsize)
    #             )
    #         else:
    #             tmp = pred == src[:, 1:]
    #         err = 1 - torch.sum(tmp * M[:, :-1], dim=0) / (
    #             torch.sum(M[:, :-1], dim=0) + eps
    #         )
    #     loss_list.append(loss.numpy(force=True))
    #     err_list.append(err.numpy(force=True))
    #     pred_list.append(pred.numpy(force=True))

    # fig, axs = plt.subplots(2, 1, figsize=(10, 6 * 2))
    # for i, src in enumerate(src_list):
    #     axs[0].plot(
    #         np.arange(T - 1, dtype=int), loss_list[i], "-o", label=src_labels[i]
    #     )
    #     axs[1].plot(np.arange(T - 1, dtype=int), err_list[i], "-o", label=src_labels[i])
    # axs[0].set_xlabel("Position", weight="bold")
    # axs[0].set_ylabel("Loss", weight="bold")
    # axs[0].set_title(
    #     "Averaged next-token prediction loss at each position", weight="bold"
    # )
    # axs[1].set_xlabel("Position", weight="bold")
    # axs[1].set_ylabel("Error", weight="bold")
    # axs[1].set_title(
    #     "Averaged next-token prediction error at each position", weight="bold"
    # )
    # axs[0].legend()
    # axs[1].legend()

    # if fig_name is None:
    #     plt.show()
    # else:
    #     plt.savefig(save_path, bbox_inches="tight")

    # out = (
    #     (loss_list, err_list, pred_list)
    #     if return_predictions
    #     else (loss_list, err_list)
    # )
    # return out
    if fig_name is not None:
        if save_dir is None:
            if not os.path.isdir("Figs"):
                os.mkdir("Figs")
            save_path = os.path.join("Figs", fig_name)
        else:
            save_path = os.path.join(save_dir, fig_name)
    src_labels = [None] * len(src_list) if src_labels is None else src_labels
    lens = [None] * len(src_list) if lens is None else lens
    vocab_halfsize = vocab_size // 2
    eps = 1e-6
    model.eval()
    loss_list = []
    err_list = []
    err2_list = []
    pred_list = []

    for i, src in enumerate(src_list):
        N, T = src.size()
        M = torch.zeros(src.shape)
        if lens[i] is not None:
            M = torch.tensor(mask_get_given_starts(src.shape, lens[i], starts[i]))
        with torch.no_grad():
            output = model(src)
            loss = (
                criterion(
                    output[:, :-1].contiguous().view(-1, vocab_size),
                    src[:, 1:].contiguous().view(-1),
                )
                .reshape(N, T - 1)
                .mean(dim=0)
            )
            pred = output.argmax(dim=2)[:, :-1]
            if ignore_Aa:
                tmp = (
                    (pred == src[:, 1:])
                    | (pred == src[:, 1:] + vocab_halfsize)
                    | (pred == src[:, 1:] - vocab_halfsize)
                )
            else:
                tmp = pred == src[:, 1:]
        err = 1 - torch.sum(tmp.cpu() * M[:, :-1], dim=0) / (
            torch.sum(M[:, :-1], dim=0) + eps
        )  # averaged err at each posiiton
        err2 = torch.zeros(vocab_size)
        for j in range(vocab_size):
            err2[j] = 1 - torch.sum((tmp * (src[:, :-1] == j)).cpu() * M[:, :-1]) / (
                torch.sum((src[:, :-1] == j).cpu() * M[:, :-1]) + eps
            )  # averaged err at each token
        loss_list.append(loss.numpy(force=True))
        err_list.append(err.numpy(force=True))
        err2_list.append(err2.numpy(force=True))
        pred_list.append(pred.numpy(force=True))

    fig, axs = plt.subplots(3, 1, figsize=(10, 6 * 3))
    for i, src in enumerate(src_list):
        axs[0].plot(
            np.arange(T - 1, dtype=int), loss_list[i], "-o", label=src_labels[i]
        )
        axs[1].plot(np.arange(T - 1, dtype=int), err_list[i], "-o", label=src_labels[i])
        axs[2].plot(
            np.arange(vocab_size, dtype=int), err2_list[i], "-o", label=src_labels[i]
        )
    axs[0].set_xlabel("Position", weight="bold")
    axs[0].set_ylabel("Loss", weight="bold")
    axs[0].set_title(
        "Averaged next-token prediction loss at each position", weight="bold"
    )
    axs[1].set_xlabel("Position", weight="bold")
    axs[1].set_ylabel("Error", weight="bold")
    axs[1].set_title(
        "Averaged next-token prediction error at each position", weight="bold"
    )
    axs[2].set_xlabel("Token", weight="bold")
    axs[2].set_ylabel("Error", weight="bold")
    axs[2].set_title(
        "Averaged next-token prediction error at each token", weight="bold"
    )
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    if fig_name is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    out = (
        (loss_list, err_list, err2_list, pred_list)
        if return_predictions
        else (loss_list, err_list)
    )
    return out


def plot_qk_subspace_matching(model, fig_name, config):
    num_svals_plot = 32
    W_q = model.h[1].mha.W_q.weight.numpy(force=True)
    W_k = model.h[1].mha.W_k.weight.numpy(force=True)
    W_v = model.h[0].mha.W_v.weight.numpy(force=True)
    W_o = model.h[0].mha.W_o.weight.numpy(force=True)
    W_qk = W_q.T @ W_k / np.sqrt(config.d_model)
    W_ov = W_o @ W_v
    U_qk, s_qk, Vt_qk = np.linalg.svd(W_qk)
    U_ov, s_ov, Vt_ov = np.linalg.svd(W_ov)

    s_match = np.zeros((2, num_svals_plot))
    for j in range(num_svals_plot):
        _, s, _ = np.linalg.svd(Vt_qk[: (j + 1), :] @ U_ov[:, : (j + 1)])
        _, s2, _ = np.linalg.svd(Vt_ov[: (j + 1), :] @ U_qk[:, : (j + 1)])
        s_match[0, j] = s[0]
        s_match[1, j] = s2[0]

    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))
    axs[0].plot(s_qk[:num_svals_plot] / s_qk[0], "-o", label="qk", linewidth=2)
    axs[0].plot(s_ov[:num_svals_plot] / s_ov[0], "-o", label="ov", linewidth=2)
    axs[0].plot(s_match[0, :num_svals_plot], "-o", label="match")
    axs[0].legend()
    axs[0].set_title("inner match")
    axs[1].plot(s_qk[:num_svals_plot] / s_qk[0], "-o", label="qk", linewidth=2)
    axs[1].plot(s_ov[:num_svals_plot] / s_ov[0], "-o", label="ov", linewidth=2)
    axs[1].plot(s_match[1, :num_svals_plot], "-o", label="match")
    axs[1].legend()
    axs[1].set_title("outer match")

    plt.savefig(fig_name)
    plt.close()


def plot_wqkov(model, fig_name, config):
    W_q = model.h[1].mha.W_q.weight.numpy(force=True)
    W_k = model.h[1].mha.W_k.weight.numpy(force=True)
    W_v = model.h[0].mha.W_v.weight.numpy(force=True)
    W_o = model.h[0].mha.W_o.weight.numpy(force=True)
    W_qk = W_q.T @ W_k / np.sqrt(config.d_model)
    W_ov = W_o @ W_v
    W_qkov = W_qk @ W_ov

    ax = sns.heatmap(W_qkov, square=True)
    plt.savefig(fig_name)
    plt.close()


#####################################################
################# Calculating errors ################
#####################################################


def hmm_calc_err(
    src_list, output_list, states_list, state_sizes, transition_mat, contains_ood=True
):
    """
    This function caluculates the errors after every epoch during training
    Args:
        src_list is a list containing train data, test data, ood data
        output_list is a list containing next-token prediction probabilities based on train data, test data, ood data
        states_list is a list containing hidden markov states for train data, test data, ood data
        if contains_ood is False, the above lists do not contain ood data related data
    Returns:
        err_ratios: a list containing train/text/ood next-token prediction errors
        err_states_ratios: a list containing errors for prediction hidden states, on train/text/ood respectively
        err_probs_ratios: predicted transition matrix vs. true transition matrix, measured under L_1 loss, on train/text/ood respectively
    """
    K = len(state_sizes)
    assert np.all(
        np.array([state_sizes[k] == state_sizes[0] for k in range(K)])
    ), "Currently only support identical state sizes"
    s = state_sizes[0]
    err_ratios = torch.zeros(len(src_list))
    err_states_ratios = torch.zeros(len(src_list))
    err_probs_ratios = torch.zeros(len(src_list))
    for k, (src, output, states) in enumerate(zip(src_list, output_list, states_list)):
        sample_size, T, vocab_size = output.size(0), output.size(1), output.size(2)
        pred = output.argmax(dim=2)
        # cleaning; remove examples from counting errors if too few states are 0
        nums_zero_state = torch.sum(states == 0, dim=1)
        id_keep = (
            nums_zero_state > 3
        )  # remove some instances from dataset if too few satisfy states==0 such that ioi is impossible
        sample_size_effective = torch.sum(id_keep)
        src, states, output, pred = (
            src[id_keep],
            states[id_keep],
            output[id_keep],
            pred[id_keep],
        )
        # read states and probabilities from pred/output
        pred_states = pred // s
        pred_states = (
            pred_states * (pred_states < K)
        ).long()  # treating special symbols as having state 0
        probs = F.softmax(output, dim=-1)
        state_probs = probs[:, :, : (s * K)].view(sample_size, T, K, s).sum(axis=3)
        state_probs[:, :, 0] += (
            probs[:, :, (s * K) :].view(sample_size, T, -1).sum(axis=2)
        )  # special symbols combined with state 0
        pred_probs = torch.zeros(K, K)
        for j1 in range(K):
            for j2 in range(K):
                pred_probs[j1, j2] = torch.sum(
                    (states == j1) * state_probs[:, :, j2]
                ) / torch.sum(states == j1)

        loc1, loc2 = torch.nonzero(states == 0, as_tuple=True)
        # loc1, loc2 = loc1[loc2!=0], loc2[loc2!=0] # NOTE: ignore this for now
        total_err = torch.sum(src[loc1, loc2] != pred[loc1, loc2 - 1])
        total_zero_state = len(loc1)
        err_ratios[k] = (total_err - sample_size_effective) / (
            total_zero_state - sample_size_effective
        )
        err_states_ratios[k] = torch.mean(
            (pred_states[:, 2:-1] != states[:, 3:]).float()
        )
        err_probs_ratios[k] = torch.sum(torch.abs(pred_probs - transition_mat))

    return err_ratios, err_states_ratios, err_probs_ratios


######### making various plots ############


def plots_maker(
    model,
    config,
    src_list,
    starts=None,
    epoch=None,
    lens=None,
    save_dir=None,
):
    """
    Making various plots for a model during/after training
    """
    assert len(src_list) == 3, "Only supports includuing OOD data"
    src, src_test, src_test_ood = src_list
    src_labels = ["train", "test", "ood"]
    num_layers = config.num_layers
    d_model = config.d_model

    # plot errors at each position
    _ = plot_err_over_pos(
        model,
        [src, src_test, src_test_ood],
        config.vocab_size,
        f"err_over_pos_epoch_{epoch}",
        lens=lens,
        starts=starts,
        src_labels=["train", "test", "ood"],
        save_dir=save_dir,
    )

    # plot Gram matrix
    if config.pos not in ["rotary", "relative"]:
        wpe = F.normalize(model.pos_embed.pe.weight, dim=-1)
        wte = F.normalize(model.embed.embed.weight, dim=-1)
        basis = torch.concat([wpe, wte], dim=0).detach().numpy()
        Gram = basis @ basis.T

        fig, axs = plt.subplots(1, 1, figsize=(12 * 1, 12 * 1))
        sns.heatmap(Gram, ax=axs, vmin=-1, vmax=1, cmap="bwr")
        axs.set_title("Gram matrix: [pos, token]", weight="bold")
        plt.savefig(os.path.join(save_dir, f"Gram_matrix_epoch_{epoch}"))
        plt.close()

    # plot attention
    attn_dir = os.path.join(save_dir, "attn")
    create_folder(attn_dir)
    attn_list = {}
    QK_list = {}
    for layer in range(num_layers):
        attn_list[layer] = []
        QK_list[layer] = []
        for k, src0 in enumerate(src_list):
            attn, QK = plot_attention(
                model,
                src0[0, :],
                pos=config.pos,
                layer=layer,
                num_heads=config.num_heads,
                norm=config.norm,
                fig_name=f"_{src_labels[k]}_{layer}_epoch_{epoch}",
                savefig_dir=attn_dir,
            )
            attn_list[layer].append(attn)
            QK_list[layer].append(QK)

    if num_layers == 1:
        layer1 = 0
        W_q = model.h[layer1].mha.W_q.weight.numpy(force=True)
        W_k = model.h[layer1].mha.W_k.weight.numpy(force=True)
        W_v = model.h[layer1].mha.W_v.weight.numpy(force=True)
        W_o = model.h[layer1].mha.W_o.weight.numpy(force=True)
        W_qk = W_q.T @ W_k / np.sqrt(d_model)
        W_ov = W_o @ W_v
        fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))
        sns.heatmap(W_qk, ax=axs[0])
        axs[0].set_title("W_qk", weight="bold")
        sns.heatmap(W_ov, ax=axs[1])
        axs[1].set_title("W_ov", weight="bold")
        plt.savefig(os.path.join(save_dir, f"QK_OV_visz_epoch_{epoch}"))
        plt.close()
        return

    # plot weight matrices
    for layer1 in range(num_layers):
        for layer2 in range(layer1 + 1, num_layers):
            W_q = model.h[layer2].mha.W_q.weight.numpy(force=True)
            W_k = model.h[layer2].mha.W_k.weight.numpy(force=True)
            W_v = model.h[layer1].mha.W_v.weight.numpy(force=True)
            W_o = model.h[layer1].mha.W_o.weight.numpy(force=True)
            W_qk = W_q.T @ W_k / np.sqrt(d_model)
            W_ov = W_o @ W_v
            W_qkov = W_qk @ W_ov
            fig, axs = plt.subplots(1, 3, figsize=(6 * 3, 6 * 1))
            sns.heatmap(W_qk, ax=axs[0], square=True)
            axs[0].set_title("W_qk", weight="bold")
            sns.heatmap(W_ov, ax=axs[1], square=True)
            axs[1].set_title("W_ov", weight="bold")
            sns.heatmap(W_qkov, ax=axs[2], square=True)
            axs[2].set_title("W_qkov", weight="bold")
            plt.savefig(
                os.path.join(
                    save_dir, f"QK_OV_visz_layer_{layer1}_{layer2}_epoch_{epoch}"
                )
            )
            plt.close()

    # plot subspace matching
    num_svals_plot = 32
    U_qk, s_qk, Vt_qk = np.linalg.svd(W_qk)
    U_ov, s_ov, Vt_ov = np.linalg.svd(W_ov)
    s_match = np.zeros((2, num_svals_plot))
    for j in range(num_svals_plot):
        _, s, _ = np.linalg.svd(Vt_qk[: (j + 1), :] @ U_ov[:, : (j + 1)])
        _, s2, _ = np.linalg.svd(Vt_ov[: (j + 1), :] @ U_qk[:, : (j + 1)])
        s_match[0, j] = s[0]
        s_match[1, j] = s2[0]

    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1))
    axs[0].plot(s_qk[:num_svals_plot] / s_qk[0], "-o", label="qk", linewidth=2)
    axs[0].plot(s_ov[:num_svals_plot] / s_ov[0], "-o", label="ov", linewidth=2)
    axs[0].plot(s_match[0, :num_svals_plot], "-o", label="match")
    axs[0].legend()
    axs[0].set_title("inner match")
    axs[1].plot(s_qk[:num_svals_plot] / s_qk[0], "-o", label="qk", linewidth=2)
    axs[1].plot(s_ov[:num_svals_plot] / s_ov[0], "-o", label="ov", linewidth=2)
    axs[1].plot(s_match[1, :num_svals_plot], "-o", label="match")
    axs[1].legend()
    axs[1].set_title("outer match")
    plt.savefig(os.path.join(save_dir, f"subspace_matching_{epoch}"))
    plt.close()
