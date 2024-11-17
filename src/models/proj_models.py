import os
import copy
import torch
from typing import List, Tuple, Optional
import random
import numpy as np

from src.models.huggingface_models import HFModel
from src.models.utils import custom_svd, calc_rotary_R_mat


class ProjectModel(HFModel):
    def __init__(
        self,
        base_model: HFModel,
        layer_head_pairs: List[Tuple[int, int]],
        projected_layer_head_pairs: List[Tuple[int, int]],
    ):
        """
        layer_head_pairs: the layer and head pairs to calculate the projection matrix
        projected_layer_head_pairs: the layer and head pairs to be projected
        """
        self._model = base_model._model
        self._device = base_model._device
        self._model_meta = base_model.model_meta
        self._model_name = base_model.model_name
        self._tokenizer = base_model._tokenizer

        self._project_layer_head_pairs = layer_head_pairs
        self._projected_layer_head_pairs = projected_layer_head_pairs
        self._projected = False
        self._project_component = None
        self._rank = 0

        # Create directory for weight storage if it doesn't exist
        self._weights_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "projection_weights_cache",
            self._model_name,
        )
        os.makedirs(self._weights_dir, exist_ok=True)

        self._save_weights(self._project_layer_head_pairs)
        self._save_weights(self._projected_layer_head_pairs)

    @property
    def model_meta(self):
        meta = copy.deepcopy(self._model_meta)
        meta["project_meta"] = {
            "project_component": self._project_component,
            "project_layer_head_pairs": self._project_layer_head_pairs,
            "projected_layer_head_pairs": self._projected_layer_head_pairs,
            "rank": self._rank,
        }
        return meta

    def _get_weight_path(self, component_name: str) -> str:
        return os.path.join(
            self._weights_dir, f"{self._model_name}_{component_name}.pt"
        )

    def _save_weights(self, layer_head_pairs: List[Tuple[int, int]]):
        for ilayer, ihead in layer_head_pairs:
            for name in ["q", "k", "v", "o"]:
                component_name = f"L_{ilayer}_H_{ihead}_{name}"
                weight_path = self._get_weight_path(component_name)

                # Only save if the file doesn't exist
                if not os.path.exists(weight_path):
                    weight = self._get_qkov_weight(ilayer, ihead, name).data.cpu()
                    torch.save(weight, weight_path)

    def _load_cached_weight(self, component_name: str) -> torch.Tensor:
        return torch.load(self._get_weight_path(component_name)).to(
            device=self._device, dtype=torch.bfloat16
        )

    def _get_Vt_common(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        pth = f"{cur_dir}/projection_artifacts/{self.model_meta['model_name']}_Vt_common.npy"
        os.makedirs(f"{cur_dir}/projection_artifacts", exist_ok=True)

        if os.path.exists(pth):
            return np.load(pth)

        use_R = not self.model_meta["model_name"].startswith("gpt")
        d_model = self.model_meta["hidden_size"]

        K = len(self._project_layer_head_pairs)
        W_qk_all = np.zeros((K, d_model, d_model))
        for i, (layer, head) in enumerate(self._project_layer_head_pairs):
            Wq = self._load_cached_weight(f"L_{layer}_H_{head}_q")
            Wk = self._load_cached_weight(f"L_{layer}_H_{head}_k")

            if use_R:
                R = calc_rotary_R_mat(
                    d_head=self.model_meta["head_dim"],
                    max_seq_len=100,
                    max_rel_dist=100,
                )[-1].to(device=self._device, dtype=torch.bfloat16)
                W_qk = Wq @ R @ Wk.T
            else:
                W_qk = Wq @ Wk.T

            W_qk_all[i] = W_qk.float().numpy(force=True)

        U, S, Vt_common = custom_svd(W_qk_all.reshape(-1, d_model))

        np.save(pth, Vt_common)

        return Vt_common

    def _get_project_matrix(self, rank: int, project_out: bool) -> np.ndarray:
        Vt = self._get_Vt_common()
        if rank == 0:
            return np.eye(Vt.shape[1])

        V = Vt[:rank, :].T
        P = V @ V.T
        P = np.eye(P.shape[0]) - P if project_out else P

        return P

    def _project(self, project_matrix: np.ndarray, component: str):
        project_matrix = torch.tensor(
            project_matrix, dtype=torch.bfloat16, device=self._device
        )

        for ilayer, ihead in self._projected_layer_head_pairs:
            if component == "qk":
                key = f"L_{ilayer}_H_{ihead}_k"
                w = self._get_qkov_weight(ilayer, ihead, "k")
                w.copy_(project_matrix @ self._load_cached_weight(key))

            elif component == "ov":
                key = f"L_{ilayer}_H_{ihead}_o"
                w = self._get_qkov_weight(ilayer, ihead, "o")
                w.copy_(project_matrix @ self._load_cached_weight(key))

    def project(self, component: str, rank: int, project_out: bool):
        assert not self._projected, "Already projected model cannot be projected again!"

        self._rank = rank
        project_matrix = self._get_project_matrix(rank, project_out)
        self._project(project_matrix, component)
        self._project_component = component
        self._projected = True

    def revert(self):
        assert self._projected, "Not projected model cannot be reverted!"

        name = "k" if self._project_component == "qk" else "o"
        for ilayer, ihead in self._projected_layer_head_pairs:
            key = f"L_{ilayer}_H_{ihead}_{name}"
            w = self._get_qkov_weight(ilayer, ihead, name)
            w.copy_(self._load_cached_weight(key))

        self._projected = False
        self._project_component = None

    def _get_qkov_weight(self, ilayer, ihead, component):
        d_head = self.model_meta["head_dim"]
        d_model = self.model_meta["hidden_size"]

        if self._model_meta["model_name"] in [
            "gpt2",
            "gpt2-xl",
        ]:
            attn = self._model.transformer.h[ilayer].attn.c_attn

            proj = self._model.transformer.h[ilayer].attn.c_proj
            return {
                "q": attn.weight[:, ihead * d_head : ihead * d_head + d_head],
                "k": attn.weight[
                    :, d_model + ihead * d_head : d_model + ihead * d_head + d_head
                ],
                "v": attn.weight[
                    :,
                    d_model * 2
                    + ihead * d_head : d_model * 2
                    + ihead * d_head
                    + d_head,
                ],
                "o": proj.weight[ihead * d_head : ihead * d_head + d_head, :].T,
            }[component].data

        if self._model_meta["model_name"] in [
            "llama2-7b",
            "gemma-7b",
            "mistral-7b",
            "olmo-7b",
            "llama3-8b",
            "gemma2-9b",
        ]:
            if self._model_meta["model_name"] in [
                "mistral-7b",
                "llama3-8b",
                "gemma2-9b",
            ]:
                ikvhead = (
                    ihead
                    * self._model_meta["num_key_value_heads"]
                    // self._model_meta["num_heads"]
                )
            else:
                ikvhead = ihead

            attn = self._model.model.layers[ilayer].self_attn

            return {
                "q": attn.q_proj.weight.T[:, ihead * d_head : ihead * d_head + d_head],
                "k": attn.k_proj.weight.T[
                    :, ikvhead * d_head : ikvhead * d_head + d_head
                ],
                "v": attn.v_proj.weight.T[
                    :, ikvhead * d_head : ikvhead * d_head + d_head
                ],
                "o": attn.o_proj.weight.T[
                    ihead * d_head : ihead * d_head + d_head, :
                ].T,
            }[component].data

        elif self._model_meta["model_name"] in ["falcon-7b"]:
            attn = self._model.transformer.h[ilayer].self_attention
            splited = attn.query_key_value.weight.T.view(
                d_model, 2 + self._model_meta["num_heads"], d_head
            )
            q, k, v = splited[:, :-2], splited[:, -2], splited[:, -1]
            return {
                "q": q[:, ihead],
                "k": k,
                "v": v,
                "o": attn.dense.weight.T[ihead * d_head : ihead * d_head + d_head, :].T,
            }[component].data
        elif self._model_meta["model_name"] in ["pythia-7b"]:
            attn = self._model.gpt_neox.layers[ilayer].attention
            splited = attn.query_key_value.weight.T.view(
                d_model, self._model_meta["num_heads"], 3 * d_head
            )
            q, k, v = (
                splited[:, :, :d_head],
                splited[:, :, d_head : 2 * d_head],
                splited[:, :, 2 * d_head :],
            )
            return {
                "q": q[:, ihead],
                "k": k[:, ihead],
                "v": v[:, ihead],
                "o": attn.dense.weight.T[ihead * d_head : ihead * d_head + d_head, :].T,
            }[component].data
