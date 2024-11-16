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
    ):
        # Store reference to base model instead of creating new one
        self._model = base_model._model
        self._device = base_model._device
        self._model_meta = base_model.model_meta
        self._model_name = base_model.model_name
        self._tokenizer = base_model._tokenizer

        self._project_layer_head_pairs = layer_head_pairs
        self._projected = False
        self._project_component = None
        self._project_vt_common = 10

        self._save_weights(self.induction_heads[: self._project_vt_common])

    @property
    def model_meta(self):
        meta = self._model_meta
        meta["project_meta"] = {
            "project_component": self._project_component,
            "project_layer_head_pairs": self._project_layer_head_pairs,
        }
        return meta

    def _save_weights(self, layer_head_pairs: List[Tuple[int, int]]):
        # if _weights not in self, initialize it
        if not hasattr(self, "_weights"):
            self._weights = {}

        for ilayer, ihead in layer_head_pairs:
            for name in ["q", "k", "v", "o"]:
                component_name = f"L_{ilayer}_H_{ihead}_{name}"
                self._weights[component_name] = copy.deepcopy(
                    self._get_qkov_weight(ilayer, ihead, name).data
                )

    def _get_Vt_common(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        pth = f"{cur_dir}/projection_artifacts/{self.model_meta['model_name']}_Vt_common.npy"
        os.makedirs(f"{cur_dir}/projection_artifacts", exist_ok=True)

        use_R = not self.model_meta["model_name"].startswith("gpt")
        d_model = self.model_meta["hidden_size"]

        K = self._project_vt_common
        W_qk_all = np.zeros((K, d_model, d_model))
        for i, (layer, head) in enumerate(self.induction_heads[:K]):
            Wq = self._weights[f"L_{layer}_H_{head}_q"]
            Wk = self._weights[f"L_{layer}_H_{head}_k"]

            if use_R:
                R = calc_rotary_R_mat(
                    d_head=self.model_meta["head_dim"],
                    max_seq_len=100,
                    max_rel_dist=100,
                )[-1]
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

        for ilayer, ihead in self._project_layer_head_pairs:
            if component == "qk":
                key = f"L_{ilayer}_H_{ihead}_k"
                w = self._get_qkov_weight(ilayer, ihead, "k")
                w.copy_(project_matrix @ self._weights[key])

            elif component == "ov":
                key = f"L_{ilayer}_H_{ihead}_o"
                w = self._get_qkov_weight(ilayer, ihead, "o")
                w.copy_(project_matrix @ self._weights[key])

    def project(self, component: str, rank: int, project_out: bool):
        assert not self._projected, "Already projected model cannot be projected again!"

        self._save_weights(self._project_layer_head_pairs)

        project_matrix = self._get_project_matrix(rank, project_out)
        self._project(project_matrix, component)

        self._project_component = component
        self._projected = True

    def revert(self):
        assert self._projected, "Not projected model cannot be reverted!"

        name = "k" if self._project_component == "qk" else "o"
        for ilayer, ihead in self._project_layer_head_pairs:
            key = f"L_{ilayer}_H_{ihead}_{name}"
            w = self._get_qkov_weight(ilayer, ihead, name)
            w.copy_(self._weights[key])

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


if __name__ == "__main__":
    from src.models.huggingface_models import HFModel
    from src.tasks.copying.task import CopyingTask

    model = HFModel("gpt2", "cuda")
    task = CopyingTask(seg_len=25, rep=3, ignore_segment=1, ignore_burning=10)

    result = task.evaluate_model(model, num_samples=100, task_random_seed=42)
    print(1 - np.mean(result.errs))

    proj_model = ProjectModel(
        model,
        layer_head_pairs=model.induction_heads[:36],
    )

    for rank in range(0, 100, 10):
        proj_model.project("qk", rank, True)
        result = task.evaluate_model(proj_model, num_samples=10, task_random_seed=42)
        print("======= Rank: ", rank)
        print(1 - np.mean(result.errs))
        proj_model.revert()
