import copy
import torch
from typing import List, Tuple, Optional
import random
import src.models.huggingface_models as hf_models


class ShuffleModel(hf_models.HFModel):
    def __init__(
        self,
        base_model: hf_models.HFModel,
        layer_head_pairs: List[Tuple[int, int]],
        seed: Optional[int] = 42,
    ):
        # Store reference to base model instead of creating new one
        self._model = base_model._model
        self._device = base_model._device
        self._model_meta = base_model.model_meta
        self._tokenizer = base_model._tokenizer

        self._layer_head_pairs = layer_head_pairs
        self._shuffled = False
        self._shuffled_type = None
        self._weights = {}

        self._seed = seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    @property
    def model_meta(self):
        meta = copy.deepcopy(self._model_meta)
        meta["shuffle_meta"] = {
            "shuffled": self._shuffled,
            "shuffled_type": self._shuffled_type,
            "shuffled_component": self._shuffled_component,
            "shuffled_layer_head_pairs": {
                f"L{ilayer}_H{ihead}": f"L{ilayer_perm}_H{ihead_perm}"
                for (ilayer, ihead), (
                    ilayer_perm,
                    ihead_perm,
                ) in self._shuffled_layer_head_pairs.items()
            },
            "seed": self._seed,
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

    def _swap(self):
        for ilayer, ihead in self._layer_head_pairs:
            ilayer_perm, ihead_perm = self._shuffled_layer_head_pairs[(ilayer, ihead)]
            for name in list(self._shuffled_component):
                key = f"L_{ilayer_perm}_H_{ihead_perm}_{name}"
                w = self._get_qkov_weight(ilayer, ihead, name)
                try:
                    w.copy_(self._weights[key])
                except Exception as e:
                    print(key)
                    print(ilayer, ihead, ilayer_perm, ihead_perm)
                    print(w.shape)
                    raise e

        self._shuffled = True

    def shuffle_inside(self, component: str):
        assert not self._shuffled, "Already shuffled model cannot be shuffled again!"

        self._shuffled_component = component
        self._shuffled_type = "inside"

        K = len(self._layer_head_pairs)
        perm = torch.randperm(K)
        self._shuffled_layer_head_pairs = {
            tuple(lh1): tuple(lh2)
            for lh1, lh2 in zip(
                self._layer_head_pairs,
                [self._layer_head_pairs[perm[j]] for j in range(K)],
            )
        }

        self._save_weights(self._layer_head_pairs)
        self._swap()

    def shuffle_outside(self, component: str):
        assert not self._shuffled, "Already shuffled model cannot be shuffled again!"

        num_layer = self.model_meta["num_layers"]
        num_head = self.model_meta["num_heads"]
        self._shuffled_component = component
        self._shuffled_type = "outside"

        self._shuffled_layer_head_pairs = {
            tuple(lh1): (
                random.randint(0, num_layer - 1),
                random.randint(0, num_head - 1),
            )
            for lh1 in self._layer_head_pairs
        }
        self._save_weights(self._layer_head_pairs)
        self._save_weights(self._shuffled_layer_head_pairs.values())
        self._swap()

    def revert(self):
        assert self._shuffled, "Not shuffled model cannot be reverted!"

        for ilayer, ihead in self._shuffled_layer_head_pairs:
            for name in list(self._shuffled_component):
                key = f"L_{ilayer}_H_{ihead}_{name}"
                w = self._get_qkov_weight(ilayer, ihead, name)
                w.copy_(self._weights[key])

        self._shuffled = False
        self._shuffled_component = None
        self._shuffled_type = None
        self._shuffled_layer_head_pairs = {}

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
