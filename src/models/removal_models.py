import torch
import importlib
import copy
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Type
from src.models.huggingface_models import HFModel


class HeadRemovalModel:
    """A wrapper that can mask specific attention heads in transformer models."""

    def __init__(self, base_model: "HFModel"):
        self._base_model = base_model
        self._model = base_model._model
        self._device = base_model.model_meta["device"]
        self._model_meta = base_model.model_meta
        self._induction_heads = base_model.induction_heads

        # Set attention implementation to eager
        for module in self._model.modules():
            if hasattr(module, "attn_implementation"):
                module.attn_implementation = "eager"
                print(f"Setting {module.__class__.__name__} to eager implementation")

        self._is_masked = False
        self._original_modules = {}
        self._masked_method = None
        self.head_mask = None
        self.head_indices = None
        self._masked_seed = None

    @property
    def model_meta(self):
        meta = copy.deepcopy(self._model_meta)
        if self._is_masked and len(self.head_indices) > 0:
            meta["mask_meta"] = {
                "is_masked": self._is_masked,
                "masked_seed": self._masked_seed,
                "masked_method": self._masked_method,
                "masked_n_heads": len(self.head_indices),
                "masked_heads": self.head_indices,
            }
        return meta

    def mask(self, layer_head_pairs: List[Tuple[int, int]]):
        """Apply masking to specified heads"""
        assert not self._is_masked, "Model is already masked. Call revert() first."
        self.head_indices = layer_head_pairs

        # Create head mask tensor
        num_layers = self._model_meta["num_layers"]
        num_heads = self._model_meta["num_heads"]
        self.head_mask = torch.ones(num_layers, num_heads, device=self._device)

        # Mask out specified heads
        for layer_idx, head_idx in layer_head_pairs:
            assert 0 <= layer_idx < num_layers, f"Invalid layer index: {layer_idx}"
            assert 0 <= head_idx < num_heads, f"Invalid head index: {head_idx}"
            self.head_mask[layer_idx, head_idx] = 0

        # Replace attention modules with masked versions
        self._replace_attention_modules()
        self._is_masked = True

    def mask_top_ih(self, n_heads: int):
        self.mask(layer_head_pairs=self._induction_heads[:n_heads])
        self._masked_method = "top_ih"

    def mask_random(self, n_heads: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            rnd = random.Random()
            rnd.seed(seed)

        all_pairs = [
            (x, y)
            for x in range(self.model_meta["num_layers"])
            for y in range(self.model_meta["num_heads"])
        ]
        masked_heads = rnd.sample(all_pairs, min(n_heads, len(all_pairs)))
        # masked_heads = [
        #     self._induction_heads[i]
        #     for i in np.random.choice(
        #         len(self._induction_heads),
        #         min(len(self._induction_heads), n_heads),
        #         replace=False,
        #     )
        # ]
        self.mask(layer_head_pairs=masked_heads)
        self._masked_method = "random"
        self._masked_seed = seed

    def _replace_attention_modules(self):
        """Replace attention modules with masked versions"""
        found_modules = 0
        for name, module in self._model.named_modules():
            if "Attention" in module.__class__.__name__:
                layer_idx = self._extract_layer_idx(name)
                if layer_idx is not None and (1 - self.head_mask[layer_idx]).sum() > 0:
                    found_modules += 1
                    # Get the corresponding Ada class
                    original_class = module.__class__
                    ada_class_name = f"Ada{original_class.__name__}"

                    try:
                        ada_module = importlib.import_module(
                            "src.models.removal_attn_modules"
                        )
                        ada_class = getattr(ada_module, ada_class_name)
                    except (ImportError, AttributeError) as e:
                        raise ImportError(
                            f"Could not find {ada_class_name} in removal_attn_modules. "
                            f"Please implement it for {original_class.__name__}"
                        )

                    # Replace the module
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = self._model.get_submodule(parent_name)

                    # Store original module
                    self._original_modules[name] = module

                    # Create and set new module
                    new_module = ada_class(module, layer_idx, self.head_mask)
                    setattr(parent, child_name, new_module)

    def _extract_layer_idx(self, module_name: str) -> Optional[int]:
        """Extract layer index from module name"""
        if "layers." in module_name:
            parts = module_name.split("layers.")[1].split(".")
            return int(parts[0])
        if "h." in module_name:
            return int(module_name.split("h.")[1].split(".")[0])
        return None

    def revert(self):
        """Revert all attention modules to their original versions"""
        assert self._is_masked, "Model is not masked. Cannot revert."

        for name, original_module in self._original_modules.items():
            parent_name, child_name = name.rsplit(".", 1)
            parent = self._model.get_submodule(parent_name)
            setattr(parent, child_name, original_module)

        self._original_modules.clear()
        self.head_mask = None
        self.head_indices = None
        self._is_masked = False
        self._masked_method = None
        self._masked_seed = None

    def __getattr__(self, name):
        return getattr(self._base_model, name)

    @property
    def is_masked(self) -> bool:
        return self._is_masked
