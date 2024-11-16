import logging
import numpy as np
import os
import json
import scipy
from typing import Optional, List, Union, Tuple, Dict, Any
import torch
import time

from src.config import (
    MODEL_CLASSES,
    MODEL_META,
)

from transformers import BitsAndBytesConfig


# squelch some excessive logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


torch.set_grad_enabled(False)
torch.manual_seed(42)  # CPU seed
torch.cuda.manual_seed_all(42)  # GPU seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _recursive_to_float(data):
    """Recursively converts nested lists of np.array/torch.tensor to lists of floats.

    Args:
        data: Can be a number, np.array, torch.tensor, or nested lists of these types

    Returns:
        The same structure with all numbers converted to floats
    """
    if isinstance(data, (np.ndarray, torch.Tensor)):
        # Convert tensor/array to list of floats
        return (
            data.detach().cpu().float().numpy().tolist()
            if torch.is_tensor(data)
            else data.tolist()
        )
    elif isinstance(data, (list, tuple)):
        # Recursively convert each element
        return [_recursive_to_float(item) for item in data]
    elif isinstance(data, (int, float, np.number)):
        # Convert numbers to float
        return float(data)
    else:
        raise TypeError(f"Unsupported type: {type(data)}")


class HFModel:
    def __init__(self, model_name: str, device: str, quantize: bool = False):
        self._model_name = model_name
        model_class = MODEL_CLASSES[self._model_name]["lm"]
        tokenizer_class = MODEL_CLASSES[self._model_name]["tokenizer"]
        self._hf_name = MODEL_CLASSES[self._model_name]["hf_name"]

        self._device = device
        self._tokenizer = tokenizer_class.from_pretrained(self._hf_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        if device == "cuda":
            if quantize:
                # 4-bit quantization configuration
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Match your original bf16
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

                self._model = model_class.from_pretrained(
                    self._hf_name,
                    local_files_only=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    device_map="auto",  # Handles device placement automatically
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    output_attentions=True,
                )
            else:
                self._model = model_class.from_pretrained(
                    self._hf_name,
                    local_files_only=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    torch_dtype=torch.bfloat16,
                    output_attentions=True,
                ).to(device)
        else:
            self._model = model_class.from_pretrained(
                self._hf_name,
                pad_token_id=self._tokenizer.eos_token_id,
                output_attentions=True,
            )

        self._model.eval()
        self._device = device
        self._model_meta = MODEL_META[self._model_name]
        self._model_meta.update({"device": self._device})

    @property
    def model_name(self):
        return self._model_name

    # @property
    # def attention_layers(self):
    #     """Get list of attention layer modules for the model"""
    #     if self._model_name not in ATTENTION_LAYER_PATHS:
    #         raise ValueError(f"Model {self._model_name} attention path not defined")

    #     # Get the transformer blocks/layers
    #     transformer_blocks = ATTENTION_LAYER_PATHS[self._model_name](self._model)
    #     attn_name = ATTENTION_MODULE_NAMES[self._model_name]

    #     # Get attention modules using recursive attribute lookup
    #     attention_layers = []
    #     for block in transformer_blocks:
    #         curr_module = block
    #         for attr in attn_name.split("."):
    #             curr_module = getattr(curr_module, attr)
    #         attention_layers.append(curr_module)

    #     return attention_layers

    @property
    def model_meta(self) -> Dict[str, Any]:
        return self._model_meta

    def _get_heads_list(self):
        from src.tasks.copying.task import make_input_ids

        seg_len, rep = 25, 3
        EPSILON = 1e-6

        input_ids = make_input_ids(
            num_samples=50,
            seg_len=seg_len,
            rep=rep,
            vocab_size=self.model_meta["vocab_size"],
            prepend_bos="bos_token_id" in self.model_meta,
            bos=self.model_meta.get("bos_token_id", None),
        )
        input_ids = torch.Tensor(input_ids).long().to(self._device)

        for i in range(input_ids.size(0)):
            cur_batch_input_ids = input_ids[i : i + 1]
            cur_attentions = self._model(cur_batch_input_ids).attentions
            cur_attentions = np.array(
                [a.float().numpy(force=True).mean(0) for a in cur_attentions]
            )[np.newaxis, :]
            if i == 0:
                attentions = cur_attentions
            else:
                attentions = np.vstack([attentions, cur_attentions])

        sample_size, num_layer, num_head, T, _ = attentions.shape
        scores = np.zeros((num_layer, num_head))
        heads_list = {}
        for offset, head_type in [
            (-(seg_len - 1), "induction_heads"),
            (-1, "previous_token_heads"),
        ]:
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

            idx_sort = np.argsort(scores, axis=None)[::-1].tolist()
            head_list = [
                [idx_sort[j] // num_head, idx_sort[j] % num_head]
                for j in range(len(idx_sort))
            ]

            heads_list[head_type] = head_list

        return heads_list

    @property
    def previous_token_heads(self):
        # if exists a json file, ends with the name of the model, load it
        local_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(local_dir, "heads_list"), exist_ok=True)
        file_path = os.path.join(
            local_dir, "heads_list", f"ih_pth_{self._model_name}.json"
        )

        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)["previous_token_heads"]
            except Exception as e:
                print(
                    f"Error loading previous token heads for {self._model_name}, {e}. continue generating..."
                )

        heads_list = self._get_heads_list()
        with open(file_path, "w") as f:
            json.dump(heads_list, f)

        return heads_list["previous_token_heads"]

    @property
    def induction_heads(self) -> List[Tuple[int, int]]:
        # if exists a json file, ends with the name of the model, load it
        local_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(local_dir, "heads_list"), exist_ok=True)
        file_path = os.path.join(
            local_dir, "heads_list", f"ih_pth_{self._model_name}.json"
        )

        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    return json.load(f)["induction_heads"]
            except Exception as e:
                print(
                    f"Error loading induction heads for {self._model_name}, {e}. continue generating..."
                )

        heads_list = self._get_heads_list()
        with open(file_path, "w") as f:
            json.dump(heads_list, f)

        return heads_list["induction_heads"]

    def score(self, input: str, target: str) -> float:
        """Calculates log probability of target given input: log p(target|input)

        Args:
            input: Input context string
            target: Target string to score

        Returns:
            float: Log probability score of target given input
        """
        # Encode full sequence (input + target)
        full_ids = self._tokenizer.encode(input + target, return_tensors="pt").to(
            self._device
        )
        target_ids = self._tokenizer.encode(target, add_special_tokens=False)
        target_len = len(target_ids)

        # Get model outputs
        logits = self._model(full_ids).logits
        logprobs = torch.nn.functional.log_softmax(logits[0], dim=-1)

        score = 0.0
        for i, target_id in enumerate(target_ids):
            score += logprobs[-target_len - 1 + i, target_id].item()

        return score

    def cond_log_prob(self, inputs: List[str], targets: List[List[str]]) -> List[float]:
        """Calculates log probability of targets given input: log p(targets|input)"""
        log_probs = []

        for input, choices in zip(inputs, targets):
            log_probs.append([self.score(input, choice) for choice in choices])

        return log_probs

    def generate_text(
        self,
        inputs: Union[str, List[str]],
        max_new_tokens: int,
        num_beams: int = 1,
        num_outputs: int = 1,
        batch_size: int = 8,
    ) -> Union[str, List[str], List[List[str]]]:
        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        generated_texts = []

        # Process inputs in batches
        for i in range(0, len(input_list), batch_size):
            end = min(i + batch_size, len(input_list))
            batch_inputs = input_list[i:end]

            input_ids = self._tokenizer(
                batch_inputs, return_tensors="pt", padding=True
            ).to(self._device)

            # Generate using the model's generate method
            outputs = self._model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_outputs,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

            # Reshape outputs if multiple sequences per input
            if num_outputs > 1:
                outputs = outputs.reshape(len(batch_inputs), num_outputs, -1)
                # Decode multiple sequences for each input in batch
                batch_generated_texts = [
                    [
                        self._tokenizer.decode(
                            seq[input_ids["input_ids"].shape[1] :],
                            skip_special_tokens=True,
                        )
                        for seq in input_outputs
                    ]
                    for input_outputs in outputs
                ]
            else:
                # Decode single sequence per input
                batch_generated_texts = [
                    self._tokenizer.decode(
                        seq[input_ids["input_ids"].shape[1] :], skip_special_tokens=True
                    )
                    for seq in outputs
                ]

            generated_texts.extend(batch_generated_texts)

        if isinstance(inputs, str):
            return generated_texts[0]
        return generated_texts


class HFModelWithHeadMask(HFModel):
    def __init__(self, base_model: HFModel, layer_head_pairs: List[Tuple[int, int]]):
        """Creates a masked version of an existing model.

        Args:
            base_model: Existing HFModel instance to mask
            head_mask: Optional tensor of shape (n_layers, n_heads) with values in [0,1]
        """
        # Store reference to base model instead of creating new one
        self._model = base_model._model
        self._device = base_model._device
        self._show_progress = base_model._show_progress
        self._model_meta = base_model.model_meta

        # Save original forward function
        self._original_forward = self._model._model.forward

        # Initialize head mask
        head_mask = torch.ones(
            self.model_meta["num_layers"],
            self.model_meta["num_heads"],
        ).to(self._device)

        if layer_head_pairs is None:
            self._head_mask = head_mask
        else:
            for l, h in layer_head_pairs:
                head_mask[l, h] = 0
            self._head_mask = head_mask

        # Apply mask
        self.apply_mask()

    def apply_mask(self):
        """Apply the head mask to model"""

        def masked_forward(*args, **kwargs):
            kwargs["head_mask"] = self._head_mask
            return self._original_forward(*args, **kwargs)

        self._model.forward = masked_forward

    def remove_mask(self):
        """Restore original unmasked model"""
        self._model.forward = self._original_forward

    def set_head_mask(self, head_mask):
        """Update head mask and reapply"""
        self._head_mask = torch.tensor(head_mask).to(self._device)
        self.apply_mask()
