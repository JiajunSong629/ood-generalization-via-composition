import logging
import numpy as np
import os
import json
import scipy
from typing import Optional, List, Union, Tuple, Dict, Any
import torch
import time
import gc

from src.config import (
    MODEL_CLASSES,
    MODEL_META,
)

from transformers import BitsAndBytesConfig


# squelch some excessive logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
torch.set_grad_enabled(False)


class HFModel:
    def __init__(self, model_name: str, device: str = None, quantize: bool = False):
        self._model_name = model_name
        model_class = MODEL_CLASSES[self._model_name]["lm"]
        tokenizer_class = MODEL_CLASSES[self._model_name]["tokenizer"]
        self._hf_name = MODEL_CLASSES[self._model_name]["hf_name"]
        self._torch_dtype = MODEL_CLASSES[self._model_name]["torch_dtype"]
        if device is None:
            self._device = MODEL_CLASSES[self._model_name]["device"]
        else:
            self._device = device

        self._tokenizer = tokenizer_class.from_pretrained(self._hf_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

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
                device_map=self._device,
                quantization_config=quantization_config,
                torch_dtype=self._torch_dtype,
                attn_implementation="eager",
                output_attentions=True,
            )
        else:
            self._model = model_class.from_pretrained(
                self._hf_name,
                local_files_only=True,
                pad_token_id=self._tokenizer.eos_token_id,
                torch_dtype=self._torch_dtype,
                attn_implementation="eager",
                output_attentions=True,
                device_map=self._device,
            )

        self._model.eval()
        self._model_meta = MODEL_META[self._model_name]
        self._model_meta.update(
            {"device": self._device, "torch_dtype": str(self._torch_dtype)}
        )

    @property
    def model_name(self):
        return self._model_name

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
        file_path = os.path.join(
            local_dir, "ih_pth_heads_list", f"ih_pth_{self._model_name}.json"
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
        file_path = os.path.join(
            local_dir, "ih_pth_heads_list", f"ih_pth_{self._model_name}.json"
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

    @property
    def diagonal_induction_heads(self):
        local_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(
            local_dir, "diagonal_heads_list", f"ih_pth_{self._model_name}.json"
        )
        return json.load(open(file_path, "r"))["induction_heads"]

    @property
    def diagonal_previous_token_heads(self):
        local_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(
            local_dir, "diagonal_heads_list", f"ih_pth_{self._model_name}.json"
        )
        return json.load(open(file_path, "r"))["previous_token_heads"]

    def score(self, input: str, target: str) -> float:
        """Calculates log probability of target given input: log p(target|input)"""
        full_ids = self._tokenizer.encode(input + target, return_tensors="pt").to(
            self._device
        )[0]
        prompt_ids = self._tokenizer.encode(input, return_tensors="pt").to(
            self._device
        )[0]

        prompt_len = len(prompt_ids)
        target_ids = full_ids[prompt_len:]

        outputs = self._model(full_ids[None, :-1])
        logits = outputs.logits[0]

        score = 0.0
        for i, target_id in enumerate(target_ids):
            idx = prompt_len - 1 + i
            token_logits = logits[idx]
            token_logprob = torch.log_softmax(token_logits, dim=-1)[target_id]
            score += token_logprob.item()

        return score
        # prompt_tokens = self._tokenizer.encode(input, return_tensors="pt").to("cuda")
        # answer_tokens = self._tokenizer.encode(target, return_tensors="pt").to("cuda")
        # combine_tokens = self._tokenizer.encode(
        #     f"{input} {target}", return_tensors="pt"
        # ).to("cuda")

        # # print("prompt tok: ", prompt_tokens.shape, prompt_tokens)
        # # print("ans tok: ",answer_tokens.shape,  answer_tokens)
        # # print("combine tok: ", combine_tokens.shape, combine_tokens)
        # # print("decode combine_tok: ", tokenizer.decode(combine_tokens.tolist()[0]))

        # # Concatenate prompt and answer tokens # TODO, should not concat tokens together, should do A then A+B
        # # concat_tokens = torch.cat((prompt_tokens, answer_tokens), dim=1)

        # # print("concat tok: ", concat_tokens.shape, concat_tokens)
        # # print(concat_tokens.tolist()[0])
        # # print("decode concat_tok: ", tokenizer.decode(concat_tokens.tolist()[0]))

        # # Get the length of the prompt and answer
        # prompt_len = prompt_tokens.size(1)
        # # print("prompt_len: ", prompt_len)

        # # Forward pass through the model
        # outputs = self._model(input_ids=combine_tokens, return_dict=True)
        # # print("output.logits: ", outputs.logits.shape)
        # logits = outputs.logits[
        #     :, prompt_len - 1 : -1, :
        # ]  # tricky, depends on the tokenizer, sos and eos token, better to print out then trunctate
        # ### logits = outputs.logits[:, :-1, :]  # tricky, depends on the tokenizer, sos and eos token, better to print out then trunctate
        # logits = logits.view(-1, logits.size(-1))  ## format as [bs, num_cls]
        # # print("logits.shape: ", logits.shape)

        # # Prepare the target tokens (shift answer tokens by one position)
        # target_tokens = combine_tokens[
        #     :, prompt_len:
        # ].squeeze()  # Remove batch dimension if necessary
        # ### target_tokens = combine_tokens[:, 1:].squeeze() # Remove batch dimension if necessary
        # target_tokens = target_tokens.view(-1)  ## format as [bs]
        # # print("target_tokens: ", target_tokens.shape, target_tokens)

        # # pdb.set_trace()

        # # assert False

        # # Compute the loss
        # import torch.nn as nn

        # loss_fct = nn.CrossEntropyLoss(reduction="sum")
        # loss = loss_fct(logits, target_tokens)

        # return loss.item()

    def cond_log_prob(self, inputs: List[str], targets: List[List[str]]) -> List[float]:
        """Calculates log probability of targets given input: log p(targets|input)"""
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        batch_size: int = 1,
    ) -> Union[str, List[str], List[List[str]]]:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        generated_texts = []

        # Process inputs in batches
        for i in range(0, len(input_list), batch_size):
            end = min(i + batch_size, len(input_list))
            batch_inputs = input_list[i:end]

            with torch.no_grad():  # Prevent gradient computation
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

                # Pythia-7b's generate seems to be different than other models
                if isinstance(outputs, dict):
                    outputs = outputs["sequences"]

                # Process outputs and clear memory
                if num_outputs > 1:
                    outputs = outputs.reshape(len(batch_inputs), num_outputs, -1)
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
                    batch_generated_texts = []
                    for seq in outputs:
                        text = self._tokenizer.decode(
                            seq[input_ids["input_ids"].shape[1] :],
                            skip_special_tokens=True,
                        )
                        batch_generated_texts.append(text)

                # Clear GPU memory after processing each batch
                del input_ids
                del outputs
                torch.cuda.empty_cache()
                gc.collect()

            generated_texts.extend(batch_generated_texts)

        if isinstance(inputs, str):
            return generated_texts[0]
        return generated_texts
