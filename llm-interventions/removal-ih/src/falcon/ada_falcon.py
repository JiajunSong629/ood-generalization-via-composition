import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, LayerNorm
from transformers.models.falcon.configuration_falcon import FalconConfig
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconFlashAttention2,
    apply_rotary_pos_emb,
    FalconMLP,
    FalconDecoderLayer,
    FalconModel,
    FalconForCausalLM,
)
from transformers.cache_utils import Cache, DynamicCache
from pdb import set_trace as pds
# falcon_layer_head_id_7b = [[5, 41], [5, 2], [5, 52], [5, 56], [5, 18], [5, 62], [5, 13], [5, 10], [5, 65], [5, 69], [5, 1], [5, 27], [7, 68], [7, 48], [5, 43], [7, 16], [19, 57], [19, 26], [5, 5], [5, 70], [19, 48], [5, 14], [7, 35], [5, 33], [5, 8], [5, 46], [5, 59], [5, 66], [5, 19], [5, 7], [7, 5], [19, 49], [7, 8], [19, 45], [5, 21], [19, 3], [5, 39], [5, 64], [5, 24], [7, 26], [5, 3], [19, 34], [7, 32], [5, 63], [19, 16], [5, 44], [19, 40], [7, 3], [7, 49], [7, 10], [5, 12], [19, 67], [9, 68], [7, 58], [19, 69], [19, 9], [5, 49], [5, 55], [7, 30], [19, 1], [24, 9], [5, 16], [7, 39], [19, 38], [7, 6], [23, 32], [19, 46], [7, 70], [19, 59], [19, 62], [19, 43], [19, 6], [7, 24], [26, 14], [7, 42], [4, 37], [18, 37], [5, 40], [5, 45], [5, 54], [9, 66], [19, 17], [19, 25], [19, 37], [5, 50], [7, 34], [27, 48], [7, 66], [11, 38], [19, 64], [25, 11], [5, 15], [19, 39], [19, 20], [28, 2], [5, 38], [16, 70], [19, 4], [7, 67], [9, 2], [7, 7], [17, 8], [19, 14], [28, 27], [5, 9], [9, 12], [19, 19], [5, 31], [19, 15], [19, 53], [5, 42], [7, 65], [19, 60], [7, 22], [18, 2], [19, 28], [19, 11], [26, 45], [19, 44], [19, 10], [19, 68], [22, 51], [19, 36], [28, 5], [27, 0], [27, 45], [5, 11], [4, 69], [18, 67], [4, 53], [23, 26], [24, 25], [27, 61], [11, 65], [17, 14], [27, 25], [27, 62], [19, 42], [19, 2], [19, 50], [24, 31], [19, 8], [17, 32], [27, 51], [27, 18], [22, 17], [28, 4], [24, 47], [9, 37], [28, 39], [24, 28], [23, 16], [27, 38], [14, 19], [27, 9], [18, 57], [19, 29], [5, 61], [27, 4], [11, 28], [24, 32], [19, 61], [27, 12], [22, 8], [18, 11], [17, 13], [7, 17], [27, 32], [24, 54], [27, 59], [19, 55], [16, 36], [27, 70], [26, 67], [7, 0], [20, 53], [17, 62], [12, 40], [19, 5], [25, 50], [25, 27], [20, 21], [26, 46], [25, 47], [24, 61], [19, 32], [12, 1], [27, 20], [22, 54], [5, 68], [5, 67], [26, 32], [24, 53], [7, 25], [18, 23], [27, 3], [7, 55], [22, 0], [18, 35], [24, 18]]

falcon_layer_head_id_7b = [[5, 41], [5, 2], [5, 18], [5, 52], [5, 65], [7, 48], [7, 68], [19, 26], [5, 10], [5, 1], [5, 69], [5, 62], [5, 13], [5, 43], [7, 32], [7, 5], [19, 57], [5, 14], [5, 59], [19, 48], [5, 39], [5, 46], [7, 35], [5, 21], [7, 16], [5, 66], [5, 49], [5, 63], [5, 33], [19, 34], [5, 7], [5, 70], [19, 3], [7, 8], [19, 16], [7, 58], [5, 5], [7, 42], [7, 26], [7, 49], [19, 40], [9, 68], [5, 56], [5, 64], [9, 66], [19, 14], [7, 3], [19, 45], [5, 24], [19, 37], [19, 67], [5, 8], [5, 9], [7, 30], [19, 39], [19, 59], [27, 48], [5, 3], [19, 9], [7, 66], [7, 17], [7, 70], [11, 65], [18, 2], [19, 25], [19, 1], [5, 44], [7, 22], [19, 17], [19, 49], [5, 45], [19, 44], [7, 34], [11, 38], [19, 38], [24, 9], [19, 11], [19, 15], [19, 69], [19, 20], [9, 12], [19, 19], [19, 55], [27, 0], [27, 3], [19, 62], [19, 43], [19, 50], [7, 39], [7, 24], [19, 28], [19, 4], [19, 5], [19, 64], [19, 53], [17, 8], [7, 6], [27, 62], [19, 61], [25, 11], [19, 36], [19, 42], [19, 6], [22, 44], [20, 21], [19, 68], [27, 55], [23, 32], [27, 12], [5, 16], [28, 27], [7, 0], [19, 27], [27, 4], [19, 8], [5, 12], [27, 51], [16, 41], [27, 19], [19, 2], [27, 25], [27, 32], [25, 14], [27, 1], [24, 25], [27, 67], [27, 44], [27, 18], [5, 27], [27, 66], [26, 45], [27, 7], [7, 55], [19, 10], [20, 53], [27, 20], [27, 70], [10, 4], [5, 40], [17, 13], [5, 55], [4, 37], [5, 42], [27, 61], [27, 59], [27, 60], [4, 53], [27, 9], [27, 64], [24, 61], [19, 47], [27, 21], [28, 39], [25, 47], [26, 14], [17, 55], [24, 31], [27, 29], [20, 63], [27, 38], [27, 23], [16, 0], [22, 17], [24, 28], [19, 51], [27, 28], [27, 17], [27, 46], [27, 33], [24, 32], [7, 7], [27, 65], [28, 2], [27, 24], [25, 8], [24, 18], [28, 4], [28, 52], [14, 49], [14, 19], [25, 39], [17, 45], [27, 45], [27, 39], [24, 57], [7, 25], [18, 54], [28, 5], [9, 37], [17, 62], [25, 50], [25, 4], [5, 11], [18, 40], [18, 37], [22, 8], [5, 54], [13, 26], [20, 24], [16, 4]]

falcon_layer_head_id_11b = [[4, 31], [4, 28], [4, 30], [20, 15], [4, 18], [4, 17], [4, 29], [29, 6], [20, 12], [48, 28], [20, 0], [7, 15], [46, 26], [54, 14], [20, 1], [41, 2], [42, 15], [42, 12], [40, 16], [49, 13], [4, 22], [37, 23], [42, 20], [43, 11], [42, 23], [40, 18], [39, 29], [7, 14], [47, 18], [40, 29], [49, 12], [39, 31], [40, 19], [26, 15], [33, 8], [25, 19], [32, 10], [20, 2], [41, 3], [22, 3], [42, 22], [49, 15], [41, 1], [49, 14], [37, 22], [47, 19], [45, 31], [37, 20], [7, 13], [43, 7], [46, 25], [47, 17], [51, 18], [46, 24], [51, 17], [43, 9], [47, 1], [47, 4], [20, 14], [29, 4], [45, 29], [39, 28], [54, 12], [42, 21], [28, 20], [47, 7], [28, 21], [46, 5], [45, 28], [51, 16], [29, 5], [49, 22], [53, 31], [54, 15], [49, 16], [46, 27], [47, 16], [50, 31], [50, 28], [48, 29], [47, 2], [41, 22], [42, 17], [49, 20], [50, 16], [41, 23], [50, 30], [48, 30], [41, 21], [43, 27], [46, 29], [48, 12], [43, 26], [49, 31], [40, 17], [47, 0], [48, 14], [39, 30], [50, 19], [50, 18], [46, 6], [30, 17], [48, 13], [53, 28], [22, 2], [50, 5], [46, 4], [46, 28], [48, 15], [50, 29], [4, 16], [50, 6], [34, 15], [44, 25], [30, 24], [54, 13], [42, 19], [36, 23], [50, 4], [49, 30], [46, 31], [50, 17], [49, 10], [43, 25], [46, 30], [38, 5], [44, 27], [34, 12], [42, 27], [52, 12], [10, 3], [43, 6], [28, 22], [38, 8], [48, 17], [47, 6], [21, 21], [41, 25], [50, 7], [42, 16], [51, 19], [48, 18], [42, 13], [41, 0], [49, 11], [52, 14], [43, 4], [49, 8], [48, 19], [28, 23], [49, 17], [42, 18], [52, 13], [27, 5], [43, 10], [48, 20], [25, 16], [55, 5], [49, 19], [27, 18], [41, 20], [25, 17], [49, 28], [49, 29], [53, 30], [51, 26], [42, 14], [32, 11], [38, 10], [30, 12], [42, 29], [4, 21], [20, 13], [37, 13], [40, 21], [33, 9], [30, 19], [43, 24], [15, 8], [37, 27], [20, 3], [7, 12], [52, 15], [27, 8], [26, 0], [46, 7], [7, 17], [26, 6], [25, 8], [21, 23], [38, 0], [26, 14], [25, 11], [37, 21], [34, 14], [38, 11], [4, 6], [3, 4], [30, 14], [42, 7]]
class Ada_FalconAttention(FalconAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    """
    mask_head_id: Union[int, list] added for masked head
    """

    def __init__(self, config: FalconConfig):
       # Initialize the parent class with the configuration
        super().__init__(config)
        self.mask_head_id = []
        
        
    def set_mask_head_id(self, mask_head_id: Union[int, list]):
        # Ensure the specified head_id is valid
        if mask_head_id < 0 or mask_head_id >= self.num_heads:
            raise ValueError(f"head_id must be between 0 and {self.num_heads - 1}, got {mask_head_id}.")
        # Assign the head_id to this instance
        self.mask_head_id.append(mask_head_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)

        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        if alibi is None:
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size, self.num_heads, kv_length, head_dim]
            #  - value: [batch_size, self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=-2)
            value_layer = torch.cat((past_value, value_layer), dim=-2)

        kv_length = key_layer.shape[-2]
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        if self._use_sdpa and query_layer.device.type == "cuda" and attention_mask is not None:
            # For torch<=2.1.2, SDPA with memory-efficient backend is bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        if alibi is None:
            if self._use_sdpa and not output_attentions:
                attn_output = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask,
                    0.0,
                    # The query_length > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case query_length == 1.
                    is_causal=self.is_causal and attention_mask is None and query_length > 1,
                )

                attention_scores = None
            else:
                attention_scores = query_layer @ key_layer.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)

                attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
                # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
                attn_output = attention_scores @ value_layer

            attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3) # [batch_size, query_length, self.num_heads, self.head_dim] [1, 47, 71, 64]

            ### added by Zhuoyan
            if len(self.mask_head_id) > 0:
                # print("mask_id: ", self.mask_head_id)
                # Create a mask for all heads initially set to one (no masking)
                mask = torch.ones_like(attn_output, device=attn_output.device)

                # Set the specified heads to zero in the mask
                mask[:, :, self.mask_head_id, :] = 0
                # print("mask: ", mask)
                
                # print(mask[:, :, self.mask_head_id, :])
                # Apply the mask to the attn_output
                attn_output_masked = attn_output * mask
                # Verify if the heads are masked correctly
                # print(attn_output_masked[:, :, self.mask_head_id, :])
                # assert False
                attn_output = attn_output_masked
            ####
            # print("attn: ", attn_output)

            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, present, attention_scores
            else:
                return attn_output, present

        else:
            if self._use_sdpa and not output_attentions and head_mask is None:
                attn_output = F.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout.p if self.training else 0.0,
                    is_causal=self.is_causal and attention_mask is None and query_length > 1,
                )
                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

                attn_output = self.dense(attn_output)
            else:
                matmul_result = query_layer @ key_layer.transpose(-1, -2)

                # change view to [batch_size, num_heads, q_length, kv_length]
                attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

                # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
                input_dtype = attention_scores.dtype
                # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
                if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                    attention_scores = attention_scores.to(torch.float32)

                attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
                attention_logits *= self.inv_norm_factor
                attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
                # [batch_size, num_heads, q_length, kv_length]
                attention_probs = self.attention_dropout(attention_probs)

                if head_mask is not None:
                    attention_probs = attention_probs * head_mask

                # change view [batch_size, num_heads, q_length, kv_length]
                attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

                # matmul: [batch_size * num_heads, q_length, head_dim]
                attn_output = (attention_probs_reshaped @ value_layer).flatten(0, 1)

                # change view [batch_size, q_length, num_heads * head_dim]
                attn_output = self._merge_heads(attn_output)

                attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, present, attention_probs
            else:
                return attn_output, present

FALCON_ATTENTION_CLASSES = {
    "eager": FalconAttention,
    "eager_mask": Ada_FalconAttention, 
    "sdpa": FalconAttention,  # FalconAttention originally implemented both a forward with & without SDPA
    "flash_attention_2": FalconFlashAttention2,
}

class Ada_FalconDecoderLayer(FalconDecoderLayer):
    """
    mask_head_id: Union[int, list] added for masked head
    """

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        # self.self_attention = FALCON_ATTENTION_CLASSES[config._attn_implementation](config)
        self.self_attention = FALCON_ATTENTION_CLASSES["eager_mask"](config)

        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if config.num_ln_in_parallel_attn is None and config.new_decoder_architecture:
            config.num_ln_in_parallel_attn = 2

        if not config.parallel_attn:
            self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        else:
            if config.num_ln_in_parallel_attn == 2:
                # The layer norm before self-attention
                self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
                # The layer norm before the MLP
                self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            else:
                self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    
    def set_mask_head_id(self, mask_head_id: Union[int, list]):
        self.self_attention.set_mask_head_id(mask_head_id = mask_head_id)
    

class Ada_FalconModel(FalconModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GemmaDecoderLayer`]

    Args:
        config: FalconConfig
    """

    def __init__(self, config: FalconConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList([Ada_FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.mask_layer_head_id = mask_layer_head_id  # Store the mask_layer_head_id
        # Set masked head ids for each specified Ada_LlamaDecoderLayer
        for layer_id, mask_head_id in mask_layer_head_id:
            if layer_id < len(self.h):
                self.h[layer_id].set_mask_head_id(mask_head_id)


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->GEMMA,Llama->Gemma,llama->gemma, see
# https://github.com/huggingface/transformers/blob/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/models/gemma/modeling_gemma.py#L1038C1-L1038C116
class Ada_FalconForCausalLM(FalconForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.transformer = Ada_FalconModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.transformer.set_mask_layer_head_id(mask_layer_head_id)
