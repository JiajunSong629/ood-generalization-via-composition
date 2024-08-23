import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, LayerNorm
from transformers.models.olmo.configuration_olmo import OlmoConfig


from transformers.models.olmo.modeling_olmo import (
    OlmoLayerNorm,
    repeat_kv,
    apply_rotary_pos_emb,
    OlmoAttention,
    OlmoSdpaAttention,
    OlmoFlashAttention2,
    OlmoMLP,
    OlmoDecoderLayer,
    OlmoModel,
    OlmoForCausalLM
)
from transformers.cache_utils import Cache, DynamicCache
from pdb import set_trace as pds
# olmo_layer_head_id_7b = [[2, 28], [27, 14], [2, 10], [26, 17], [28, 28], [2, 22], [2, 16], [2, 4], [15, 15], [27, 26], [29, 30], [6, 25], [29, 24], [29, 8], [29, 29], [29, 12], [29, 21], [29, 27], [26, 12], [28, 15], [28, 31], [28, 11], [29, 14], [30, 23], [29, 20], [29, 18], [28, 8], [27, 20], [28, 24], [29, 1], [24, 7], [27, 15], [29, 10], [30, 30], [29, 26], [29, 23], [30, 17], [30, 29], [29, 7], [30, 1], [27, 13], [29, 17], [30, 5], [30, 19], [30, 13], [27, 8], [27, 7], [30, 3], [29, 11], [30, 6], [31, 13], [7, 8], [31, 21], [31, 25], [28, 0], [24, 27], [31, 19], [26, 9], [27, 6], [26, 1], [22, 16], [7, 20], [26, 29], [26, 23], [26, 5], [21, 4], [27, 9], [24, 9], [26, 19], [28, 16], [28, 23], [7, 26], [31, 7], [31, 9], [19, 14], [21, 22], [15, 31], [26, 11], [24, 1], [31, 1], [9, 14], [28, 14], [6, 13], [22, 22], [20, 17], [19, 17], [22, 2], [22, 14], [22, 8], [16, 6], [20, 31], [5, 7], [9, 8], [27, 2], [25, 5], [2, 30], [26, 2], [28, 30], [5, 30], [4, 10], [28, 5], [26, 21], [5, 22], [30, 0], [15, 3], [24, 13], [19, 26], [22, 26], [5, 25], [0, 8], [13, 14], [12, 24], [0, 27], [31, 12], [11, 13], [26, 6], [25, 12], [23, 15], [27, 19], [14, 23], [16, 18], [25, 3], [19, 0], [26, 31], [30, 7], [11, 0], [10, 13], [23, 28], [0, 15], [17, 1], [14, 28], [28, 9], [9, 6], [2, 2], [19, 23], [3, 16], [9, 13], [22, 17], [23, 19], [13, 17], [23, 4], [23, 16], [21, 8], [17, 30], [24, 24], [19, 12], [18, 11], [27, 23], [22, 3], [21, 26], [8, 12], [30, 12], [20, 16], [26, 14], [12, 6], [25, 31], [24, 26], [18, 6], [24, 6], [11, 17], [9, 17], [21, 3], [22, 12], [30, 10], [27, 25], [20, 4], [13, 16], [26, 20], [16, 13], [23, 31], [10, 12], [28, 2], [3, 4], [11, 12], [15, 16], [1, 16], [0, 4], [27, 21], [5, 5], [22, 1], [30, 16], [19, 10], [18, 26], [26, 10], [1, 5], [30, 18], [23, 26], [8, 26], [0, 12], [5, 16], [18, 31], [15, 8], [8, 10], [11, 22], [0, 23], [23, 10], [25, 8], [14, 13], [21, 15], [25, 15]]

olmo_layer_head_id_7b = [[27, 14], [2, 28], [2, 22], [26, 17], [2, 10], [28, 28], [2, 16], [15, 15], [27, 26], [29, 30], [24, 7], [29, 8], [28, 15], [29, 21], [29, 24], [26, 12], [30, 17], [29, 18], [6, 25], [29, 12], [26, 9], [27, 20], [29, 29], [28, 8], [29, 7], [29, 10], [30, 13], [29, 17], [29, 26], [29, 11], [27, 6], [29, 23], [27, 15], [30, 1], [29, 27], [29, 14], [30, 23], [28, 31], [29, 1], [29, 20], [27, 8], [24, 1], [28, 11], [28, 0], [30, 6], [30, 19], [30, 30], [28, 24], [30, 29], [30, 3], [28, 16], [27, 9], [26, 1], [30, 5], [22, 22], [27, 7], [26, 11], [26, 23], [26, 29], [27, 13], [31, 19], [26, 5], [31, 25], [22, 14], [31, 7], [31, 13], [31, 21], [31, 9], [24, 9], [7, 20], [26, 21], [26, 2], [22, 16], [26, 19], [16, 18], [21, 4], [26, 6], [24, 24], [24, 27], [22, 8], [23, 19], [2, 4], [27, 2], [26, 31], [30, 0], [28, 30], [28, 14], [22, 2], [7, 8], [25, 16], [25, 12], [31, 1], [28, 25], [24, 13], [28, 23], [22, 1], [2, 30], [19, 14], [24, 19], [22, 26], [30, 7], [15, 31], [20, 31], [26, 22], [22, 28], [19, 26], [25, 3], [30, 12], [23, 31], [23, 10], [24, 18], [28, 27], [26, 18], [16, 6], [26, 26], [21, 22], [24, 31], [22, 10], [24, 21], [28, 17], [29, 3], [7, 26], [23, 16], [3, 16], [28, 5], [27, 25], [28, 9], [15, 19], [23, 15], [27, 19], [15, 27], [4, 21], [5, 22], [23, 26], [9, 17], [21, 26], [3, 11], [26, 14], [10, 2], [14, 28], [23, 4], [22, 7], [10, 13], [18, 11], [17, 1], [4, 25], [19, 16], [25, 8], [19, 23], [15, 7], [24, 6], [18, 6], [25, 14], [31, 2], [11, 0], [22, 17], [29, 16], [19, 0], [5, 7], [21, 3], [24, 14], [13, 16], [22, 12], [12, 6], [23, 28], [22, 31], [20, 4], [26, 28], [11, 22], [28, 2], [16, 13], [26, 10], [15, 16], [26, 20], [28, 29], [17, 30], [25, 31], [5, 16], [26, 27], [21, 25], [30, 10], [9, 14], [12, 15], [14, 12], [15, 8], [31, 12], [10, 9], [11, 16], [13, 17], [10, 12], [9, 13], [20, 8], [18, 24], [19, 18], [11, 17], [23, 3], [20, 16], [16, 14], [8, 26], [19, 6]]


class Ada_OlmoAttention(OlmoAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    """
    mask_head_id: Union[int, list] added for masked head
    """

    def __init__(self, config: OlmoConfig, layer_idx: Optional[int] = None):
        # Initialize the parent class with the configuration
        super().__init__(config, layer_idx)
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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # attn_output:  [bs, num_head, query_len, head_dim]   torch.Size([1, 32, 48, 128])
        
        attn_output = attn_output.transpose(1, 2).contiguous()  # [bs, query_len, num_head, head_dim]   torch.Size([1, 48, 32, 128])
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

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


OLMO_ATTENTION_CLASSES = {
    "eager": OlmoAttention,
    "eager_mask": Ada_OlmoAttention,
    "flash_attention_2": OlmoFlashAttention2,
    "sdpa": OlmoSdpaAttention,
}



class Ada_OlmoDecoderLayer(OlmoDecoderLayer):
    """
    mask_head_id: Union[int, list] added for masked head
    """

    def __init__(self, config: OlmoConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        

        # self.self_attn = OLMO_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn = OLMO_ATTENTION_CLASSES["eager_mask"](config=config, layer_idx=layer_idx)

        self.mlp = OlmoMLP(config)
        self.input_layernorm = OlmoLayerNorm(config.hidden_size)
        self.post_attention_layernorm = OlmoLayerNorm(config.hidden_size)
    
    def set_mask_head_id(self, mask_head_id: Union[int, list]):
        self.self_attn.set_mask_head_id(mask_head_id = mask_head_id)
    


class Ada_OlmoModel(OlmoModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GemmaDecoderLayer`]

    Args:
        config: FalconConfig
    """
    
    def __init__(self, config: OlmoConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Ada_OlmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OlmoLayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.mask_layer_head_id = mask_layer_head_id  # Store the mask_layer_head_id
        # Set masked head ids for each specified Ada_LlamaDecoderLayer
        for layer_id, mask_head_id in mask_layer_head_id:
            if layer_id < len(self.layers):
                self.layers[layer_id].set_mask_head_id(mask_head_id)


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->GEMMA,Llama->Gemma,llama->gemma, see
# https://github.com/huggingface/transformers/blob/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/models/gemma/modeling_gemma.py#L1038C1-L1038C116
class Ada_OlmoForCausalLM(OlmoForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = Ada_OlmoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.model.set_mask_layer_head_id(mask_layer_head_id)




