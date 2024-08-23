import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, LayerNorm
from transformers.models.olmo.configuration_olmo import OlmoConfig



from transformers.models.gpt_neox.modeling_gpt_neox import(
    GPTNeoXAttention,
    GPTNeoXForCausalLM,
    GPTNeoXModel,
    GPTNeoXLayer,
    GPTNeoXFlashAttention2,
    GPTNeoXMLP,
    apply_rotary_pos_emb
)


from transformers.cache_utils import Cache, DynamicCache
from pdb import set_trace as pds


# pythia_layer_head_id_7b = [[31, 13], [14, 25], [12, 9], [31, 20], [31, 24], [13, 27], [16, 6], [28, 3], [30, 6], [21, 31], [30, 9], [4, 0], [6, 13], [4, 29], [4, 18], [7, 1], [8, 4], [6, 30], [7, 26], [11, 1], [5, 21], [8, 17], [8, 24], [12, 13], [21, 11], [21, 8], [8, 11], [7, 20], [4, 5], [5, 1], [17, 14], [15, 21], [7, 30], [19, 24], [17, 22], [17, 28], [8, 29], [9, 1], [9, 30], [10, 30], [9, 25], [4, 11], [7, 2], [8, 15], [10, 1], [19, 31], [13, 12], [16, 17], [20, 24], [22, 21], [19, 20], [15, 17], [22, 30], [5, 23], [17, 4], [17, 12], [23, 27], [20, 2], [10, 28], [18, 23], [7, 23], [10, 13], [11, 20], [7, 10], [22, 25], [11, 22], [8, 13], [11, 14], [14, 18], [24, 4], [20, 9], [16, 1], [23, 25], [23, 6], [17, 27], [22, 5], [9, 29], [21, 4], [22, 7], [23, 21], [22, 1], [22, 15], [19, 19], [22, 29], [23, 3], [24, 22], [22, 8], [20, 6], [19, 5], [9, 13], [16, 18], [18, 3], [16, 22], [23, 10], [15, 1], [17, 3], [18, 5], [19, 3], [13, 24], [19, 7], [14, 0], [17, 13], [24, 30], [10, 27], [22, 23], [23, 2], [5, 3], [14, 15], [19, 29], [25, 19], [24, 31], [19, 22], [16, 28], [8, 18], [20, 23], [23, 12], [11, 6], [23, 1], [26, 24], [24, 15], [20, 31], [21, 10], [7, 16], [19, 1], [20, 8], [24, 9], [12, 18], [15, 19], [8, 21], [24, 19], [26, 6], [17, 15], [24, 14], [5, 29], [14, 24], [23, 15], [25, 15], [18, 10], [18, 28], [23, 11], [21, 16], [20, 21], [25, 17], [17, 21], [12, 27], [25, 6], [15, 20], [25, 14], [13, 28], [24, 11], [25, 1], [24, 27], [26, 7], [26, 17], [25, 26], [17, 8], [25, 4], [10, 4], [11, 24], [22, 2], [30, 28], [24, 24], [12, 7], [23, 5], [20, 29], [25, 11], [23, 23], [26, 18], [24, 10], [14, 23], [25, 13], [10, 9], [26, 27], [19, 10], [28, 6], [20, 10], [22, 28], [13, 8], [27, 28], [6, 21], [19, 9], [20, 5], [23, 14], [26, 29], [25, 10], [17, 30], [15, 25], [23, 31], [31, 26], [28, 19], [25, 22], [22, 22], [17, 5], [12, 29], [28, 5], [26, 31], [24, 0], [24, 8], [29, 21], [21, 23]]

# pythia_layer_head_id_7b = [[22, 6], [22, 7], [22, 11], [22, 18], [22, 24], [30, 25], [30, 24], [22, 29], [22, 31], [13, 27], [23, 3], [23, 6], [29, 4], [23, 15], [23, 21], [29, 30], [27, 16], [18, 0], [24, 9], [27, 13], [24, 10], [30, 9], [29, 12], [24, 15], [30, 6], [27, 9], [26, 28], [14, 25], [29, 25], [29, 29], [25, 16], [27, 21], [28, 24], [31, 29], [16, 6], [31, 27], [28, 12], [19, 18], [31, 24], [19, 16], [28, 27], [28, 28], [31, 20], [31, 19], [28, 9], [31, 17], [31, 16], [21, 14], [27, 29], [31, 1], [12, 9], [22, 0], [21, 31], [31, 5], [27, 31], [28, 3], [28, 31], [21, 23], [21, 16], [31, 11], [31, 12], [31, 13], [31, 0], [16, 11], [7, 1], [4, 0], [6, 13], [7, 26], [8, 4], [6, 30], [12, 13], [8, 24], [5, 1], [8, 11], [11, 1], [4, 18], [10, 30], [21, 11], [21, 8], [17, 14], [15, 21], [17, 22], [8, 17], [7, 20], [7, 30], [8, 29], [19, 24], [9, 25], [17, 12], [7, 2], [5, 23], [13, 12], [17, 28], [9, 30], [10, 1], [10, 13], [5, 21], [15, 17], [9, 1], [23, 25], [4, 5], [10, 28], [5, 29], [16, 17], [11, 22], [22, 1], [11, 20], [8, 18], [20, 24], [22, 22], [8, 21], [8, 15], [20, 2], [14, 18], [22, 25], [22, 21], [11, 14], [14, 24], [22, 2], [4, 29], [23, 27], [18, 7], [18, 5], [22, 5], [8, 13], [16, 1], [7, 16], [14, 23], [17, 4], [25, 6], [21, 4], [19, 31], [18, 23], [19, 1], [19, 26], [22, 15], [19, 9], [17, 27], [24, 22], [24, 24], [16, 18], [25, 14], [9, 13], [20, 23], [23, 10], [5, 3], [14, 0], [22, 23], [23, 17], [19, 22], [11, 6], [17, 21], [20, 31], [22, 28], [17, 3], [23, 5], [22, 30], [25, 26], [16, 22], [20, 9], [14, 15], [15, 19], [18, 3], [15, 20], [21, 10], [18, 2], [19, 20], [15, 1], [13, 28], [21, 5], [20, 29], [12, 18], [14, 19], [23, 12], [7, 23], [8, 16], [25, 31], [16, 28], [19, 10], [14, 16], [19, 29], [23, 23], [17, 30], [24, 14], [20, 6], [13, 24], [15, 22], [7, 10], [9, 31], [25, 2], [25, 19], [26, 24], [17, 8], [24, 27], [21, 30], [24, 23], [24, 1], [23, 1], [25, 1], [18, 10]]


pythia_layer_head_id_7b = [[7, 1], [4, 0], [6, 13], [8, 4], [7, 26], [6, 30], [12, 13], [8, 24], [5, 1], [8, 11], [11, 1], [4, 18], [10, 30], [15, 21], [7, 20], [17, 22], [17, 14], [8, 17], [7, 30], [21, 11], [8, 29], [9, 25], [7, 2], [13, 12], [17, 12], [5, 23], [19, 24], [9, 30], [10, 1], [10, 13], [21, 8], [17, 28], [15, 17], [9, 1], [5, 21], [5, 29], [10, 28], [11, 22], [16, 17], [4, 5], [8, 18], [23, 25], [11, 20], [8, 21], [22, 1], [8, 15], [14, 18], [14, 24], [11, 14], [20, 2], [4, 29], [20, 24], [18, 7], [22, 22], [8, 13], [16, 1], [23, 27], [18, 5], [14, 23], [7, 16], [22, 5], [22, 2], [17, 4], [22, 25], [19, 31], [18, 23], [19, 26], [22, 21], [19, 1], [16, 18], [14, 0], [5, 3], [9, 13], [24, 22], [20, 23], [21, 4], [17, 27], [11, 6], [17, 21], [22, 15], [23, 10], [24, 24], [19, 22], [19, 9], [25, 14], [16, 22], [17, 3], [15, 19], [14, 15], [15, 1], [22, 30], [12, 18], [23, 17], [7, 23], [25, 26], [15, 20], [18, 2], [14, 19], [20, 31], [19, 20], [18, 3], [13, 28], [23, 15], [9, 31], [15, 22], [8, 16], [17, 30], [14, 16], [21, 10], [22, 23], [19, 29], [25, 31], [23, 5], [16, 28], [17, 8], [20, 9], [7, 10], [21, 5], [19, 10], [23, 23], [18, 10], [13, 24], [26, 24], [19, 0], [23, 12], [20, 29], [24, 1], [24, 14], [17, 26], [12, 29], [23, 21], [9, 29], [24, 31], [24, 27], [26, 1], [25, 1], [18, 27], [23, 6], [20, 6], [27, 4], [24, 23], [26, 9], [25, 2], [17, 15], [26, 23], [20, 25], [9, 22], [25, 19], [28, 26], [12, 7], [18, 4], [19, 19], [22, 28], [15, 4], [25, 22], [14, 27], [21, 30], [23, 1], [26, 30], [23, 3], [5, 27], [7, 4], [25, 13], [8, 9], [8, 5], [12, 3], [23, 2], [10, 27], [15, 25], [12, 6], [27, 29], [22, 7], [17, 11], [17, 13], [18, 28], [12, 5], [24, 11], [19, 3], [22, 29], [23, 28], [11, 24], [21, 16], [20, 12], [28, 31], [28, 21], [26, 6], [21, 20], [19, 5], [22, 8], [9, 8], [13, 18], [19, 25], [6, 21], [23, 14], [24, 10], [16, 10], [20, 8], [29, 1], [13, 9], [28, 4]]

class Ada_GPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, config):
        # Initialize the parent class with the configuration
        super().__init__(config)
        self.mask_head_id = []
    
    def set_mask_head_id(self, mask_head_id: Union[int, list]):
        # Ensure the specified head_id is valid
        if mask_head_id < 0 or mask_head_id >= self.num_attention_heads:
            raise ValueError(f"head_id must be between 0 and {self.num_attention_heads - 1}, got {mask_head_id}.")
        # Assign the head_id to this instance
        self.mask_head_id.append(mask_head_id)
        
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        # attn_output:  [bs, num_head, query_len, head_dim]   torch.Size([1, 32, 48, 128])

        ### added by Zhuoyan
        if len(self.mask_head_id) > 0:
            # print("mask_id: ", self.mask_head_id)
            # Create a mask for all heads initially set to one (no masking)
            mask = torch.ones_like(attn_output, device=attn_output.device)

            # Set the specified heads to zero in the mask
            mask[:, self.mask_head_id, :, :] = 0
            # print("mask: ", mask)
            
            # print(mask[:, :, self.mask_head_id, :])
            # Apply the mask to the attn_output
            attn_output_masked = attn_output * mask
            # Verify if the heads are masked correctly
            # print(attn_output_masked[:, :, self.mask_head_id, :])
            # assert False
            attn_output = attn_output_masked
        ####

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    

GPT_NEOX_ATTENTION_CLASSES = {
    "eager": GPTNeoXAttention,
    "eager_mask": Ada_GPTNeoXAttention,
    "flash_attention_2": GPTNeoXFlashAttention2,
}


class Ada_PythiaLayer(GPTNeoXLayer):
    """
    mask_head_id: Union[int, list] added for masked head
    """
    def __init__(self, config):
        super().__init__(config)
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)

        
        # self.attention = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation](config)
        self.attention = GPT_NEOX_ATTENTION_CLASSES["eager_mask"](config)

        self.mlp = GPTNeoXMLP(config)
    
    def set_mask_head_id(self, mask_head_id: Union[int, list]):
        self.attention.set_mask_head_id(mask_head_id = mask_head_id)



class Ada_PythiaModel(GPTNeoXModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GemmaDecoderLayer`]

    Args:
        config: FalconConfig
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)

        # self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)])
        self.layers = nn.ModuleList([Ada_PythiaLayer(config) for _ in range(config.num_hidden_layers)])


        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.mask_layer_head_id = mask_layer_head_id  # Store the mask_layer_head_id
        # Set masked head ids for each specified Ada_LlamaDecoderLayer
        for layer_id, mask_head_id in mask_layer_head_id:
            if layer_id < len(self.layers):
                self.layers[layer_id].set_mask_head_id(mask_head_id)




class Ada_PythiaForCausalLM(GPTNeoXForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    
    
    def __init__(self, config):
        super().__init__(config)

        # self.gpt_neox = GPTNeoXModel(config)
        self.gpt_neox = Ada_PythiaModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.gpt_neox.set_mask_layer_head_id(mask_layer_head_id)


