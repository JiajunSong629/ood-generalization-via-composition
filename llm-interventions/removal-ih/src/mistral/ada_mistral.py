import math
from typing import List, Optional, Tuple, Union
import warnings

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    MistralMLP,
    MistralRMSNorm,
    MistralDecoderLayer,
    MistralModel,
    MistralForCausalLM,
)
from transformers.cache_utils import Cache, DynamicCache



# mistral_layer_head_id =  [[12, 6], [12, 7], [12, 4], [2, 21], [18, 2], [2, 22], [18, 0], [28, 0], [2, 23], [18, 3], [19, 8], [22, 30], [14, 28], [4, 12], [26, 6], [20, 28], [26, 5], [25, 29], [14, 31], [14, 19], [18, 1], [21, 3], [17, 26], [16, 17], [30, 2], [21, 2], [4, 15], [26, 14], [19, 10], [31, 4], [30, 8], [28, 2], [21, 0], [22, 29], [20, 6], [31, 31], [28, 16], [23, 30], [24, 6], [22, 8], [31, 29], [27, 29], [28, 26], [19, 17], [20, 9], [29, 9], [13, 5], [24, 5], [21, 9], [20, 29]]

mistral_layer_head_id = [[12, 7], [12, 6], [18, 0], [28, 0], [18, 2], [2, 21], [18, 3], [12, 4], [2, 23], [2, 22], [18, 1], [19, 8], [22, 29], [19, 9], [26, 6], [20, 6], [31, 31], [19, 17], [23, 30], [11, 14], [4, 12], [17, 26], [28, 2], [21, 3], [30, 2], [20, 29], [20, 30], [25, 29], [22, 6], [30, 8], [19, 19], [20, 28], [24, 5], [22, 4], [19, 16], [18, 22], [26, 5], [15, 25], [27, 29], [21, 1], [13, 5], [17, 25], [29, 10], [22, 8], [31, 4], [16, 12], [29, 9], [4, 15], [20, 31], [15, 26], [14, 31], [14, 18], [24, 6], [22, 30], [16, 22], [15, 27], [24, 11], [20, 17], [26, 14], [31, 5], [19, 10], [21, 0], [25, 27], [21, 2], [23, 13], [23, 15], [28, 25], [23, 12], [28, 26], [18, 30], [23, 14], [25, 18], [9, 25], [22, 1], [22, 0], [21, 10], [22, 3], [29, 22], [14, 26], [20, 9], [18, 21], [14, 24], [30, 1], [25, 16], [30, 10], [21, 16], [19, 5], [21, 11], [31, 17], [31, 16], [28, 18], [16, 13], [30, 3], [15, 8], [31, 18], [21, 9], [20, 18], [22, 11], [19, 12], [31, 29], [31, 19], [31, 6], [19, 11], [24, 21], [18, 12], [24, 14], [26, 16], [9, 26], [30, 26], [28, 16], [28, 31], [27, 27], [19, 4], [24, 12], [18, 29], [17, 24], [26, 17], [16, 17], [26, 4], [18, 31], [25, 10], [2, 0], [17, 0], [19, 14], [25, 8], [25, 9], [16, 21], [24, 13], [15, 16], [18, 11], [24, 15], [26, 24], [30, 4], [16, 29], [26, 12], [18, 19], [21, 4], [20, 13], [19, 18], [16, 30], [27, 26], [16, 7], [17, 27], [25, 30], [5, 5], [20, 26], [15, 0], [17, 1], [26, 27], [14, 19], [20, 11], [14, 28], [30, 6], [20, 4], [11, 12], [21, 8], [29, 16], [21, 17], [24, 23], [30, 22], [20, 10], [20, 14], [16, 14], [15, 22], [15, 1], [17, 2], [19, 6], [12, 9], [28, 15], [14, 21], [19, 7], [20, 5], [18, 13], [29, 2], [12, 28], [13, 28], [22, 31], [24, 22], [19, 15], [15, 10], [21, 19], [20, 23], [20, 20], [24, 4], [19, 30], [16, 27], [29, 14], [11, 20], [15, 7], [21, 6], [18, 14], [20, 12], [15, 20], [30, 11], [26, 9], [16, 25], [20, 27], [23, 8], [31, 27], [20, 2]]

class Ada_MistralAttention(MistralAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    mask_head_id: Union[int, list] added for masked head
    """
    def __init__(self, config: MistralConfig,layer_idx: Optional[int] = None):
       # Initialize the parent class with the configuration
        super().__init__(config, layer_idx = layer_idx)
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
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # print("attn_weights: ", type(attn_weights), attn_weights.shape) # [bs, n_head. seq_len,seq_len] [1, 32, 40, 40]
        # print(attn_weights[0][0])

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # print("attn_output: ", type(attn_output), attn_output.shape) # [1, 40, 32, 128]
        # assert False

        #### added by Zhuoyan
        if len(self.mask_head_id) > 0:
            # print("mask_id: ", self.mask_head_id)
            # Create a mask for all heads initially set to one (no masking)
            mask = torch.ones_like(attn_output)

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



class Ada_MistralDecoderLayer(MistralDecoderLayer):
    '''
    mask_head_id: Union[int, list] added for masked head
    '''

    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = (
            Ada_MistralAttention(config=config, layer_idx=layer_idx)
        )
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def set_mask_head_id(self, mask_head_id: Union[int, list]):
        self.self_attn.set_mask_head_id(mask_head_id = mask_head_id)



class Ada_MistralModel(MistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Ada_MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.mask_layer_head_id = mask_layer_head_id  # Store the mask_layer_head_id
        # Set masked head ids for each specified Ada_LlamaDecoderLayer
        for layer_id, mask_head_id in mask_layer_head_id:
            if layer_id < len(self.layers):
                self.layers[layer_id].set_mask_head_id(mask_head_id)


class Ada_MistralForCausalLM(MistralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Ada_MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.model.set_mask_layer_head_id(mask_layer_head_id)


