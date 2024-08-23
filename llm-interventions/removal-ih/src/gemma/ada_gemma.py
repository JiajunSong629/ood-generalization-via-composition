import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaFlashAttention2,
    GemmaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    GemmaMLP,
    GemmaRMSNorm,
    GemmaDecoderLayer,
    GemmaModel,
    GemmaForCausalLM,
)
from transformers.cache_utils import Cache, DynamicCache

# gemma_layer_head_id_7b = [[20, 13], [14, 15], [5, 0], [20, 1], [21, 1], [2, 14], [21, 2], [23, 6], [18, 13], [15, 2], [22, 12], [21, 5], [20, 0], [22, 7], [22, 11], [22, 10], [24, 5], [23, 4], [23, 10], [22, 2], [23, 8], [21, 3], [24, 3], [23, 0], [22, 3], [17, 6], [20, 15], [22, 14], [24, 6], [24, 11], [4, 5], [24, 4], [25, 1], [21, 6], [24, 1], [8, 6], [19, 5], [22, 1], [20, 12], [25, 7], [21, 11], [17, 2], [21, 15], [19, 4], [24, 14], [14, 13], [9, 12], [23, 9], [19, 6], [22, 0], [14, 6], [19, 1], [8, 12], [11, 12], [15, 6], [24, 13], [5, 5], [19, 8], [23, 2], [12, 3], [17, 10], [10, 0], [25, 15], [13, 0], [12, 13], [15, 13], [15, 15], [18, 8], [21, 8], [13, 10], [9, 6], [19, 0], [26, 12], [16, 12], [18, 5], [10, 6], [18, 0], [11, 2], [14, 10], [17, 12], [21, 12], [3, 13], [15, 11], [13, 12], [8, 0], [6, 14], [14, 3], [16, 4], [13, 13], [14, 9], [10, 3], [17, 4], [17, 15], [18, 7], [14, 8], [8, 10], [5, 12], [8, 3], [11, 10], [18, 3], [25, 13], [16, 14], [19, 7], [16, 13], [16, 5], [9, 2], [19, 9], [14, 11], [7, 9], [10, 4], [24, 2], [12, 5], [12, 10], [24, 15], [14, 5], [17, 3], [9, 0], [10, 9], [21, 14], [24, 7], [9, 7], [1, 8], [7, 10], [19, 2], [15, 0], [11, 8], [20, 9], [7, 7], [7, 4], [2, 7], [0, 12], [1, 12], [16, 2], [20, 10], [11, 9], [20, 6], [27, 12], [16, 6], [15, 4], [18, 15], [12, 2], [17, 5], [20, 2], [0, 11], [22, 13], [0, 13], [9, 15], [17, 7], [13, 1], [17, 9], [0, 3], [8, 13], [16, 11], [18, 12], [14, 1], [26, 13], [15, 12], [15, 5], [16, 3], [10, 15], [18, 1], [19, 11], [2, 12], [6, 12], [9, 1], [11, 11], [11, 13], [15, 8], [5, 3], [2, 9], [15, 3], [9, 9], [8, 15], [17, 1], [16, 10], [12, 11], [21, 7], [10, 8], [1, 11], [25, 12], [10, 1], [24, 10], [0, 5], [12, 14], [1, 1], [14, 7], [20, 3], [4, 3], [6, 2], [15, 1], [16, 8], [13, 6], [22, 15], [7, 8], [13, 7], [15, 14], [26, 2], [5, 1], [20, 11], [23, 13]]

gemma_layer_head_id_7b = [[20, 13], [14, 15], [21, 2], [21, 1], [24, 10], [23, 6], [18, 13], [5, 0], [20, 0], [22, 12], [20, 12], [21, 5], [21, 11], [20, 1], [22, 7], [20, 11], [21, 12], [16, 1], [22, 10], [23, 4], [20, 4], [24, 5], [22, 11], [27, 14], [23, 0], [25, 7], [21, 3], [23, 8], [24, 14], [22, 1], [24, 1], [24, 13], [22, 3], [24, 3], [24, 4], [26, 0], [24, 6], [23, 10], [24, 15], [22, 14], [24, 7], [22, 2], [24, 11], [21, 8], [15, 2], [25, 1], [18, 10], [18, 0], [2, 14], [19, 2], [17, 2], [20, 15], [21, 6], [20, 10], [18, 6], [22, 0], [25, 9], [21, 14], [19, 7], [18, 2], [14, 3], [18, 3], [18, 14], [19, 14], [26, 5], [26, 2], [19, 8], [8, 6], [19, 1], [16, 10], [26, 12], [16, 4], [21, 15], [22, 15], [18, 8], [5, 5], [16, 7], [24, 8], [18, 4], [19, 12], [12, 3], [18, 1], [19, 10], [25, 5], [16, 3], [26, 13], [20, 9], [19, 4], [20, 2], [25, 6], [26, 11], [23, 12], [24, 2], [25, 11], [16, 15], [14, 2], [15, 12], [16, 6], [14, 1], [23, 2], [20, 14], [16, 12], [17, 13], [25, 0], [23, 9], [25, 13], [25, 10], [26, 14], [18, 7], [18, 9], [25, 15], [26, 9], [14, 6], [4, 5], [16, 8], [12, 8], [20, 6], [15, 11], [19, 3], [25, 3], [19, 5], [11, 13], [14, 7], [16, 14], [19, 6], [25, 12], [23, 14], [17, 6], [14, 11], [17, 3], [16, 5], [22, 13], [23, 11], [4, 15], [19, 0], [14, 8], [10, 8], [11, 12], [14, 0], [23, 7], [20, 3], [6, 6], [27, 12], [4, 11], [12, 4], [12, 1], [4, 13], [20, 7], [5, 2], [10, 1], [14, 5], [17, 5], [2, 6], [5, 4], [17, 9], [12, 14], [10, 0], [8, 12], [6, 12], [7, 4], [12, 7], [14, 4], [4, 2], [10, 11], [5, 3], [2, 11], [6, 14], [21, 4], [6, 15], [7, 13], [23, 13], [6, 0], [17, 4], [6, 13], [9, 3], [5, 7], [1, 8], [7, 10], [6, 5], [27, 9], [26, 8], [3, 3], [5, 12], [0, 12], [22, 9], [6, 2], [26, 6], [18, 5], [13, 10], [7, 9], [9, 1], [10, 15], [13, 14], [12, 2], [12, 12], [8, 1], [17, 14], [5, 14], [2, 7], [12, 6]]

gemma2_layer_head_id_9b = [[7, 1], [5, 1], [7, 0], [11, 2], [15, 3], [17, 0], [9, 15], [15, 2], [14, 12], [17, 5], [25, 13], [25, 15], [28, 2], [29, 10], [21, 4], [24, 10], [24, 0], [29, 11], [34, 3], [11, 3], [29, 15], [21, 5], [20, 11], [27, 8], [31, 0], [35, 14], [25, 12], [33, 9], [31, 15], [31, 10], [19, 8], [40, 6], [28, 8], [32, 14], [13, 7], [19, 0], [11, 9], [35, 4], [38, 1], [21, 12], [34, 12], [28, 9], [26, 9], [38, 4], [20, 15], [38, 8], [21, 3], [26, 4], [39, 5], [29, 14], [26, 5], [26, 2], [41, 7], [28, 4], [18, 8], [26, 0], [27, 9], [23, 13], [12, 8], [22, 1], [14, 13], [23, 3], [26, 12], [29, 6], [15, 11], [33, 10], [26, 1], [30, 12], [21, 2], [23, 15], [15, 4], [13, 15], [28, 6], [29, 0], [33, 1], [21, 13], [33, 15], [38, 3], [35, 0], [35, 1], [41, 10], [36, 4], [14, 6], [25, 8], [18, 10], [37, 12], [20, 5], [34, 6], [14, 0], [18, 9], [28, 12], [19, 1], [31, 7], [7, 7], [23, 12], [24, 15], [29, 13], [30, 8], [32, 5], [34, 13], [13, 10], [33, 14], [20, 4], [22, 6], [24, 9], [13, 4], [30, 9], [13, 13], [18, 6], [18, 15], [14, 9], [36, 7], [37, 14], [25, 10], [4, 0], [32, 13], [12, 0], [18, 12], [40, 3], [37, 1], [7, 9], [36, 1], [32, 12], [17, 1], [24, 7], [34, 14], [34, 15], [30, 13], [17, 6], [7, 2], [35, 7], [28, 3], [30, 3], [8, 8], [21, 10], [34, 5], [25, 9], [8, 15], [39, 7], [28, 10], [39, 2], [4, 11], [27, 10], [23, 4], [19, 2], [22, 13], [13, 11], [24, 1], [31, 8], [32, 3], [9, 4], [21, 7], [26, 7], [17, 4], [39, 10], [35, 2], [39, 11], [41, 13], [13, 8], [20, 8], [31, 4], [28, 13], [18, 14], [18, 11], [23, 14], [15, 5], [32, 15], [27, 3], [29, 1], [27, 14], [22, 3], [9, 12], [6, 6], [40, 14], [26, 8], [37, 0], [29, 8], [11, 8], [14, 4], [23, 9], [24, 12], [8, 2], [21, 0], [31, 13], [41, 12], [39, 9], [29, 7], [28, 11], [37, 6], [41, 8], [36, 6], [29, 12], [17, 11], [18, 2], [37, 9], [32, 7], [22, 2], [30, 4], [39, 13], [19, 6]]

'''
class Ada_LlamaMLP(LlamaMLP):
    def forward(self, x):
        ### simple hack by Zhuoyan
        print(f"1 Before operation, x device: {x.device}, gate_proj weight device: {self.gate_proj.weight.device}")

        # Print device of input tensor
        print(f'Input x device: {x.device}')
        
        # Print devices of model components
        print(f'gate_proj device: {self.gate_proj.weight.device}')
        print(f'up_proj device: {self.up_proj.weight.device}')
        print(f'down_proj device: {self.down_proj.weight.device}')

        # Determine the target device from the input tensor 'x'
        target_device = x.device
        
        # Ensure model parameters are on the same device as 'x'
        self.gate_proj.to(target_device)
        self.up_proj.to(target_device)
        self.down_proj.to(target_device)
        print(f"2 Before operation, x device: {x.device}, gate_proj weight device: {self.gate_proj.weight.device}")

        ### 

        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            print("else")
            print(f"3 Before operation, x device: {x.device}, gate_proj weight device: {self.gate_proj.weight.device}")

            # Ensure intermediate results are also moved to the correct device
            gate_proj_output = self.gate_proj(x).to(target_device)
            up_proj_output = self.up_proj(x).to(target_device)
            act_fn_output = self.act_fn(gate_proj_output).to(target_device)

            # The final operation
            down_proj = self.down_proj(act_fn_output * up_proj_output)
            #down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
##

'''


class Ada_GemmaAttention(GemmaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    """
    mask_head_id: Union[int, list] added for masked head
    """

    def __init__(self, config: GemmaConfig,layer_idx: Optional[int] = None):
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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        

        # print("attn_weights: ", type(attn_weights), attn_weights.shape, attn_weights[0][0]) # [bs, num_head, seq_len, seq_len] 2b: [1, 8, 15, 15]
        # print("attn_weights: ", type(attn_weights), attn_weights.shape) # [bs, num_head, seq_len, seq_len] 7b: [1, 16, 15, 15]

        # assert False

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        
        # print("attention_mask: ", type(attention_mask), attention_mask.shape, attention_mask) # [1, 1, 15, 16]
        # print("attention_mask: ", type(attention_mask), attention_mask.shape) # [1, 1, 15, 16]

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # print("value_states: ", type(value_states), value_states.shape) # [1, 8, 15, 256]
        attn_output = torch.matmul(attn_weights, value_states)
        # print("attn_output: ", type(attn_output), attn_output.shape) # [1, 8, 15, 256]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()


        # print("attn_output: ", type(attn_output), attn_output.shape) # [bs, seq_len, num_head, head_dim] 2b: [1, 15, 8, 256]
        # print("attn_output: ", type(attn_output), attn_output.shape) # [bs, seq_len, num_head, head_dim] 7b: [1, 15, 16, 256]
        # print("attn device: ", attn_output.device)


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

        # assert False

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # print("attn_output: ", type(attn_output), attn_output.shape)
        # print("zhuoyan==============================")
        # print("output_attentions: ", output_attentions)
        # print("attn_output: ", type(attn_output), attn_output.shape, attn_output)
        # assert False
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    
GEMMA_ATTENTION_CLASSES = {
    "eager": GemmaAttention,
    "eager_mask": Ada_GemmaAttention, 
    "flash_attention_2": GemmaFlashAttention2,
    "sdpa": GemmaSdpaAttention,
}

class Ada_GemmaDecoderLayer(GemmaDecoderLayer):
    """
    mask_head_id: Union[int, list] added for masked head
    """

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__(config, layer_idx)   
        self.hidden_size = config.hidden_size
        # config._attn_implementation = "eager_mask"
        # print("inside attn imple: ", config._attn_implementation)
        # self.self_attn = GEMMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn = GEMMA_ATTENTION_CLASSES["eager_mask"](config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    
    def set_mask_head_id(self, mask_head_id: Union[int, list]):
        self.self_attn.set_mask_head_id(mask_head_id = mask_head_id)
    
    '''
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        coied from original forward function
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # print(f"1 residual device: {residual.device} | hidden_states device {hidden_states.device}")
        hidden_states = self.post_attention_layernorm(hidden_states)
        # print(f"2 residual device: {residual.device} | hidden_states device {hidden_states.device}")

        ### simple hack by Zhuoyan
        # target_device = hidden_states.device
        # self.mlp.to(target_device)
        ###
        hidden_states = self.mlp(hidden_states)
        # print(f"3 residual device: {residual.device} | hidden_states device {hidden_states.device}")

        ### simple hack by Zhuoyan
        # Check if residual is not on the same device as hidden_states
        if residual.device != hidden_states.device:
            # Move residual to the same device as hidden_states
            residual = residual.to(hidden_states.device)
        ### end zhuoyan
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    '''



class Ada_GemmaModel(GemmaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GemmaDecoderLayer`]

    Args:
        config: GemmaConfig
    """

    def __init__(self, config: GemmaConfig, ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Ada_GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        #self.layers = nn.ModuleList([GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
class Ada_GemmaForCausalLM(GemmaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Ada_GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.model.set_mask_layer_head_id(mask_layer_head_id)
