import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)
from transformers.cache_utils import Cache, DynamicCache

llama2_layer_head_id_7b = [[11, 15], [8, 26], [6, 9], [7, 4], [6, 30], [21, 30], [7, 12], [16, 19], [26, 28], [6, 16], [17, 22], [18, 30], [7, 10], [19, 15], [12, 26], [11, 2], [8, 31], [12, 2], [13, 23], [22, 22], [7, 28], [7, 13], [20, 8], [24, 30], [24, 3], [6, 11], [19, 27], [15, 14], [24, 11], [21, 28], [17, 0], [19, 10], [17, 18], [22, 27], [19, 9], [24, 29], [20, 29], [24, 8], [21, 1], [14, 18], [21, 16], [22, 30], [22, 8], [17, 16], [20, 1], [20, 0], [22, 19], [23, 20], [20, 3], [14, 7], [3, 6], [20, 10], [21, 26], [17, 13], [20, 27], [13, 11], [19, 17], [18, 18], [17, 11], [18, 16], [22, 24], [23, 8], [23, 7], [18, 10], [21, 29], [20, 30], [11, 14], [10, 14], [19, 14], [17, 31], [21, 27], [18, 31], [16, 30], [14, 24], [16, 24], [23, 31], [26, 26], [25, 17], [19, 21], [20, 28], [26, 25], [24, 15], [10, 18], [17, 2], [21, 4], [22, 16], [22, 9], [26, 24], [21, 5], [19, 19], [24, 17], [11, 18], [24, 16], [14, 13], [23, 0], [14, 3], [26, 27], [17, 30], [18, 9], [25, 21], [24, 24], [29, 19], [15, 25], [24, 4], [6, 20], [16, 17], [14, 9], [12, 4], [20, 11], [19, 8], [16, 1], [28, 14], [16, 2], [22, 23], [21, 9], [19, 6], [10, 29], [14, 26], [11, 22], [21, 31], [19, 25], [17, 10], [14, 2], [26, 21], [29, 13], [16, 27], [16, 29], [26, 3], [19, 7], [14, 29], [25, 0], [16, 13], [15, 5], [16, 5], [12, 25], [14, 1], [21, 15], [12, 16], [18, 23], [20, 12], [30, 14], [5, 7], [25, 3], [14, 19], [19, 12], [8, 5], [15, 19], [22, 17], [11, 30], [26, 23], [7, 31], [18, 15], [14, 20], [24, 14], [31, 16], [16, 18], [13, 9], [11, 17], [21, 10], [17, 27], [18, 1], [16, 31], [15, 31], [29, 26], [19, 4], [20, 5], [17, 9], [25, 22], [9, 25], [15, 10], [29, 21], [8, 22], [28, 21], [21, 13], [14, 4], [23, 28], [8, 0], [22, 31], [9, 17], [15, 27], [28, 7], [10, 4], [14, 27], [25, 7], [17, 25], [14, 12], [23, 17], [25, 11], [11, 31], [14, 15], [25, 26], [26, 9], [14, 17], [14, 11], [19, 3], [11, 4], [15, 22], [15, 24], [16, 9], [19, 31]]

llama2_layer_head_id_70b = [[69, 32], [4, 37], [44, 63], [5, 41], [4, 13], [4, 36], [44, 59], [44, 57], [40, 55], [4, 33], [6, 34], [4, 34], [44, 58], [44, 47], [5, 40], [4, 9], [44, 46], [5, 42], [5, 43], [4, 15], [5, 47], [4, 12], [4, 39], [46, 61], [4, 8], [44, 43], [48, 40], [6, 32], [6, 12], [6, 11], [44, 41], [44, 32], [5, 46], [44, 36], [69, 28], [5, 44], [6, 39], [43, 14], [13, 31], [43, 31], [6, 9], [6, 25], [4, 32], [52, 27], [44, 42], [8, 4], [40, 54], [48, 4], [40, 48], [44, 45], [6, 37], [4, 38], [43, 15], [6, 28], [6, 15], [46, 2], [44, 60], [6, 31], [43, 27], [6, 10], [50, 55], [3, 35], [50, 14], [46, 16], [6, 13], [7, 40], [53, 37], [55, 9], [44, 34], [52, 30], [48, 7], [50, 10], [48, 44], [5, 45], [46, 60], [46, 1], [48, 6], [44, 44], [68, 5], [44, 35], [50, 9], [46, 57], [55, 10], [57, 38], [29, 1], [48, 3], [53, 35], [52, 29], [52, 57], [44, 38], [5, 12], [48, 46], [40, 23], [51, 48], [46, 3], [48, 0], [22, 36], [4, 35], [53, 34], [48, 1], [46, 5], [50, 53], [52, 63], [54, 6], [52, 20], [47, 24], [46, 17], [46, 23], [6, 33], [46, 58], [43, 20], [42, 0], [53, 36], [52, 25], [36, 0], [54, 9], [43, 11], [40, 52], [52, 37], [46, 59], [50, 49], [48, 17], [4, 10], [52, 59], [52, 19], [52, 56], [44, 33], [52, 61], [45, 44], [46, 62], [46, 7], [14, 29], [40, 49], [52, 35], [46, 22], [52, 62], [46, 20], [46, 19], [52, 28], [51, 13], [40, 50], [53, 39], [51, 26], [51, 29], [50, 15], [52, 24], [42, 7], [46, 21], [57, 35], [44, 17], [52, 14], [43, 2], [52, 23], [68, 40], [46, 63], [53, 32], [44, 20], [53, 38], [44, 18], [48, 21], [43, 28], [50, 52], [55, 14], [52, 51], [50, 48], [57, 37], [7, 42], [50, 50], [4, 22], [46, 0], [40, 17], [52, 21], [43, 8], [52, 31], [43, 10], [46, 30], [48, 42], [6, 35], [14, 30], [48, 2], [48, 5], [57, 32], [35, 2], [50, 54], [52, 16], [52, 22], [47, 26], [44, 56], [44, 61], [42, 28], [52, 13], [54, 8], [50, 11], [52, 12], [55, 11], [52, 48], [35, 6], [45, 45], [52, 49], [54, 2], [40, 51], [48, 18], [51, 25], [54, 11], [3, 34], [62, 60], [55, 8], [57, 25], [44, 22], [55, 12], [59, 48], [52, 15], [52, 8], [31, 17], [51, 14], [7, 46], [54, 3], [26, 16], [49, 38], [45, 24], [51, 27], [54, 7], [51, 55], [52, 52], [45, 31], [58, 6], [5, 17], [51, 49], [52, 60], [40, 16], [52, 54], [51, 54], [38, 17], [51, 12], [54, 5], [45, 26], [47, 29], [50, 12], [51, 50], [7, 63], [57, 33], [9, 44], [41, 15], [54, 15], [52, 17], [45, 25], [31, 20], [9, 43], [5, 61], [7, 47], [53, 42], [44, 39], [52, 50], [47, 48], [54, 13], [50, 51], [13, 25], [4, 55], [44, 51], [71, 60], [75, 34], [55, 45], [51, 10], [54, 12], [50, 8], [42, 5], [45, 46], [44, 5], [46, 4], [65, 54], [54, 1], [41, 16], [36, 5], [43, 49], [54, 0], [53, 43], [39, 32], [42, 22], [48, 20], [46, 18], [50, 31], [59, 54], [50, 13], [45, 40], [73, 5], [53, 41], [52, 53], [65, 49], [55, 13], [15, 43], [52, 58], [36, 47], [41, 45], [55, 40], [46, 43], [43, 25], [48, 31], [42, 4], [51, 8], [40, 27]]


llama3_layer_head_id_8b = [[15, 30], [2, 21], [2, 22], [16, 20], [2, 23], [2, 20], [26, 15], [2, 12], [5, 8], [2, 25], [8, 1], [24, 27], [5, 11], [5, 9], [20, 14], [5, 10], [16, 23], [15, 1], [20, 1], [19, 3], [27, 5], [22, 14], [10, 14], [20, 13], [2, 26], [27, 6], [15, 28], [27, 7], [28, 15], [26, 13], [16, 1], [20, 15], [22, 12], [27, 4], [22, 15], [24, 24], [24, 23], [27, 20], [20, 26], [22, 13], [27, 23], [27, 22], [22, 29], [24, 25], [25, 5], [24, 22], [25, 13], [26, 12], [13, 6], [24, 20], [26, 14], [20, 25], [30, 21], [25, 7], [25, 6], [22, 28], [30, 29], [20, 2], [25, 12], [19, 0], [24, 18], [25, 4], [23, 22], [24, 16], [25, 14], [24, 17], [23, 5], [26, 29], [30, 30], [25, 15], [26, 31], [30, 11], [20, 23], [17, 29], [17, 24], [23, 20], [22, 0], [23, 13], [30, 12], [20, 24], [23, 6], [16, 3], [22, 2], [23, 12], [10, 13], [8, 0], [18, 28], [28, 27], [24, 26], [2, 14], [22, 1], [17, 26], [23, 14], [19, 2], [22, 31], [23, 25], [21, 11], [30, 15], [18, 8], [19, 23], [16, 19], [2, 13], [18, 4], [6, 31], [17, 21], [29, 22], [23, 27], [15, 2], [14, 22], [14, 18], [14, 20], [20, 20], [17, 27], [15, 3], [26, 2], [21, 8], [17, 31], [17, 25], [16, 25], [28, 13], [21, 31], [16, 8], [9, 27], [30, 14], [16, 21], [16, 2], [22, 8], [21, 26], [29, 21], [13, 27], [13, 4], [29, 20], [13, 18], [9, 31], [16, 0], [18, 9], [18, 30], [10, 12], [18, 5], [22, 30], [17, 28], [16, 26], [19, 13], [29, 23], [19, 14], [15, 29], [18, 22], [19, 12], [20, 27], [31, 21], [14, 13], [11, 5], [15, 13], [20, 9], [14, 31], [19, 9], [18, 29], [21, 25], [19, 1], [27, 21], [30, 3], [21, 1], [18, 20], [18, 18], [14, 29], [16, 27], [26, 30], [20, 0], [21, 14], [22, 10], [18, 16], [21, 30], [17, 23], [15, 21], [23, 7], [22, 11], [13, 17], [27, 16], [30, 13], [30, 20], [18, 21], [15, 14], [29, 9], [2, 27], [14, 30], [21, 29], [21, 2], [30, 26], [26, 27], [21, 22], [19, 27], [10, 31], [22, 27], [13, 5], [21, 24], [21, 21], [20, 10], [29, 11], [14, 5], [8, 3]]

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


class Ada_LlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    '''
    mask_head_id: Union[int, list] added for masked head
    '''

    def __init__(self, config: LlamaConfig,layer_idx: Optional[int] = None):
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

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
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
    
        '''
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # print("attn_weights: ", type(attn_weights), attn_weights.shape, attn_weights[0][0]) # [1, 32, 40, 40]


        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
        
        # print("attention_mask: ", type(attention_mask), attention_mask.shape, attention_mask) # [1, 32, 40, 40]
        '''


        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # print("value_states: ", type(value_states), value_states.shape) # [1, 32, 40, 128]
        attn_output = torch.matmul(attn_weights, value_states)
        # print("attn_output: ", type(attn_output), attn_output.shape) # [1, 32, 40, 128]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        # print("attn_output: ", type(attn_output), attn_output.shape) # [1, 40, 32, 128]
        # print("attn device: ", attn_output.device)
        #### added by Zhuoyan
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

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # print("attn_output: ", type(attn_output), attn_output.shape) # [1, 40, 4096]

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        # print("attn_output: ", type(attn_output), attn_output.shape)
        # print("zhuoyan==============================")
        # print("output_attentions: ", output_attentions)
        # print("attn_output: ", type(attn_output), attn_output.shape, attn_output)
        # assert False
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "eager_mask": Ada_LlamaAttention, 
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}

class Ada_LlamaDecoderLayer(LlamaDecoderLayer):
    '''
    mask_head_id: Union[int, list] added for masked head
    '''

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)   
        self.hidden_size = config.hidden_size
        # config._attn_implementation = "eager_mask"
        # print("inside attn inple: ", config._attn_implementation)
        # self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn = LLAMA_ATTENTION_CLASSES["eager_mask"](config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        # self.mlp = Ada_LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    
    def set_mask_head_id(self, mask_head_id: Union[int, list]):
        self.self_attn.set_mask_head_id(mask_head_id = mask_head_id)
    
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


# purely debug
class LlamaDecoderLayer2(LlamaDecoderLayer):
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)


class Ada_LlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Ada_LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # print("llamadecoder layer 2")
        # self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # self.layers = nn.ModuleList([LlamaDecoderLayer2(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # self._use_sdpa = config._attn_implementation == "sdpa"
        # self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.mask_layer_head_id = mask_layer_head_id  # Store the mask_layer_head_id
        # Set masked head ids for each specified Ada_LlamaDecoderLayer
        for layer_id, mask_head_id in mask_layer_head_id:
            if layer_id < len(self.layers):
                self.layers[layer_id].set_mask_head_id(mask_head_id)
        

class Ada_LlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Ada_LlamaModel(config)
        # self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def set_mask_layer_head_id(self, mask_layer_head_id: List[List[int]]):
        self.model.set_mask_layer_head_id(mask_layer_head_id)
