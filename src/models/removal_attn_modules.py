import torch.nn.functional as F
from typing import Optional, Tuple, Union
import torch
import math
from transformers.cache_utils import Cache
from torch import nn
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.olmo.modeling_olmo import OlmoAttention
from transformers.models.falcon.modeling_falcon import FalconAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention


class AdaGPT2Attention(GPT2Attention):
    """GPT2 attention with head masking capability"""

    def __init__(
        self,
        original_attention: GPT2Attention,
        mask_layer_idx: int,
        head_mask: torch.Tensor,
    ):
        super().__init__(
            original_attention.config, layer_idx=original_attention.layer_idx
        )
        for name, value in vars(original_attention).items():
            if name != "forward":
                setattr(self, name, value)

        self.mask_layer_idx = mask_layer_idx
        self.head_mask = head_mask

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

        # attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()

        # added by zhuoyan
        mask = torch.ones_like(attn_output, device=attn_output.device)
        mask_heads_ids = self.head_mask[self.mask_layer_idx] == 0
        mask[:, :, mask_heads_ids, :] = 0
        attn_output_masked = attn_output * mask
        attn_output = attn_output_masked
        # end of added by zhuoyan

        new_shape = attn_output.size()[:-2] + (self.num_heads * self.head_dim,)
        attn_output = attn_output.view(new_shape)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class AdaGemmaAttention(GemmaAttention):
    """Gemma attention with head masking capability"""

    def __init__(
        self,
        original_attention: GemmaAttention,
        mask_layer_idx: int,
        head_mask: torch.Tensor,
    ):
        super().__init__(
            original_attention.config, layer_idx=original_attention.layer_idx
        )
        for name, value in vars(original_attention).items():
            if name != "forward":
                setattr(self, name, value)

        self.mask_layer_idx = mask_layer_idx
        self.head_mask = head_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.mask_layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # added by zhuoyan
        mask = torch.ones_like(attn_output, device=attn_output.device)
        mask_heads_ids = self.head_mask[self.mask_layer_idx] == 0
        mask[:, :, mask_heads_ids, :] = 0
        attn_output_masked = attn_output * mask
        attn_output = attn_output_masked
        # end of added by zhuoyan

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class AdaGemma2Attention(Gemma2Attention):
    """Gemma attention with head masking capability"""

    def __init__(
        self,
        original_attention: Gemma2Attention,
        mask_layer_idx: int,
        head_mask: torch.Tensor,
    ):
        super().__init__(
            original_attention.config, layer_idx=original_attention.layer_idx
        )
        for name, value in vars(original_attention).items():
            if name != "forward":
                setattr(self, name, value)

        self.mask_layer_idx = mask_layer_idx
        self.head_mask = head_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        if self.config.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # added by zhuoyan
        mask = torch.ones_like(attn_output, device=attn_output.device)
        mask_heads_ids = self.head_mask[self.mask_layer_idx] == 0
        mask[:, :, mask_heads_ids, :] = 0
        attn_output_masked = attn_output * mask
        attn_output = attn_output_masked
        # end of added by zhuoyan

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class AdaLlamaAttention(LlamaAttention):
    """Llama attention with head masking capability"""

    def __init__(
        self,
        original_attention: GemmaAttention,
        mask_layer_idx: int,
        head_mask: torch.Tensor,
    ):
        super().__init__(
            original_attention.config, layer_idx=original_attention.layer_idx
        )
        for name, value in vars(original_attention).items():
            if name != "forward":
                setattr(self, name, value)

        self.mask_layer_idx = mask_layer_idx
        self.head_mask = head_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # added by zhuoyan
        mask = torch.ones_like(attn_output, device=attn_output.device)
        mask_heads_ids = self.head_mask[self.mask_layer_idx] == 0
        mask[:, :, mask_heads_ids, :] = 0
        attn_output_masked = attn_output * mask
        attn_output = attn_output_masked
        # end of added by zhuoyan

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class AdaGPTNeoXAttention(GPTNeoXAttention):
    """GPTNeoX attention with head masking capability"""

    def __init__(
        self,
        original_attention: GPTNeoXAttention,
        mask_layer_idx: int,
        head_mask: torch.Tensor,
    ):
        super().__init__(
            original_attention.config, layer_idx=original_attention.layer_idx
        )
        for name, value in vars(original_attention).items():
            if name != "forward":
                setattr(self, name, value)

        self.mask_layer_idx = mask_layer_idx
        self.head_mask = head_mask

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
    ):
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

        # Compute attention
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        # added by zhuoyan
        mask = torch.ones_like(attn_output, device=attn_output.device)
        mask_heads_ids = self.head_mask[self.mask_layer_idx] == 0
        mask[:, mask_heads_ids, :, :] = 0
        attn_output_masked = attn_output * mask
        attn_output = attn_output_masked
        # end of added by zhuoyan

        # Reshape outputs
        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.head_size
        )
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class AdaMistralAttention(MistralAttention):
    """Mistral attention with head masking capability"""

    def __init__(
        self,
        original_attention: GPTNeoXAttention,
        mask_layer_idx: int,
        head_mask: torch.Tensor,
    ):
        super().__init__(
            original_attention.config, layer_idx=original_attention.layer_idx
        )
        for name, value in vars(original_attention).items():
            if name != "forward":
                setattr(self, name, value)

        self.mask_layer_idx = mask_layer_idx
        self.head_mask = head_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # added by zhuoyan
        mask = torch.ones_like(attn_output, device=attn_output.device)
        mask_heads_ids = self.head_mask[self.mask_layer_idx] == 0
        mask[:, :, mask_heads_ids, :] = 0
        attn_output_masked = attn_output * mask
        attn_output = attn_output_masked
        # end of added by zhuoyan

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class AdaFalconAttention(FalconAttention):
    """Falcon attention with head masking capability"""

    def __init__(
        self,
        original_attention: GPTNeoXAttention,
        mask_layer_idx: int,
        head_mask: torch.Tensor,
    ):
        super().__init__(
            original_attention.config, layer_idx=original_attention.layer_idx
        )
        for name, value in vars(original_attention).items():
            if name != "forward":
                setattr(self, name, value)

        self.mask_layer_idx = mask_layer_idx
        self.head_mask = head_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Cache] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
    ):
        fused_qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = (
            self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        )
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(
            batch_size, self.num_heads, query_length, self.head_dim
        )
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size, num_kv_heads, query_length, self.head_dim
        )
        value_layer = value_layer.transpose(1, 2).reshape(
            batch_size, num_kv_heads, query_length, self.head_dim
        )

        if alibi is None:
            if position_embeddings is None:
                # logger.warning_once(
                #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                #     "removed and `position_embeddings` will be mandatory."
                # )
                cos, sin = self.rotary_emb(value_layer, position_ids)
            else:
                cos, sin = position_embeddings
            query_layer, key_layer = apply_rotary_pos_emb(
                query_layer, key_layer, cos, sin
            )

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            if alibi is None:
                cache_kwargs.update({"sin": sin, "cos": cos})
            key_layer, value_layer = layer_past.update(
                key_layer, value_layer, self.layer_idx, cache_kwargs
            )

        kv_length = key_layer.shape[-2]
        if (
            self._use_sdpa
            and query_layer.device.type == "cuda"
            and attention_mask is not None
        ):
            # For torch<=2.1.2, SDPA with memory-efficient backend is bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_layer.shape[-2]]

        if alibi is None:
            if self._use_sdpa and not output_attentions:
                # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
                # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
                # The query_length > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not
                # create a causal mask in case query_length == 1.
                is_causal = (
                    True
                    if self.is_causal and attention_mask is None and query_length > 1
                    else False
                )
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
                attention_scores = None
            else:
                attention_scores = query_layer @ key_layer.transpose(-1, -2)
                attention_scores /= math.sqrt(self.head_dim)

                attention_scores = F.softmax(
                    attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype
                )
                # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
                attn_output = attention_scores @ value_layer

            attn_output = attn_output.view(
                batch_size, self.num_heads, query_length, self.head_dim
            )
            attn_output = attn_output.permute(0, 2, 1, 3)

            # added by zhuoyan
            mask = torch.ones_like(attn_output, device=attn_output.device)
            mask_heads_ids = self.head_mask[self.mask_layer_idx] == 0
            mask[:, :, mask_heads_ids, :] = 0
            attn_output_masked = attn_output * mask
            attn_output = attn_output_masked
            # end of added by zhuoyan

            attn_output = attn_output.reshape(
                batch_size, query_length, self.num_heads * self.head_dim
            )

            attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, layer_past, attention_scores
            else:
                return attn_output, layer_past

        else:
            if self._use_sdpa and not output_attentions and head_mask is None:
                # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
                # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
                is_causal = (
                    True
                    if self.is_causal and attention_mask is None and query_length > 1
                    else False
                )
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_layer,
                    key_layer,
                    value_layer,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout.p if self.training else 0.0,
                    is_causal=is_causal,
                )
                attn_output = attn_output.transpose(1, 2)
                attn_output = attn_output.reshape(
                    batch_size, query_length, self.num_heads * self.head_dim
                )

                attn_output = self.dense(attn_output)
            else:
                matmul_result = query_layer @ key_layer.transpose(-1, -2)

                # change view to [batch_size, num_heads, q_length, kv_length]
                attention_scores = matmul_result.view(
                    batch_size, self.num_heads, query_length, kv_length
                )

                # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
                input_dtype = attention_scores.dtype
                # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
                if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                    attention_scores = attention_scores.to(torch.float32)

                attention_logits = attention_scores + alibi.view(
                    batch_size, self.num_heads, 1, -1
                )
                attention_logits *= self.inv_norm_factor
                attention_probs = F.softmax(
                    attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype
                )
                # [batch_size, num_heads, q_length, kv_length]
                attention_probs = self.attention_dropout(attention_probs)

                if head_mask is not None:
                    attention_probs = attention_probs * head_mask

                # change view [batch_size, num_heads, q_length, kv_length]
                attention_probs_reshaped = attention_probs.view(
                    batch_size, self.num_heads, query_length, kv_length
                )

                # matmul: [batch_size * num_heads, q_length, head_dim]
                attn_output = (attention_probs_reshaped @ value_layer).flatten(0, 1)

                # change view [batch_size, q_length, num_heads * head_dim]
                attn_output = self._merge_heads(attn_output)

                attn_output = self.dense(attn_output)

            if output_attentions:
                return attn_output, layer_past, attention_probs
            else:
                return attn_output, layer_past


class AdaOlmoAttention(OlmoAttention):
    """Olmo attention with head masking capability"""

    def __init__(
        self,
        original_attention: GPTNeoXAttention,
        mask_layer_idx: int,
        head_mask: torch.Tensor,
    ):
        super().__init__(
            original_attention.config, layer_idx=original_attention.layer_idx
        )
        for name, value in vars(original_attention).items():
            if name != "forward":
                setattr(self, name, value)

        self.mask_layer_idx = mask_layer_idx
        self.head_mask = head_mask

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

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # added by zhuoyan
        mask = torch.ones_like(attn_output, device=attn_output.device)
        mask_heads_ids = self.head_mask[self.mask_layer_idx] == 0
        mask[:, :, mask_heads_ids, :] = 0
        attn_output_masked = attn_output * mask
        attn_output = attn_output_masked
        # end of added by zhuoyan

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
