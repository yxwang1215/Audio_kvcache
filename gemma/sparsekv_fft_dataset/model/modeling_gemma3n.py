import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.auto import AutoModel
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nAudioConfig, Gemma3nConfig, Gemma3nTextConfig, Gemma3nVisionConfig

# Import SparseMM utilities
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
try:
    from sparsemm_utils import (
        init_sparsemm, 
        init_adakv, 
        init_pyramidkv, 
        init_snapkv,
        init_mixkv, 
        DynamicCacheSplitHeadFlatten, 
        repeat_kv
    )
    SPARSEM_AVAILABLE = True
except ImportError:
    SPARSEM_AVAILABLE = False
    init_sparsemm = None
    init_adakv = None
    init_pyramidkv = None
    init_snapkv = None
    init_mixkv = None
    print("[WARNING] sparsemm_utils not available, SparseMM features will be disabled")

# Import flash_attn_varlen_func for SparseMM decode phase
try:
    from flash_attn import flash_attn_varlen_func as sparsemm_flash_attn_varlen_func
    FLASH_ATTN_VARLEN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_VARLEN_AVAILABLE = False
    sparsemm_flash_attn_varlen_func = None
    print("[WARNING] flash_attn_varlen_func not available, SparseMM decode will use fallback")

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Gemma3n outputs, with hidden states and attentions.
    """
)
class Gemma3nModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None

    audio_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Gemma3n causal language model (or autoregressive) outputs.
    """
)
class Gemma3nCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None

    audio_hidden_states: Optional[torch.FloatTensor] = None


class Gemma3nRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float()) * self.weight.float()
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ==== Audio Encoder ====


class Gemma3nAudioRelativePositionEmbedding(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.channels = self.config.hidden_size
        self.head_dim = self.channels // self.num_heads
        self.max_backward = max(0, self.config.conf_attention_context_left - 1)
        self.max_forward = self.config.conf_attention_context_right

        self.pos_proj = nn.Linear(self.channels, self.num_heads * self.head_dim, bias=False)

        min_timescale = 1.0
        max_timescale = 1.0e4
        num_timescales = self.channels // 2
        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
        self.register_buffer(
            "inv_timescales",
            inv_timescales.float().unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _get_timing_signal_1d_pos(self, position: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        position = position.float().unsqueeze(-1)
        scaled_time = position * self.inv_timescales.to(device=position.device, dtype=torch.float32)
        timing_signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return timing_signal.type(dtype)

    def _relative_shift(
        self,
        term_bd_before_shift: torch.Tensor,
        batch_size: int,
        num_heads: int,
        num_query_blocks: int,
        query_block_size: int,
        key_context_size: int,
        max_span_plus_1: int,
    ) -> torch.Tensor:
        """Performs the relative shift.

        Args:
          term_bd_before_shift: Tensor of shape [B, N, U, W, F_span]. batch_size
            (B), num_heads (N), num_query_blocks (U), query_block_size (W),
            key_context_size (C = W+L+R), max_span_plus_1 (F_span = L+R+1).

        Returns:
          Tensor of shape [B, N, U, W, C].
        """
        # term_bd_before_shift shape: [B, N, U, W, F_span]
        # Target shape after shift:  [B, N, U, W, C]

        # Padding amount for the last dimension (F_span) to become (C + 1)
        # C = key_context_size
        # F_span = max_span_plus_1
        pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1

        # PyTorch F.pad expects (pad_left, pad_right, pad_top, pad_bottom ...)
        # We only pad the last dimension on the right.
        padding_tuple = (0, pad_amount_last_dim)

        term_bd_padded = nn.functional.pad(term_bd_before_shift, padding_tuple)
        # Shape after pad: [B, N, U, W, C+1]

        # Reshape for slicing (emulating JAX's behavior)
        # [B, N, U, W * (C+1)]
        term_bd_reshaped = term_bd_padded.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size * (key_context_size + 1),
            )
        )

        # Slice to effective [B, N, U, W * C]
        term_bd_sliced = term_bd_reshaped[:, :, :, : query_block_size * key_context_size]

        # Reshape back to [B, N, U, W, C]
        term_bd_shifted = term_bd_sliced.reshape(
            (
                batch_size,
                num_heads,
                num_query_blocks,
                query_block_size,
                key_context_size,
            )
        )
        return term_bd_shifted

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # queries: [B, U, W, N, H] (batch, num_query_blocks, query_block_size, num_heads, head_dim)
        # keys:    [B, U, C, N, H] (batch, num_query_blocks, key_context_size, num_heads, head_dim)
        # C = W + L + R (key_context_size)
        # F_span = L + R + 1 (max_span + 1)

        batch_size, num_query_blocks, query_block_size, num_heads, head_dim = queries.shape
        _, _, key_context_size, _, _ = keys.shape

        # Relative positions for sinusoidal embeddings: [L, L-1, ..., -R]
        # Length is L+R+1 = self.max_span + 1
        pos_indices = torch.arange(self.max_backward, -self.max_forward - 1, -1, device=queries.device).unsqueeze(
            0
        )  # Shape [1, F_span]

        max_span_plus_1 = pos_indices.shape[1]  # F_span

        sin_emb_timing_signal = self._get_timing_signal_1d_pos(
            pos_indices, dtype=queries.dtype
        )  # Shape [1, F_span, self.channels]

        # Project sinusoidal embeddings: [1, F_span, self.channels] -> [1, F_span, N*H]
        projected_sin_emb = self.pos_proj(sin_emb_timing_signal)
        # Reshape to [1, F_span, N, H] then squeeze to [F_span, N, H]
        sin_emb = projected_sin_emb.reshape(1, max_span_plus_1, self.num_heads, self.head_dim).squeeze(
            0
        )  # Shape [F, N, H]

        # term_ac: Query-Key content interaction
        # queries: [B, U, W, N, H] -> permute to [B, N, U, W, H] for matmul
        # keys:    [B, U, C, N, H] -> permute to [B, N, U, H, C] for matmul
        queries_p = queries.permute(0, 3, 1, 2, 4)  # [B, N, U, W, H]
        keys_p_t = keys.permute(0, 3, 1, 4, 2)  # [B, N, U, H, C]
        term_ac = torch.matmul(queries_p, keys_p_t)  # [B, N, U, W, C]

        # term_bd: Query-Position interaction
        # Original einsum: term_bd_unshifed = torch.einsum('buwnh,fnh->bnuwf', queries, sin_emb)
        # queries shape: [B, U, W, N, H]
        # sin_emb shape: [F, N, H]
        # Target output shape: [B, N, U, W, F]

        # Permute queries to [B, N, U, W, H] for easier broadcasting with sin_emb
        q_permuted = queries.permute(0, 3, 1, 2, 4)

        # Permute sin_emb to [N, H, F] to prepare for matmul
        # sin_emb original is [F, N, H]
        s_permuted = sin_emb.permute(1, 2, 0)  # Shape: [N, H, F]

        # Reshape queries for matmul: [B, N, U*W, H]
        q_reshaped = q_permuted.reshape(batch_size, num_heads, num_query_blocks * query_block_size, head_dim)

        # Perform matmul: [B, N, U*W, H] @ [N, H, F]
        # s_permuted ([N, H, F]) will be broadcast to [B, N, H, F]
        # Result: [B, N, U*W, F]
        term_bd_unshifed_matmul = torch.matmul(q_reshaped, s_permuted)

        # Reshape to target [B, N, U, W, F]
        term_bd_unshifed = term_bd_unshifed_matmul.reshape(
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        )

        # Apply relative shift to term_bd_unshifed
        term_bd_shifted = self._relative_shift(
            term_bd_unshifed,
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
            max_span_plus_1,
        )  # Shape [B, N, U, W, C]

        return term_ac + term_bd_shifted


class Gemma3nAudioAttention(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.num_heads = self.config.conf_num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        self.chunk_size = self.config.conf_attention_chunk_size
        self.max_future_horizon = self.config.conf_attention_context_right
        self.max_past_horizon = max(0, self.config.conf_attention_context_left - 1)
        self.attention_logits_soft_cap = self.config.conf_attention_logit_cap
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon

        self.relative_position_embedding = Gemma3nAudioRelativePositionEmbedding(config)
        self.per_dim_scale = nn.Parameter(torch.zeros((self.head_dim,)))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        q_scale = self.head_dim**-0.5
        r_softplus_0 = 1.0 / torch.nn.functional.softplus(torch.tensor(0.0))
        self.register_buffer("q_scale", (q_scale * r_softplus_0).clone().detach(), persistent=False)

        lower_causal_mask = torch.tril(
            torch.ones((self.context_size, self.chunk_size), dtype=torch.bool),
            diagonal=0,
        ).T
        upper_causal_mask = torch.tril(
            torch.ones((self.chunk_size, self.context_size), dtype=torch.bool),
            diagonal=self.max_past_horizon + self.max_future_horizon,
        )
        local_causal_valid_mask = torch.ones((self.chunk_size, self.context_size), dtype=torch.bool)
        local_causal_valid_mask = local_causal_valid_mask * lower_causal_mask * upper_causal_mask
        self.register_buffer("local_causal_valid_mask", local_causal_valid_mask, persistent=False)

        self.register_buffer(
            "softcap",
            torch.tensor(self.attention_logits_soft_cap).float(),
            persistent=False,
        )

    def _pad_dim1(self, x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
        batch, _, *tail_shape = x.shape
        left = x.new_zeros((batch, pad_left, *tail_shape))
        right = x.new_zeros((batch, pad_right, *tail_shape))
        x = torch.cat([left, x, right], dim=1)
        return x

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Turns a sequence to non overlapping blocks.

        Args:
            hidden_states: a tensor of [batch, time, ...].

        Returns:
            A tensor of [batch, num_blocks, block_size, ...], with necessary
            paddings,
            where output[:, i, ...] are x[:, i*block_size:(i+1)*block_size, ...].
        """
        shape = hidden_states.shape
        b, t = shape[:2]
        num_blocks = (t + self.chunk_size - 1) // self.chunk_size

        if (padding_len := num_blocks * self.chunk_size - t) > 0:
            hidden_states = self._pad_dim1(hidden_states, 0, padding_len)

        permute_dims = (b, num_blocks, self.chunk_size) + shape[2:]
        hidden_states = hidden_states.reshape(permute_dims).contiguous()
        return hidden_states

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extracts temporal context for every block.

        Args:
            hidden_states: a tensor of [batch, time, ...].

        Returns:
            A tensor of [batch, num_blocks, context_size, ...], with necessary
            paddings,
            where context_size = block_size + left_context + right_context,
            and output[:, i, ...] are x[:, start-left_context:end+right_context,
            ...],
            start = i * block_size, end = (i + 1) * block_size.
        """
        pad_left = self.max_past_horizon
        # The JAX equivalent padding for signal.frame with pad_mode='valid' is
        # (left_context, right_context + block_size - 1) on the time dimension.
        # PyTorch's _pad_dim1 applies padding symmetrically if only one value is given,
        # or (pad_dim_start, pad_dim_end) if two are given.
        # Our _pad_dim1(x, pad_left, pad_right) pads dim -2 (time for [B,T,N,H])
        # or dim 1 (time for [B,T]).
        # The current pad_right calculation matches the JAX effective padding.
        pad_right = self.max_future_horizon + self.chunk_size - 1
        hidden_states = self._pad_dim1(hidden_states, pad_left, pad_right)

        frame_len = self.context_size
        frame_step = self.chunk_size

        # Directly use unfold without the subframe_factor logic
        # x.unfold(dimension, size, step)
        # dimension=1 (time dimension, assuming x is [B, T_padded, ...])
        # size=frame_len (context_size)
        # step=frame_step (chunk_size)
        x_unfolded = hidden_states.unfold(dimension=1, size=frame_len, step=frame_step)

        # If x was [B, T_padded], x_unfolded is [B, num_blocks, frame_len]
        # If x was [B, T_padded, N, H], x_unfolded is [B, num_blocks, N, H, frame_len]
        # We want to match JAX's typical output for such operations which might be
        # [B, num_blocks, frame_len, N, H] if N, H are present.
        # The relative_position_embedding expects keys as [B, U, C, N, H].
        # If x_unfolded is [B, U, N, H, C(frame_len)], we need to move C.
        if hidden_states.ndim > 2 and x_unfolded.ndim > 3:  # Check if inner dimensions (like N, H) exist
            # Current shape after unfold for [B, T_pad, N, H] is [B, U, N, H, C]
            # Target shape for keys in RPE: [B, U, C, N, H]
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)

        return x_unfolded.contiguous()

    def forward(self, hidden_states: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        # sl.Dense uses jax.numpy.einsum("...a,abcd->...bcd") and jax.numpy.select()
        qkv_shape = (*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        query_states = self.q_proj(hidden_states).reshape(qkv_shape).contiguous()
        key_states = self.k_proj(hidden_states).reshape(qkv_shape).contiguous()
        value_states = self.v_proj(hidden_states).reshape(qkv_shape).contiguous()

        per_dim_scale_sp = torch.nn.functional.softplus(self.per_dim_scale)

        broadcast_shape = (1, 1, 1, self.head_dim)
        per_dim_scale_sp_broadcast = per_dim_scale_sp.view(broadcast_shape)
        query_states = query_states * self.q_scale * per_dim_scale_sp_broadcast

        batch_size, q_time = query_states.shape[:2]

        query_blocks = self._convert_to_block(query_states)
        key_blocks = self._extract_block_context(key_states)
        value_blocks = self._extract_block_context(value_states)
        num_query_blocks = query_blocks.shape[1]

        # 1. Create a mask indicating originally valid positions.
        original_valid_mask = ~mask  # True for valid, False for padded

        # 2. Extract blocks from this validity mask.
        extracted_valid_mask_blocks = self._extract_block_context(original_valid_mask)

        # If subframe_factor was used in _extract_block_context for a [B, T] input mask,
        # the shape might be [B, U, C/SF, SF]. Reshape to [B, U, C].
        # batch_size and num_query_blocks are known from query_blocks.
        # self.context_size is C.
        if (
            extracted_valid_mask_blocks.ndim == 4
            and extracted_valid_mask_blocks.shape[2] * extracted_valid_mask_blocks.shape[3] == self.context_size
        ):
            extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
                batch_size, num_query_blocks, self.context_size
            )
        # After potential reshape, ensure it's [B, U, C] if it was from a [B,T] mask.
        # This assertion might be too strict if _extract_block_context handles higher-rank inputs differently,
        # but for the mask case, this should hold.
        if extracted_valid_mask_blocks.shape != (
            batch_size,
            num_query_blocks,
            self.context_size,
        ):
            raise ValueError(
                "Shape of extracted_valid_mask_blocks"
                f" {extracted_valid_mask_blocks.shape} is not ({batch_size},"
                f" {num_query_blocks}, {self.context_size}) after potential reshape."
            )

        # 3. Expand dimensions for broadcasting with logits and causal mask.
        # Target shape for broadcasting with logits [B,N,U,W,C]
        # extracted_valid_mask_blocks to [B, 1, U, 1, C]
        condition_from_input_validity = extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2)

        # self.local_causal_valid_mask is [W, C], True where allowed by local window.
        # Expand to [1, 1, 1, W, C]
        condition_from_causality = self.local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # 4. Combine the two conditions.
        # final_condition will be True where a key is *both* originally valid *and* causally accessible.
        # Broadcasts to [B, 1, U, W, C]
        final_condition_for_where = torch.logical_and(
            condition_from_input_validity,
            condition_from_causality.to(condition_from_input_validity.device),  # Ensure same device
        )

        # Embed queries and keys
        logits = self.relative_position_embedding(query_blocks, key_blocks)

        # Apply attention logit softcap
        # Ensure softcap is on the same device as logits
        softcap_val = self.softcap.to(logits.device)
        logits = logits / softcap_val
        logits = torch.tanh(logits)
        logits = logits * softcap_val

        # Apply the combined mask.
        # final_condition_for_where will broadcast with logits [B,N,U,W,C]
        logits = torch.where(final_condition_for_where, logits, torch.finfo(logits.dtype).min)
        probabilities = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=value_blocks.dtype)

        # context_vectors is adapted from jax.numpy.einsum("BNuwc,BucNH->BuwNH", ...)
        b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
        h_dim = value_blocks.shape[-1]
        prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
        v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
        result_bmm = torch.bmm(prob_bun, v_bun)
        context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(0, 1, 3, 2, 4)
        context_vectors = context_vectors.reshape(
            (
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            )
        )
        context_vectors = context_vectors[:, :q_time]

        return context_vectors


class Gemma3nAudioCumulativeGroupNorm(nn.Module):
    """Applies Group Normalization cumulatively over the time dimension.

    This layer normalizes the input by calculating the mean and variance
    cumulatively over the time dimension (dim 1). The statistics are computed
    over all feature dimensions (specified by `feature_dims` and `num_channels`)
    for elements marked as valid by the optional `mask`.

    If a `mask` is provided (True for valid, False for invalid/padded),
    invalid time steps do not contribute to the statistics calculation, and
    their corresponding output values are zeroed out.

    Scale and bias, if enabled, are applied per-channel (last dimension).
    This behavior is similar to JAX's `GroupNormalization` with `num_groups=1`
    and `cumulative=True`.
    """

    def __init__(
        self,
        num_channels: int,  # Number of channels (size of the last dimension)
        feature_dims: Sequence[int],  # Sizes of non-channel feature dimensions, e.g., (H, W) for input [B,T,H,W,C]
        eps: float = 1e-3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.feature_dims = tuple(feature_dims)
        self.eps = eps

        # Scale parameter depends only on the channel dimension
        self.weight = nn.Parameter(torch.ones(num_channels))

        # Axes for normalization: all dimensions except Batch (0) and Time (1).
        # For input [B, T, *feature_dims, C], these are dims from 2 onwards.
        self.reduction_axes = tuple(range(2, 2 + len(self.feature_dims) + 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Applies cumulative group norm, optionally using a mask.

        Args:
          hidden_states: Input tensor, shape [B, T, *feature_dims, C].

        Returns:
          Normalized tensor with the same shape as x.
        """
        expected_input_suffix = self.feature_dims + (self.num_channels,)
        if hidden_states.shape[2:] != expected_input_suffix:
            raise ValueError(
                f"Input tensor shape suffix {hidden_states.shape[2:]} does not match expected"
                f" suffix (feature_dims + num_channels) {expected_input_suffix}"
            )

        input_dtype = hidden_states.dtype
        # Calculations are performed in float32 for numerical stability.
        calc_dtype = torch.float32
        x_calc = hidden_states.to(calc_dtype)

        # Prepare a broadcastable mask (`mask_calc`).
        # If no mask is provided, treat all elements as valid
        # (mask_calc is all ones).
        # Otherwise, expand the [B, T] mask to [B, T, 1, ..., 1] for broadcasting.
        mask_calc = torch.ones_like(x_calc, dtype=calc_dtype)

        # Cumulative Statistics Calculation
        # 1. Sum of values over reduction axes at each time step.
        sum_values_at_t = torch.sum(x_calc, dim=self.reduction_axes, keepdim=True)
        # 2. Cumulative sum of values over time.
        cum_sum_values = torch.cumsum(sum_values_at_t, dim=1)

        # 3. Count of valid elements in the normalization group at each time step.
        #    (A "group" here consists of all features at a given Batch, Time).
        elements_in_group_at_t = torch.sum(mask_calc, dim=self.reduction_axes, keepdim=True)
        # 4. Cumulative count of valid elements over time.
        cum_count_elements = torch.cumsum(elements_in_group_at_t, dim=1)
        # Avoid division by zero if all preceding elements were masked.
        safe_cum_count_elements = torch.clamp(cum_count_elements, min=1.0)

        # 5. Cumulative mean.
        cum_mean = cum_sum_values / safe_cum_count_elements

        # 6. Sum of squared differences from the cumulative mean.
        #    Only sum for valid elements: (x_calc - cum_mean)^2 * mask_calc.
        #    Using x_calc here for the difference, as cum_mean already accounts for masking.
        squared_diff_from_mean = (x_calc - cum_mean).pow(2)
        sum_sq_diff_at_t = torch.sum(squared_diff_from_mean, dim=self.reduction_axes, keepdim=True)

        # 7. Cumulative sum of squared differences over time.
        cum_sum_sq_diff = torch.cumsum(sum_sq_diff_at_t, dim=1)

        # 8. Cumulative variance.
        cum_variance = cum_sum_sq_diff / safe_cum_count_elements

        # Normalize the input using the calculated cumulative statistics:
        # (x - E[x]) / sqrt(Var[x] + eps)
        normalized_x = (x_calc - cum_mean) * torch.rsqrt(cum_variance + self.eps)

        # Apply affine transformation (scale and bias) if enabled.
        # Scale and bias are applied per-channel (last dimension).
        scale = self.weight.to(calc_dtype)
        # Reshape for broadcasting: [C] -> [1, ..., 1, C]
        scale_view_shape = [1] * (hidden_states.dim() - 1) + [self.num_channels]
        normalized_x = normalized_x * scale.view(scale_view_shape)

        # Zero out outputs for time steps that were originally masked (where mask_calc is 0).
        # This ensures padded/invalid positions in the input result in zero output.
        final_output = normalized_x * mask_calc

        return final_output.to(input_dtype)


class Gemma3nAudioSSCPConvBlock(nn.Module):
    """A single convolution block for the SubSampleConvProjection.

    This block consists of a 2D convolution, followed by CumulativeGroupNorm,
    and a ReLU activation. It handles manual padding for the convolution.
    """

    def __init__(
        self,
        config: Gemma3nAudioConfig,
        idx: int,
        input_freq_dim: int,  # Changed from input_spatial_dim
        manual_padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        super().__init__()
        self.config = config
        self.manual_padding = manual_padding

        # in_channels is 1 for the first block, or C_out from previous block's conv
        in_channels = 1 if idx == 0 else self.config.sscp_conv_channel_size[idx - 1]
        out_channels = self.config.sscp_conv_channel_size[idx]
        kernel_h, kernel_w = self.config.sscp_conv_kernel_size[idx]
        stride_h, stride_w = self.config.sscp_conv_stride_size[idx]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(
                kernel_h,
                kernel_w,
            ),  # Kernel (kH, kW) operates on (Time, Freq_dim)
            stride=(stride_h, stride_w),
            padding=(0, 0),  # Manual padding is used
            bias=False,
        )

        # Calculate output frequency dimension (f_out_conv) after this convolution.
        # input_freq_dim is the unpadded width (feature dimension).
        # self.manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        f_in_padded = input_freq_dim + self.manual_padding[0] + self.manual_padding[1]
        f_out_conv = (f_in_padded - kernel_w) // stride_w + 1

        self.norm = Gemma3nAudioCumulativeGroupNorm(
            num_channels=out_channels,  # Channels of the conv output
            feature_dims=(f_out_conv,),  # The frequency dimension size after conv
            eps=self.config.sscp_conv_group_norm_eps,
        )

        self.activation = nn.ReLU()

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        # Input audio_encodings is [B, C_in, T_in, F_in] (e.g., C_in=1)
        # manual_padding is (pad_F_left, pad_F_right, pad_T_top, pad_T_bottom)
        # F.pad applies to last two dims: F_in then T_in
        audio_encodings_padded = F.pad(audio_encodings, self.manual_padding, mode="constant", value=0.0).to(
            self.conv.weight.dtype
        )
        # Expected padded shape for F_in, k_w=3, pad_F=(1,1) -> F_padded = F_in+2
        # Expected padded shape for T_in, k_h=3, pad_T=(0,2) -> T_padded = T_in+2
        audio_encodings_conv = self.conv(audio_encodings_padded)
        # Expected conv output shape: [B, C_out, T_out, F_out]
        # Input to norm is [B, T_out, F_out, C_out]
        x_for_norm = audio_encodings_conv.permute(0, 2, 3, 1).contiguous()
        x_normed = self.norm(x_for_norm)
        # Output of norm is [B, T_out, F_out, C_out], permute back to [B, C_out, T_out, F_out]
        audio_encodings_normed = x_normed.permute(0, 3, 1, 2).contiguous()
        return self.activation(audio_encodings_normed)


class Gemma3nAudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        current_f_for_block_input = config.input_feat_size  # Start with original feature dim
        calculated_block_padding = []
        calculated_f_out_dims = []  # Tracking frequency dimension output sizes

        for i in range(2):  # Assuming 2 conv layers as per sscp_conv_... arrays
            kernel_h, kernel_w = config.sscp_conv_kernel_size[i]
            stride_h, stride_w = config.sscp_conv_stride_size[i]

            # Padding for Time (Height for Conv2d) - REVERSE_CAUSAL like
            # JAX 'reverse_causal' padding is (0, kernel_size - 1)
            pad_t_top = 0
            pad_t_bottom = kernel_h - 1

            # Frequency Padding (Width for Conv2d)
            # Based on JAX effective padding (1,1) for F_in=10, K_w=3, S_w=2
            # and the successful test configuration.
            # If kernel/stride/input_freq for frequency changes, this might need re-evaluation
            # to match generic JAX 'SAME' behavior if it differs.
            pad_f_left = 1
            pad_f_right = 1

            manual_padding_tuple = (
                pad_f_left,
                pad_f_right,
                pad_t_top,
                pad_t_bottom,
            )
            calculated_block_padding.append(manual_padding_tuple)

            # Calculate output frequency dimension after this convolution
            # This uses the actual padding applied and kernel/stride.
            f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right
            f_out_after_conv = (f_in_padded - kernel_w) // stride_w + 1  # Assuming dilation_w = 1
            calculated_f_out_dims.append(f_out_after_conv)
            current_f_for_block_input = f_out_after_conv

        self.conv_0 = Gemma3nAudioSSCPConvBlock(
            idx=0,
            input_freq_dim=config.input_feat_size,  # Pass original feature dim
            config=config,
            manual_padding=calculated_block_padding[0],
        )
        self.conv_1 = Gemma3nAudioSSCPConvBlock(
            idx=1,
            input_freq_dim=calculated_f_out_dims[0],  # Output freq dim from conv_0
            config=config,
            manual_padding=calculated_block_padding[1],
        )
        final_c_out = config.sscp_conv_channel_size[-1]
        final_f_out = calculated_f_out_dims[-1]  # Final frequency dimension
        self.input_proj_in_features = final_c_out * final_f_out
        self.input_proj_linear = nn.Linear(self.input_proj_in_features, self.config.hidden_size, bias=False)

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        # audio_encodings is [B, T, F_in]
        # Reshape to [B, 1, T, F_in] (Batch, Channels=1, Height=Time, Width=F_in)
        audio_encodings_reshaped = audio_encodings.unsqueeze(1)
        x = self.conv_0(audio_encodings_reshaped)
        x = self.conv_1(x)
        # x from conv_1 is [B, C_out_1, T_out_1, F_out_1]
        b, c_out, t_out, f_out = x.shape
        # Permute to [B, T_out_1, F_out_1, C_out_1] then flatten F_out_1 and C_out_1
        x_permuted = x.permute(0, 2, 3, 1).contiguous()
        output_flattened = x_permuted.view(b, t_out, f_out * c_out)
        output = self.input_proj_linear(output_flattened)
        return output


class Gemma3nAudioConformerAttention(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config
        self.post_in_features = self.config.hidden_size
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.pre_attn_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.attn = Gemma3nAudioAttention(config)
        self.post = nn.Linear(self.post_in_features, self.config.hidden_size, bias=False)
        self.post_norm = Gemma3nRMSNorm(self.config.hidden_size)

    def forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
        audio_encodings_input_to_attn = audio_encodings
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings_norm = self.pre_attn_norm(audio_encodings)
        # Output of self.attn is [B, T, NumHeads, HeadDim]
        audio_encodings_attn_out = self.attn(audio_encodings_norm, audio_mel_mask)

        # Reshape from [B, T, NumHeads, HeadDim] to [B, T, NumHeads * HeadDim]
        # NumHeads * HeadDim = hidden_size
        b, t, num_heads, head_dim = audio_encodings_attn_out.shape
        audio_encodings_reshaped = audio_encodings_attn_out.reshape(b, t, num_heads * head_dim)

        audio_encodings = self.post(audio_encodings_reshaped)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        return audio_encodings_input_to_attn + self.post_norm(audio_encodings)


class Gemma3nAudioConformerFeedForward(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)

        self.pre_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.ffw_layer_1 = nn.Linear(self.config.hidden_size, self.config.hidden_size * 4, bias=False)
        self.ffw_layer_2 = nn.Linear(self.config.hidden_size * 4, self.config.hidden_size, bias=False)
        self.post_layer_norm = Gemma3nRMSNorm(self.config.hidden_size)
        self.post_layer_scale = torch.tensor(self.config.conf_residual_weight)

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        residual = audio_encodings
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings: torch.Tensor = self.ffw_layer_1(audio_encodings)
        audio_encodings = nn.functional.silu(audio_encodings)
        audio_encodings: torch.Tensor = self.ffw_layer_2(audio_encodings)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings = self.post_layer_norm(audio_encodings)
        return residual + (audio_encodings * self.post_layer_scale)


class Gemma3nAudioConformerLightConv1d(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.pre_layer_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.linear_start = nn.Linear(self.config.hidden_size, self.config.hidden_size * 2, bias=False)
        self.depthwise_conv1d = nn.Conv1d(
            in_channels=self.config.hidden_size,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.conf_conv_kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            groups=self.config.hidden_size,  # Depthwise
            bias=False,
        )
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.conv_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.linear_end = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

        self.causal_padding = self.config.conf_conv_kernel_size - 1

    def forward(self, audio_encodings: torch.Tensor) -> torch.Tensor:
        audio_encodings_residual = audio_encodings  # Save for residual connection

        audio_encodings = self.pre_layer_norm(audio_encodings)
        audio_encodings = self.linear_start(audio_encodings)
        audio_encodings = torch.nn.functional.glu(audio_encodings, dim=-1)
        # Permute for Conv1d: [B, T, D] -> [B, D, T]
        audio_encodings_permuted = audio_encodings.permute(0, 2, 1)
        # Apply manual causal padding
        audio_encodings_permuted_padded = F.pad(audio_encodings_permuted, (self.causal_padding, 0))
        audio_encodings = self.depthwise_conv1d(audio_encodings_permuted_padded)
        # Permute back: [B, D, T_out] -> [B, T_out, D]
        audio_encodings = audio_encodings.permute(0, 2, 1)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        audio_encodings = self.conv_norm(audio_encodings)
        audio_encodings = nn.functional.silu(audio_encodings)
        audio_encodings = self.linear_end(audio_encodings)
        output = audio_encodings + audio_encodings_residual
        return output


class Gemma3nAudioConformerBlock(nn.Module):
    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__()
        self.config = config

        self.ffw_layer_start = Gemma3nAudioConformerFeedForward(self.config)
        self.attention = Gemma3nAudioConformerAttention(self.config)
        self.lconv1d = Gemma3nAudioConformerLightConv1d(self.config)
        self.ffw_layer_end = Gemma3nAudioConformerFeedForward(self.config)
        self.register_buffer("gradient_clipping", torch.tensor(self.config.gradient_clipping), persistent=False)
        self.norm = Gemma3nRMSNorm(self.config.hidden_size)

    def forward(self, audio_encodings: torch.Tensor, audio_mel_mask: torch.BoolTensor) -> torch.Tensor:
        audio_encodings = self.ffw_layer_start(audio_encodings)
        audio_encodings = self.attention(audio_encodings, audio_mel_mask)
        validity_mask_for_lconv = ~audio_mel_mask  # True for valid
        audio_encodings_for_lconv_input = audio_encodings * validity_mask_for_lconv.unsqueeze(-1).to(
            audio_encodings.dtype
        )
        audio_encodings = self.lconv1d(audio_encodings_for_lconv_input)

        audio_encodings = self.ffw_layer_end(audio_encodings)
        audio_encodings = torch.clamp(audio_encodings, -self.gradient_clipping, self.gradient_clipping)
        output = self.norm(audio_encodings)
        return output


class Gemma3nAudioEncoder(PreTrainedModel):
    """
    An audio encoder based on the [Universal Speech Model](https://huggingface.co/papers/2303.01037) architecture.
    """

    config: Gemma3nAudioConfig

    main_input_name = "audio_mel"

    def __init__(self, config: Gemma3nAudioConfig):
        super().__init__(config)
        self.config = config

        self.subsample_conv_projection = Gemma3nAudioSubSampleConvProjection(config)
        self.conformer = nn.ModuleList(
            [Gemma3nAudioConformerBlock(config) for _ in range(config.conf_num_hidden_layers)]
        )

    def forward(
        self, audio_mel: torch.Tensor, audio_mel_mask: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Encodes a batch of MELs.

        Args:
            audio_mel: a torch.Tensor of shape [batch, num_frames, num_channels,
              mel_bins].

        Returns:
            audio_encodings: a torch.Tensor of shape
                `[batch_size, self.config.audio_soft_tokens_per_image,
                self.config.audio_config.hidden_size]`
            audio_mel_mask: a torch.BoolTensor of shape [batch, num_frames].
        """
        audio_encodings = self.subsample_conv_projection(audio_mel)  # audio_encodings: [B, T_sub, D]

        # Subsample the input audio_mel_mask to match the time dimension of audio_encodings (T_sub)
        t_sub = audio_encodings.shape[1]

        time_stride_product = 1
        for stride_pair_idx in range(len(self.config.sscp_conv_stride_size)):
            time_stride_product *= self.config.sscp_conv_stride_size[stride_pair_idx][0]

        # Create indices for gathering from the original mask.
        # These indices map to original time steps corresponding to the start of each
        # receptive field in the subsampled output.
        indices = torch.arange(t_sub, device=audio_mel_mask.device) * time_stride_product
        indices = torch.clamp(indices, max=audio_mel_mask.shape[1] - 1)  # Ensure indices are valid

        # Expand indices for batch compatibility if B > 1 and indices is 1D.
        if audio_mel_mask.ndim > 1 and indices.ndim == 1:
            indices = indices.unsqueeze(0).expand(audio_mel_mask.shape[0], -1)  # [B, T_sub]
        elif (
            audio_mel_mask.ndim == indices.ndim
            and audio_mel_mask.shape[0] == 1
            and indices.shape[0] != 1
            and t_sub == indices.shape[0]
        ):
            # Handle case where B=1 but indices became [T_sub] instead of [1, T_sub]
            indices = indices.unsqueeze(0)

        current_mask = torch.gather(audio_mel_mask, 1, indices)  # [B, T_sub]

        for block in self.conformer:
            audio_encodings = block(audio_encodings, current_mask)  # Pass the processed mask

        if self.config.conf_reduction_factor > 1:
            audio_encodings = audio_encodings[:, :: self.config.conf_reduction_factor]
            # Reduce the mask as well
            current_mask = current_mask[:, :: self.config.conf_reduction_factor]

        audio_encodings = audio_encodings.masked_fill(current_mask.unsqueeze(-1), 0.0)
        return audio_encodings, current_mask


class Gemma3nTextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class Gemma3nTextLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.config = config

        self.linear_left = nn.Linear(self.config.hidden_size, self.config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(self.config.laurel_rank, self.config.hidden_size, bias=False)
        self.post_laurel_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        laurel_hidden_states: torch.Tensor = self.linear_left(hidden_states)
        laurel_hidden_states: torch.Tensor = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        return hidden_states + normed_laurel_hidden_states


class Gemma3nTextMLP(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size[layer_idx]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_proj = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_states)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(self.activation_sparsity, dtype=torch.float32, device=inputs.device)
        # normal_dist and std_multiplier are adapted from jax.scipy.stats.norm.ppf().
        #
        # References:
        #   *   https://docs.jax.dev/en/latest/_autosummary/jax.scipy.stats.norm.ppf.html
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.normal.Normal
        #   *   https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return nn.functional.relu(inputs - cutoff_x)


class Gemma3nTextAltUp(nn.Module):
    """Alternating Updates (AltUp)

    The AltUp module wraps transformer layers. The `predict` step modifies the
    input to the transformer layer, and the `correct` step propagates the output
    of the transformer layer to the sparsely updated dimensions.

    See more in the research paper:

    https://proceedings.neurips.cc/paper_files/paper/2023/file/f2059277ac6ce66e7e5543001afa8bb5-Paper-Conference.pdf
    """

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__()
        self.config = config
        self.correct_output_scale = nn.Parameter(torch.zeros(self.config.hidden_size))
        self.correction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(self.config.hidden_size, self.config.altup_num_inputs, bias=False)
        self.router_norm = Gemma3nRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(self.config.hidden_size**-1.0), persistent=False)

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predicts the output of a layer using a trainable map.

        Args:
            hidden_states: A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` derived by
                stacking the input embeddings and preprocessing the last `num_altup_inputs - 1` matrices.

        Returns:
            A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` containing the predictions.
        """
        modalities = self.compute_router_modalities(hidden_states[self.config.altup_active_idx])

        if self.training and self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # Project and then transpose all 2D matrices contained so that mulmat gives the correct result
        all_coefs: torch.Tensor = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )

        # permute hidden_states to [batch_size, num_tokens, hidden_size, altup_num_inputs]
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # undo the permute
        predictions += hidden_states  # add the original input
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        """Corrects the predictions relative to the

        Args:
            predictions: A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` derived by
                stacking the input embeddings and preprocessing the last `num_altup_inputs - 1` matrices.
            activated: A 3D tensor of shape `[batch_size, num_tokens, hidden_size]` containing the activated inputs.

        Returns:
            A 4D tensor of shape `[num_altup_inputs, batch_size, num_tokens, hidden_size]` correcting the original
                predictions relative to the activated input embeddings.
        """
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.altup_active_idx]  # (batch, num_tokens, hidden_size)
        innovation = innovation.repeat(self.config.altup_num_inputs, 1, 1, 1)  # Repeat on dim0 to match predictions

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        # all_coefs adapted from jax.numpy.einsum("...p,pi->...i", ...)
        # Permute to (altup_num_inputs, batch_size, num_tokens) as the last dim is a scalar applied to each altup input
        # and expand on dim1 for broadcastability
        all_coefs: torch.Tensor = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions  # add the original input
        return corrected.contiguous().type_as(activated)

    def forward(self, corrected: torch.Tensor) -> torch.Tensor:
        """
        This is only defined as the `forward` so that accelerate hooks can move correctly `correct_output_scale`
        (which is a nn.Parameter, not a Module) between devices when offloading. It is otherwise only used in
        `scale_corrected_output`
        """
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        """Scales the provided 3D tensor of shape [batch_size, num_tokens, hidden_size]."""
        return self.forward(corrected)


class Gemma3nTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma3nTextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class Gemma3nTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma3nRMSNorm(dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False)

        first_kv_shared_layer_idx = self.config.num_hidden_layers - self.config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            # For shared layers, find the last non-shared layer of the same type before sharing starts
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            # For non-shared layers, store full-length kv if this is the last non-shared layer of its type
            self.store_full_length_kv = layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(
                config.layer_types[layer_idx]
            )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer
        if self.is_kv_shared_layer and past_key_values is not None:
            # Check if shared_layers exists and contains the required key
            if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                # Device of past layer may be different from current one
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
            else:
                # Fallback: compute key/value states normally if shared layer data is not available
                # This can happen during decode phase or with custom cache implementations
                key_states = self.k_proj(hidden_states).view(hidden_shape)
                key_states = self.k_norm(key_states)
                key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                key_states = key_states.transpose(1, 2)

                value_states = self.v_proj(hidden_states).view(hidden_shape)
                value_states = self.v_norm(value_states)
                value_states = value_states.transpose(1, 2)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3nTextAttentionSparseMM(Gemma3nTextAttention):
    """
    Gemma3n text attention module with SparseMM integration for adaptive KVCache.
    This module inherits from `Gemma3nTextAttention` and integrates SparseMM for
    head-adaptive KV cache compression.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer
        if self.is_kv_shared_layer and past_key_values is not None:
            # Check if shared_layers exists and contains the required key
            if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                # Device of past layer may be different from current one
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
            else:
                # Fallback: compute key/value states normally if shared layer data is not available
                # This can happen during decode phase or with custom cache implementations
                key_states = self.k_proj(hidden_states).view(hidden_shape)
                key_states = self.k_norm(key_states)
                key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                key_states = key_states.transpose(1, 2)

                value_states = self.v_proj(hidden_states).view(hidden_shape)
                value_states = self.v_norm(value_states)
                value_states = value_states.transpose(1, 2)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        is_prefill = query_states.shape[2] != 1

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }

            # CRITICAL FIX: Use class name check instead of isinstance to handle different import paths
            # This is necessary because DynamicCacheSplitHeadFlatten might be imported from different modules
            # (e.g., "model.sparsemm_utils" vs "sparsemm_utils"), causing isinstance to fail
            is_dynamic_cache_split_head = (
                isinstance(past_key_values, DynamicCacheSplitHeadFlatten) or
                type(past_key_values).__name__ == "DynamicCacheSplitHeadFlatten"
            )
            if is_dynamic_cache_split_head:
                # IMPORTANT: Handle KV shared layers properly with SparseMM
                # KV shared layers reuse KV from previous layers (store_full_length_kv layers)
                if self.is_kv_shared_layer:
                    # For KV shared layers:
                    # - Prefill: use shared_layers KV (already set in key_states/value_states at lines 1429-1446)
                    # - Decode: need to get updated shared_layers KV which includes new tokens
                    
                    if not is_prefill:
                        # DECODE PHASE for KV shared layers
                        # The source layer (kv_shared_layer_index) should have updated shared_layers
                        # We need to get the updated KV from shared_layers
                        if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                            key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                            key_states = key_states.to(query_states.device)
                            value_states = value_states.to(query_states.device)
                    
                    # NOTE: Do NOT call repeat_kv here - eager_attention_forward will handle it
                    # Skip SparseMM processing and go directly to attention computation
                    # (handled by the attention_interface below)
                else:
                    # Non-shared layers: use SparseMM processing
                    if not hasattr(self, "kv_cluster") or self.kv_cluster is None:
                        init_sparsemm(self)
                    
                    # IMPORTANT: For layers that provide KV for sharing, store full-length KV
                    if self.store_full_length_kv and is_prefill:
                        if not hasattr(past_key_values, "shared_layers"):
                            past_key_values.shared_layers = {}
                        # Store full-length KV for sharing before any compression
                        past_key_values.shared_layers[self.layer_idx] = (key_states.clone(), value_states.clone())
                    
                    if is_prefill:
                        # PREFILL PHASE
                        kv_seq_len = key_states.shape[-2]
                        
                        # CRITICAL: If budget is a ratio, dynamically adjust base_capacity before computing threshold
                        budget_ratio = getattr(self.config, 'budget_ratio', None)
                        if budget_ratio is not None and 0 < budget_ratio < 1.0:
                            # Compute dynamic budget based on actual sequence length
                            dynamic_budget = max(64, int(kv_seq_len * budget_ratio))
                            dynamic_base_capacity = dynamic_budget - self.kv_cluster.window_size
                            if dynamic_base_capacity < 1:
                                dynamic_base_capacity = 1
                            # Update base_capacity for this sample
                            self.kv_cluster.base_capacity = dynamic_base_capacity
                            # Recompute head_adaptive_capacity with new base_capacity
                            self.kv_cluster._compute_head_adaptive_capacity()
                        
                        # CRITICAL FIX: Use actual sequence length to determine if compression is needed
                        # This matches the logic in update_kv: compressible_len = q_len - window_size
                        # Compression is needed if: base_capacity < compressible_len
                        # Which is equivalent to: base_capacity < (kv_seq_len - window_size)
                        # Which is equivalent to: kv_seq_len > (base_capacity + window_size)
                        compressible_len = kv_seq_len - self.kv_cluster.window_size
                        needs_compression = compressible_len > 0 and self.kv_cluster.base_capacity < compressible_len
                        
                        # Debug: Print compression info only ONCE globally (first sample, first layer)
                        if self.layer_idx == 0 and not getattr(Gemma3nTextAttentionSparseMM, '_global_debug_printed', False):
                            budget_info = f", budget_ratio={budget_ratio}" if budget_ratio is not None else ""
                            print(f"[SparseMM DEBUG] kv_seq_len={kv_seq_len}, "
                                  f"compressible_len={compressible_len}, "
                                  f"base_capacity={self.kv_cluster.base_capacity}, "
                                  f"window_size={self.kv_cluster.window_size}{budget_info}, "
                                  f"needs_compression={needs_compression}")
                            Gemma3nTextAttentionSparseMM._global_debug_printed = True
                        
                        if not needs_compression:
                            # Sequence too short, no compression needed
                            num_kv_heads = self.config.num_key_value_heads
                            _device = key_states.device
                            
                            # Initialize metadata for decode phase
                            k_lens = [kv_seq_len] * num_kv_heads
                            klen_sum = kv_seq_len * num_kv_heads
                            max_seqlen_k = kv_seq_len
                            
                            if getattr(self.kv_cluster, 'head_lens', None) is None:
                                self.kv_cluster.reset_token_usage_stats()
                                self.kv_cluster._current_sample_started = True
                            
                            used_per_head = kv_seq_len
                            total_used = num_kv_heads * used_per_head
                            self.kv_cluster.total_token_budget += total_used
                            self.kv_cluster.total_heads_counted += num_kv_heads
                            self.kv_cluster.head_token_usage.extend([used_per_head] * num_kv_heads)
                            
                            self.kv_cluster.head_lens = torch.tensor(k_lens, dtype=torch.int32, device=_device)
                            self.kv_cluster.klen_sum = klen_sum
                            self.kv_cluster.max_seqlen_k = max_seqlen_k
                            self.kv_cluster.cu_headlens = torch.cumsum(self.kv_cluster.head_lens, dim=0, dtype=torch.int32)
                            self.kv_cluster.cu_klen = self.kv_cluster.cu_headlens - self.kv_cluster.head_lens
                            self.kv_cluster.cu_klen = torch.cat(
                                [self.kv_cluster.cu_klen, torch.tensor([klen_sum], dtype=torch.int32, device=_device)], dim=0)
                            self.kv_cluster.layer_qlens = torch.ones(num_kv_heads, dtype=torch.int32, device=_device)
                            self.kv_cluster.qlen_sum = num_kv_heads
                            self.kv_cluster.cu_qlen = torch.cumsum(self.kv_cluster.layer_qlens, dim=0, dtype=torch.int32) - self.kv_cluster.layer_qlens
                            self.kv_cluster.cu_qlen = torch.cat(
                                [self.kv_cluster.cu_qlen, torch.tensor([num_kv_heads], dtype=torch.int32, device=_device)], dim=0)
                            self.kv_cluster.cu_offset = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=_device)
                            self.kv_cluster.cu_head_offset = torch.arange(1, num_kv_heads + 1, dtype=torch.int32, device=_device)
                            
                            # Set cache_kwargs for prefill phase (important for DynamicCacheSplitHeadFlatten)
                            if cache_kwargs is None:
                                cache_kwargs = {}
                            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
                            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
                            
                            # Flatten key_states and value_states for DynamicCacheSplitHeadFlatten
                            key_states_flat = key_states.permute(0, 1, 2, 3).contiguous().reshape(-1, self.head_dim)
                            value_states_flat = value_states.permute(0, 1, 2, 3).contiguous().reshape(-1, self.head_dim)
                            past_key_values.update(key_states_flat, value_states_flat, self.layer_idx, cache_kwargs)
                        else:
                            # Compress KV and store to cache
                            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                                key_states, query_states, value_states
                            )
                            # Ensure cache_kwargs has head_lens and cu_klen after update_kv
                            # (update_kv should have initialized these)
                            if cache_kwargs is None:
                                cache_kwargs = {}
                            if self.kv_cluster.head_lens is not None and self.kv_cluster.cu_klen is not None:
                                cache_kwargs["head_lens"] = self.kv_cluster.head_lens
                                cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
                            past_key_values.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
                        
                        # Always use uncompressed KV for attention computation in prefill
                        # NOTE: Do NOT call repeat_kv here - eager_attention_forward will handle it
                        # key_states and value_states are already in correct shape [bsz, num_key_value_heads, seq_len, head_dim]
                    else:
                        # DECODE PHASE
                        device = key_states.device
                        num_kv_heads = self.config.num_key_value_heads
                        
                        # Initialize metadata if needed
                        # NOTE: head_lens should have been initialized during prefill phase.
                        # If it's None here, something went wrong. We should not initialize to zeros
                        # as that would break attention computation (all heads would have length 0).
                        is_first_init = (self.kv_cluster.head_lens is None)
                        if is_first_init:
                            # This should not happen if prefill phase worked correctly.
                            # As a safety measure, we'll raise an error to catch the issue early.
                            raise RuntimeError(
                                f"[SparseMM ERROR] head_lens is None in decode phase for layer {self.layer_idx}. "
                                "This indicates prefill phase did not properly initialize SparseMM state. "
                                "Please check that prefill phase completed successfully."
                            )
                        else:
                            # Ensure tensors are on the correct device
                            if self.kv_cluster.head_lens.device != device:
                                self.kv_cluster.head_lens = self.kv_cluster.head_lens.to(device)
                            if isinstance(self.kv_cluster.cu_qlen, torch.Tensor) and self.kv_cluster.cu_qlen.device != device:
                                self.kv_cluster.cu_qlen = self.kv_cluster.cu_qlen.to(device)
                            elif self.kv_cluster.cu_qlen is None:
                                self.kv_cluster.cu_qlen = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device)
                            if self.kv_cluster.cu_offset is None:
                                self.kv_cluster.cu_offset = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device)
                            elif isinstance(self.kv_cluster.cu_offset, torch.Tensor) and self.kv_cluster.cu_offset.device != device:
                                self.kv_cluster.cu_offset = self.kv_cluster.cu_offset.to(device)
                            if not isinstance(self.kv_cluster.cu_klen, torch.Tensor):
                                if self.kv_cluster.head_lens is not None:
                                    self.kv_cluster.cu_headlens = torch.cumsum(
                                        self.kv_cluster.head_lens, dim=0, dtype=torch.int32
                                    )
                                    self.kv_cluster.cu_klen = self.kv_cluster.cu_headlens - self.kv_cluster.head_lens
                                    self.kv_cluster.cu_klen = torch.cat(
                                        [
                                            self.kv_cluster.cu_klen,
                                            torch.tensor([self.kv_cluster.klen_sum], dtype=torch.int32, device=device),
                                        ],
                                        dim=0,
                                    )
                                else:
                                    self.kv_cluster.cu_klen = torch.zeros(num_kv_heads + 1, dtype=torch.int32, device=device)
                            elif self.kv_cluster.cu_klen.device != device:
                                self.kv_cluster.cu_klen = self.kv_cluster.cu_klen.to(device)
                        
                        # Set cache_kwargs if head_lens and cu_klen are valid tensors
                        if (self.kv_cluster.head_lens is not None and 
                            self.kv_cluster.cu_klen is not None and
                            isinstance(self.kv_cluster.head_lens, torch.Tensor) and
                            isinstance(self.kv_cluster.cu_klen, torch.Tensor)):
                            cache_kwargs["head_lens"] = self.kv_cluster.head_lens
                            cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
                        
                        # DECODE PHASE with DynamicCacheSplitHeadFlatten
                        # Use flash_attn_varlen_func for decode phase (like Qwen implementation)
                        # This is the correct way to handle per-head variable-length KV cache
                        
                        bsz = query_states.shape[0]
                        device = key_states.device
                        num_kv_heads = self.config.num_key_value_heads
                        
                        # Update cache with 4D KV states (like Qwen does)
                        # DynamicCacheSplitHeadFlatten.update handles the format internally
                        # Save original 4D key_states and value_states before update (for fallback cases)
                        key_states_original = key_states.clone()
                        value_states_original = value_states.clone()
                        key_states_updated, value_states_updated = past_key_values.update(
                            key_states, value_states, self.layer_idx, cache_kwargs
                        )
                        
                        # CRITICAL FIX: Update shared_layers in decode phase for store_full_length_kv layers
                        # This ensures KV shared layers can see newly generated tokens
                        if self.store_full_length_kv:
                            if not hasattr(past_key_values, "shared_layers"):
                                past_key_values.shared_layers = {}
                            
                            # Get existing KV from shared_layers (from prefill phase)
                            if self.layer_idx in past_key_values.shared_layers:
                                prev_key, prev_value = past_key_values.shared_layers[self.layer_idx]
                                prev_key = prev_key.to(key_states.device)
                                prev_value = prev_value.to(value_states.device)
                                # Append new token's KV to existing KV
                                # key_states shape: [bsz, num_kv_heads, 1, head_dim]
                                # prev_key shape: [bsz, num_kv_heads, prev_seq_len, head_dim]
                                updated_key = torch.cat([prev_key, key_states], dim=2)
                                updated_value = torch.cat([prev_value, value_states], dim=2)
                                past_key_values.shared_layers[self.layer_idx] = (updated_key, updated_value)
                            else:
                                # First time in decode phase (shouldn't happen if prefill ran correctly)
                                past_key_values.shared_layers[self.layer_idx] = (key_states.clone(), value_states.clone())
                        
                        # Update metadata AFTER cache update (so cu_klen reflects the updated cache)
                        kv_cluster = self.kv_cluster
                        kv_cluster.klen_sum += num_kv_heads
                        kv_cluster.max_seqlen_k += 1
                        kv_cluster.cu_klen.add_(kv_cluster.cu_offset)
                        kv_cluster.head_lens.add_(1)
                        
                        # Use flash_attn_varlen_func for decode phase
                        # This handles variable-length per-head KV cache correctly
                        if FLASH_ATTN_VARLEN_AVAILABLE:
                            # Reshape query for flash_attn_varlen_func: [total_q, num_key_value_groups, head_dim]
                            # query_states shape: [bsz, num_heads, 1, head_dim]
                            query_states_varlen = query_states.view(-1, self.num_key_value_groups, self.head_dim)
                            
                            # key_states_updated and value_states_updated from cache.update
                            # Shape: [total_kv, head_dim] -> [total_kv, 1, head_dim]
                            key_states_varlen = key_states_updated.view(-1, 1, self.head_dim)
                            value_states_varlen = value_states_updated.view(-1, 1, self.head_dim)
                            
                            # Get cumulative sequence lengths for variable-length attention
                            cu_seqlens_q = kv_cluster.cu_qlen
                            cu_seqlens_k = kv_cluster.cu_klen
                            max_seqlen_q = 1
                            max_seqlen_k = kv_cluster.max_seqlen_k
                            
                            # Call sparsemm_flash_attn_varlen_func (flash_attn_varlen_func with alias)
                            # CRITICAL: Gemma uses scaling=1.0 because Q and K are already normalized
                            # by q_norm and k_norm. We must pass softmax_scale=1.0 to match this behavior.
                            attn_output = sparsemm_flash_attn_varlen_func(
                                query_states_varlen,
                                key_states_varlen,
                                value_states_varlen,
                                cu_seqlens_q,
                                cu_seqlens_k,
                                max_seqlen_q,
                                max_seqlen_k,
                                causal=True,
                                softmax_scale=1.0
                            )
                            
                            # Reshape output: [num_kv_heads, num_key_value_groups, head_dim] -> [bsz, 1, hidden_size]
                            assert bsz == 1, "SparseMM decode currently only supports batch size 1"
                            num_attention_heads = self.config.num_attention_heads
                            hidden_size = self.config.hidden_size
                            q_len = 1  # decode phase always has q_len = 1
                            attn_output = attn_output.reshape(bsz, num_attention_heads, q_len, self.head_dim)
                            attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, hidden_size).contiguous()
                            attn_output = self.o_proj(attn_output)
                            return attn_output, None
                        else:
                            # Fallback: reconstruct 4D tensors (may have issues with variable-length heads)
                            max_seqlen_k = kv_cluster.max_seqlen_k
                            if max_seqlen_k > 0 and kv_cluster.cu_klen is not None and len(kv_cluster.cu_klen) > num_kv_heads:
                                # Initialize reconstructed KV tensors with zeros
                                key_states_recon = torch.zeros(
                                    bsz, num_kv_heads, max_seqlen_k, self.head_dim,
                                    dtype=key_states_updated.dtype, device=device
                                )
                                value_states_recon = torch.zeros(
                                    bsz, num_kv_heads, max_seqlen_k, self.head_dim,
                                    dtype=value_states_updated.dtype, device=device
                                )
                                
                                # Reconstruct each head's KV from flattened cache
                                cu_klen = kv_cluster.cu_klen
                                for head_idx in range(num_kv_heads):
                                    head_len = kv_cluster.head_lens[head_idx].item()
                                    if head_len > 0 and head_idx < len(cu_klen) - 1:
                                        start_idx = cu_klen[head_idx].item()
                                        end_idx = cu_klen[head_idx + 1].item()
                                        
                                        if end_idx > start_idx and end_idx <= key_states_updated.shape[0]:
                                            head_key = key_states_updated[start_idx:end_idx]
                                            head_value = value_states_updated[start_idx:end_idx]
                                            
                                            actual_len = head_key.shape[0]
                                            if actual_len <= max_seqlen_k:
                                                key_states_recon[0, head_idx, :actual_len] = head_key
                                                value_states_recon[0, head_idx, :actual_len] = head_value
                                
                                key_states = key_states_recon
                                value_states = value_states_recon
                            else:
                                # Fallback: key_states_updated and value_states_updated are 2D flattened tensors
                                # Need to reshape them back to 4D for repeat_kv
                                # key_states_updated shape: [total_kv, head_dim]
                                # We need to reconstruct 4D: [bsz, num_kv_heads, seq_len, head_dim]
                                # Use the current head_lens to determine the shape
                                if kv_cluster.head_lens is not None and len(kv_cluster.head_lens) > 0:
                                    max_len = kv_cluster.head_lens.max().item()
                                    if max_len > 0:
                                        # Reconstruct 4D tensor from 2D flattened cache
                                        key_states_recon = torch.zeros(
                                            bsz, num_kv_heads, max_len, self.head_dim,
                                            dtype=key_states_updated.dtype, device=device
                                        )
                                        value_states_recon = torch.zeros(
                                            bsz, num_kv_heads, max_len, self.head_dim,
                                            dtype=value_states_updated.dtype, device=device
                                        )
                                        
                                        cu_klen = kv_cluster.cu_klen if kv_cluster.cu_klen is not None else None
                                        if cu_klen is not None and len(cu_klen) > num_kv_heads:
                                            for head_idx in range(num_kv_heads):
                                                head_len = kv_cluster.head_lens[head_idx].item()
                                                if head_len > 0 and head_idx < len(cu_klen) - 1:
                                                    start_idx = cu_klen[head_idx].item()
                                                    end_idx = cu_klen[head_idx + 1].item()
                                                    
                                                    if end_idx > start_idx and end_idx <= key_states_updated.shape[0]:
                                                        head_key = key_states_updated[start_idx:end_idx]
                                                        head_value = value_states_updated[start_idx:end_idx]
                                                        
                                                        actual_len = head_key.shape[0]
                                                        if actual_len <= max_len:
                                                            key_states_recon[0, head_idx, :actual_len] = head_key
                                                            value_states_recon[0, head_idx, :actual_len] = head_value
                                        
                                        key_states = key_states_recon
                                        value_states = value_states_recon
                                    else:
                                        # No cached tokens, use original 4D key_states and value_states (current token only)
                                        # key_states_updated and value_states_updated are 2D, so use original 4D tensors
                                        key_states = key_states_original
                                        value_states = value_states_original
                                else:
                                    # No metadata available, use original 4D key_states and value_states (current token only)
                                    # key_states_updated and value_states_updated are 2D, so use original 4D tensors
                                    key_states = key_states_original
                                    value_states = value_states_original
                            
                            # NOTE: Do NOT call repeat_kv here - eager_attention_forward will handle it
            else:
                # For standard DynamicCache, use original logic
                if not self.is_kv_shared_layer:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )
                if self.store_full_length_kv:
                    if not hasattr(past_key_values, "shared_layers"):
                        past_key_values.shared_layers = {}
                    past_key_values.shared_layers[self.layer_idx] = key_states, value_states
                
                # NOTE: Do NOT call repeat_kv here - eager_attention_forward will handle it

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3nTextAttentionSnapKV(Gemma3nTextAttention):
    """
    Gemma3n text attention module with SnapKV integration.
    SnapKV selects important tokens based on attention patterns with pooling.
    
    Unlike AdaKV/SparseMM, SnapKV uses uniform compression across all heads,
    so we can use standard DynamicCache with 4D tensors instead of 
    DynamicCacheSplitHeadFlatten with variable-length attention.
    
    IMPORTANT: This class properly handles Gemma3n's KV sharing mechanism:
    - is_kv_shared_layer: Some layers reuse KV from previous layers
    - store_full_length_kv: Some layers store full KV for later sharing
    
    Implementation pattern:
    - Prefill: compress KV and store to cache, use original KV for attention
    - Decode: use standard attention with compressed KV from cache (all heads same length)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # CRITICAL: Handle KV shared layers (Gemma3n specific)
        # For layers with shared KV, reuse KV from previous non-shared layer
        if self.is_kv_shared_layer and past_key_values is not None:
            if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
            else:
                # Fallback: compute KV normally if shared layer data not available
                key_states = self.k_proj(hidden_states).view(hidden_shape)
                key_states = self.k_norm(key_states)
                key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                key_states = key_states.transpose(1, 2)

                value_states = self.v_proj(hidden_states).view(hidden_shape)
                value_states = self.v_norm(value_states)
                value_states = value_states.transpose(1, 2)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv_heads = self.config.num_key_value_heads
        
        # Initialize SnapKV cluster (only for non-shared layers)
        if not self.is_kv_shared_layer:
            if not hasattr(self, "kv_cluster") or self.kv_cluster is None:
                init_snapkv(self)

        is_prefill = q_len != 1

        # Save original KV for attention computation (before any compression)
        key_states_for_attn = key_states
        value_states_for_attn = value_states

        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            
            # IMPORTANT: For layers that provide KV for sharing, store full-length KV BEFORE compression
            # This must be done before compression so shared layers get the full uncompressed KV
            if self.store_full_length_kv and is_prefill:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                # Store full-length KV for sharing before any compression
                past_key_values.shared_layers[self.layer_idx] = (key_states.clone(), value_states.clone())
            
            # Only update cache for non-shared layers
            if not self.is_kv_shared_layer:
                if is_prefill:
                    # Prefill phase: compress KV and store to cache
                    # CRITICAL: Store compressed KV to cache, but use ORIGINAL KV for attention
                    key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
                    
                    # Store compressed KV to cache (for future decode steps)
                    past_key_values.update(
                        key_states_compress, value_states_compress, self.layer_idx, cache_kwargs
                    )
                    # key_states_for_attn and value_states_for_attn remain as original (uncompressed)
                else:
                    # Decode phase: update cache with new token and use cached KV for attention
                    key_states_for_attn, value_states_for_attn = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs
                    )
            
            # For decode phase, update shared_layers with cached KV (includes new tokens)
            if self.store_full_length_kv and not is_prefill:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states_for_attn, value_states_for_attn

        # Standard attention - use key_states_for_attn and value_states_for_attn
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states_for_attn,
            value_states_for_attn,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3nTextAttentionAdaKV(Gemma3nTextAttention):
    """
    Gemma3n text attention module with AdaKV integration for adaptive KVCache.
    AdaKV uses head-adaptive capacity allocation based on attention scores.
    
    Following Qwen's implementation pattern:
    - Prefill: compress KV and store to cache, but use original KV for attention
    - Decode: use variable-length flash attention with compressed KV from cache
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # CRITICAL: Handle KV shared layers (Gemma3n specific)
        # For layers with shared KV, reuse KV from previous non-shared layer
        if self.is_kv_shared_layer and past_key_values is not None:
            if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
            else:
                # Fallback: compute KV normally if shared layer data not available
                key_states = self.k_proj(hidden_states).view(hidden_shape)
                key_states = self.k_norm(key_states)
                key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                key_states = key_states.transpose(1, 2)

                value_states = self.v_proj(hidden_states).view(hidden_shape)
                value_states = self.v_norm(value_states)
                value_states = value_states.transpose(1, 2)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv_heads = self.config.num_key_value_heads
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "sliding_window": self.sliding_window} if past_key_values is not None else None

        # Initialize AdaKV cluster (only for non-shared layers)
        if not self.is_kv_shared_layer:
            if not hasattr(self, "kv_cluster") or self.kv_cluster is None:
                init_adakv(self)

        is_prefill = q_len != 1

        # Check for DynamicCacheSplitHeadFlatten - use both isinstance and class name check
        # to handle cases where the class is imported from different modules
        is_split_head_flatten_cache = (
            past_key_values is not None and (
                isinstance(past_key_values, DynamicCacheSplitHeadFlatten) or
                type(past_key_values).__name__ == "DynamicCacheSplitHeadFlatten"
            )
        )

        if is_split_head_flatten_cache:
            # IMPORTANT: Handle KV shared layers properly with AdaKV
            # KV shared layers reuse KV from previous layers (store_full_length_kv layers)
            if self.is_kv_shared_layer:
                # For KV shared layers:
                # - Prefill: use shared_layers KV (already set above)
                # - Decode: need to get updated shared_layers KV which includes new tokens
                
                if not is_prefill:
                    # DECODE PHASE for KV shared layers
                    if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                        key_states = key_states.to(query_states.device)
                        value_states = value_states.to(query_states.device)
                
                # Repeat KV for GQA and set key_states_attn/value_states_attn for attention computation
                key_states_attn = repeat_kv(key_states, self.num_key_value_groups)
                value_states_attn = repeat_kv(value_states, self.num_key_value_groups)
                # Skip AdaKV processing - use standard attention below
            elif is_prefill:
                # PREFILL PHASE for non-shared layers
                
                # IMPORTANT: For layers that provide KV for sharing, store full-length KV BEFORE compression
                # This must be done before compression so shared layers get the full uncompressed KV
                if self.store_full_length_kv:
                    if not hasattr(past_key_values, "shared_layers"):
                        past_key_values.shared_layers = {}
                    # Store full-length KV for sharing before any compression
                    past_key_values.shared_layers[self.layer_idx] = (key_states.clone(), value_states.clone())
                
                # Compress KV and store to cache
                # Following Qwen's pattern: compress for cache, use original for attention
                
                # CRITICAL FIX: If budget is a ratio, dynamically adjust base_capacity before compression
                kv_seq_len = key_states.shape[-2]
                budget_ratio = getattr(self.config, 'budget_ratio', None)
                if budget_ratio is not None and 0 < budget_ratio <= 1.0:
                    # Compute dynamic budget based on actual sequence length
                    dynamic_budget = max(64, int(kv_seq_len * budget_ratio))
                    dynamic_base_capacity = dynamic_budget - self.kv_cluster.window_size
                    if dynamic_base_capacity < 1:
                        dynamic_base_capacity = 1
                    # Update base_capacity for this sample
                    old_base_capacity = self.kv_cluster.base_capacity
                    self.kv_cluster.base_capacity = dynamic_base_capacity
                    # Recompute floor_capacity and adaptive_capacity with new base_capacity
                    self.kv_cluster.floor_capacity = int(dynamic_base_capacity * self.kv_cluster.floor_ratio)
                    self.kv_cluster.adaptive_capacity = dynamic_base_capacity - self.kv_cluster.floor_capacity
                    
                    # Debug: Print dynamic budget adjustment (only once globally for first layer)
                    if self.layer_idx == 0 and not getattr(Gemma3nTextAttentionAdaKV, '_global_debug_printed', False):
                        print(f"[AdaKV DEBUG] Dynamic budget adjustment: kv_seq_len={kv_seq_len}, "
                              f"budget_ratio={budget_ratio}, dynamic_budget={dynamic_budget}, "
                              f"base_capacity: {old_base_capacity} -> {dynamic_base_capacity}")
                        Gemma3nTextAttentionAdaKV._global_debug_printed = True
                
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
                past_key_values.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
                
                # Use original (uncompressed) KV for attention computation during prefill
                # This ensures prefill attention quality is not degraded
                key_states_attn = key_states
                value_states_attn = value_states
            else:
                # Decode phase: use variable-length flash attention with compressed KV
                device = key_states.device
                
                # CRITICAL FIX: head_lens should have been initialized during prefill phase.
                # If it's None here, something went wrong. We should not initialize to zeros
                # as that would break attention computation (all heads would have length 0).
                is_first_init = (self.kv_cluster.head_lens is None)
                if is_first_init:
                    # This should not happen if prefill phase worked correctly.
                    # As a safety measure, we'll raise an error to catch the issue early.
                    raise RuntimeError(
                        f"[AdaKV ERROR] head_lens is None in decode phase for layer {self.layer_idx}. "
                        "This indicates prefill phase did not properly initialize AdaKV state. "
                        "Please check that prefill phase completed successfully."
                    )
                else:
                    # Ensure tensors are on the correct device
                    if self.kv_cluster.head_lens.device != device:
                        self.kv_cluster.head_lens = self.kv_cluster.head_lens.to(device)
                    if isinstance(self.kv_cluster.cu_qlen, torch.Tensor) and self.kv_cluster.cu_qlen.device != device:
                        self.kv_cluster.cu_qlen = self.kv_cluster.cu_qlen.to(device)
                    elif self.kv_cluster.cu_qlen is None:
                        self.kv_cluster.cu_qlen = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device)
                    if self.kv_cluster.cu_offset is None:
                        self.kv_cluster.cu_offset = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device)
                    elif isinstance(self.kv_cluster.cu_offset, torch.Tensor) and self.kv_cluster.cu_offset.device != device:
                        self.kv_cluster.cu_offset = self.kv_cluster.cu_offset.to(device)
                    if not isinstance(self.kv_cluster.cu_klen, torch.Tensor):
                        if self.kv_cluster.head_lens is not None:
                            self.kv_cluster.cu_headlens = torch.cumsum(
                                self.kv_cluster.head_lens, dim=0, dtype=torch.int32
                            )
                            self.kv_cluster.cu_klen = self.kv_cluster.cu_headlens - self.kv_cluster.head_lens
                            self.kv_cluster.cu_klen = torch.cat(
                                [
                                    self.kv_cluster.cu_klen,
                                    torch.tensor([self.kv_cluster.klen_sum], dtype=torch.int32, device=device),
                                ],
                                dim=0,
                            )
                        else:
                            self.kv_cluster.cu_klen = torch.zeros(num_kv_heads + 1, dtype=torch.int32, device=device)
                    elif self.kv_cluster.cu_klen.device != device:
                        self.kv_cluster.cu_klen = self.kv_cluster.cu_klen.to(device)
                
                # Set cache_kwargs if head_lens and cu_klen are valid tensors
                if (self.kv_cluster.head_lens is not None and 
                    self.kv_cluster.cu_klen is not None and
                    isinstance(self.kv_cluster.head_lens, torch.Tensor) and
                    isinstance(self.kv_cluster.cu_klen, torch.Tensor)):
                    cache_kwargs["head_lens"] = self.kv_cluster.head_lens
                    cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
                
                key_states_updated, value_states_updated = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

                # CRITICAL FIX: Update shared_layers in decode phase for store_full_length_kv layers
                # This ensures KV shared layers can see newly generated tokens
                if self.store_full_length_kv:
                    if not hasattr(past_key_values, "shared_layers"):
                        past_key_values.shared_layers = {}
                    
                    # Get existing KV from shared_layers (from prefill phase)
                    if self.layer_idx in past_key_values.shared_layers:
                        prev_key, prev_value = past_key_values.shared_layers[self.layer_idx]
                        prev_key = prev_key.to(key_states.device)
                        prev_value = prev_value.to(value_states.device)
                        # Append new token's KV to existing KV
                        # key_states shape: [bsz, num_kv_heads, 1, head_dim]
                        # prev_key shape: [bsz, num_kv_heads, prev_seq_len, head_dim]
                        updated_key = torch.cat([prev_key, key_states], dim=2)
                        updated_value = torch.cat([prev_value, value_states], dim=2)
                        past_key_values.shared_layers[self.layer_idx] = (updated_key, updated_value)
                    else:
                        # First time in decode phase (shouldn't happen if prefill ran correctly)
                        past_key_values.shared_layers[self.layer_idx] = (key_states.clone(), value_states.clone())

                # Update metadata for next decode step
                self.kv_cluster.klen_sum += num_kv_heads
                self.kv_cluster.max_seqlen_k += 1
                self.kv_cluster.cu_klen.add_(self.kv_cluster.cu_offset)
                self.kv_cluster.head_lens.add_(1)

                # Use sparsemm_flash_attn_varlen_func for variable-length attention
                # Reshape for varlen attention: query [num_kv_heads * num_groups, head_dim], kv [total_kv, head_dim]
                num_key_value_groups = num_heads // num_kv_heads
                query_states_varlen = query_states.view(-1, num_key_value_groups, head_dim)
                key_states_varlen = key_states_updated.view(-1, 1, head_dim)
                value_states_varlen = value_states_updated.view(-1, 1, head_dim)

                # CRITICAL: Gemma uses scaling=1.0 because Q and K are already normalized
                # by q_norm and k_norm. We must pass softmax_scale=1.0 to match this behavior.
                attn_output = sparsemm_flash_attn_varlen_func(
                    query_states_varlen, key_states_varlen, value_states_varlen,
                    self.kv_cluster.cu_qlen, self.kv_cluster.cu_klen,
                    1, self.kv_cluster.max_seqlen_k, causal=True,
                    softmax_scale=1.0
                )
                attn_output = attn_output.reshape(bsz, num_heads, q_len, head_dim)
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None
        elif past_key_values is not None and not is_split_head_flatten_cache:
            # Fallback: use standard DynamicCache with standard attention
            # NOTE: This branch is ONLY for standard DynamicCache, NOT for DynamicCacheSplitHeadFlatten
            # DynamicCacheSplitHeadFlatten returns 2D tensors which are incompatible with repeat_kv
            # Update cache with current KV states
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Some custom cache implementations (e.g. AdaKV / SplitHeadFlatten variants)
            # may still return flattened 2D tensors here. In that case, repeat_kv (which
            # expects [batch, num_kv_heads, seqlen, head_dim]) will crash with a shape
            # unpack error. To be robust, only apply repeat_kv when we indeed receive 4D
            # tensors, otherwise fall back to using the cache output directly.
            if key_states.dim() == 4 and value_states.dim() == 4:
                # Repeat KV for GQA
                key_states_attn = repeat_kv(key_states, self.num_key_value_groups)
                value_states_attn = repeat_kv(value_states, self.num_key_value_groups)
            else:
                # Already flattened / incompatible with repeat_kv; use as-is
                key_states_attn = key_states
                value_states_attn = value_states
        else:
            # No cache or DynamicCacheSplitHeadFlatten that wasn't handled above - use original KV
            key_states_attn = key_states
            value_states_attn = value_states

        # Standard attention for prefill (or no cache case)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states_attn,
            value_states_attn,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3nTextAttentionPyramidKV(Gemma3nTextAttention):
    """
    Gemma3n text attention module with PyramidKV integration.
    PyramidKV uses layer-wise pyramidal capacity allocation.
    
    Following Qwen's implementation pattern:
    - Prefill: compress KV and store to cache, but use original KV for attention
    - Decode: use variable-length flash attention with compressed KV from cache
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # CRITICAL: Handle KV shared layers (Gemma3n specific)
        # For layers with shared KV, reuse KV from previous non-shared layer
        if self.is_kv_shared_layer and past_key_values is not None:
            if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
            else:
                # Fallback: compute KV normally if shared layer data not available
                key_states = self.k_proj(hidden_states).view(hidden_shape)
                key_states = self.k_norm(key_states)
                key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                key_states = key_states.transpose(1, 2)

                value_states = self.v_proj(hidden_states).view(hidden_shape)
                value_states = self.v_norm(value_states)
                value_states = value_states.transpose(1, 2)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv_heads = self.config.num_key_value_heads
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "sliding_window": self.sliding_window} if past_key_values is not None else None

        # Initialize PyramidKV cluster (only for non-shared layers)
        if not self.is_kv_shared_layer:
            if not hasattr(self, "kv_cluster") or self.kv_cluster is None:
                init_pyramidkv(self)

        is_prefill = q_len != 1

        # Check for DynamicCacheSplitHeadFlatten - use both isinstance and class name check
        # to handle cases where the class is imported from different modules
        is_split_head_flatten_cache = (
            past_key_values is not None and (
                isinstance(past_key_values, DynamicCacheSplitHeadFlatten) or
                type(past_key_values).__name__ == "DynamicCacheSplitHeadFlatten"
            )
        )

        if is_split_head_flatten_cache:
            # IMPORTANT: Handle KV shared layers properly with PyramidKV
            if self.is_kv_shared_layer:
                # For KV shared layers: use shared_layers KV directly
                if not is_prefill:
                    # DECODE PHASE for KV shared layers
                    if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                        key_states = key_states.to(query_states.device)
                        value_states = value_states.to(query_states.device)
                
                # Repeat KV for GQA and set key_states_attn/value_states_attn for attention computation
                key_states_attn = repeat_kv(key_states, self.num_key_value_groups)
                value_states_attn = repeat_kv(value_states, self.num_key_value_groups)
                # Skip PyramidKV processing - use standard attention below
            elif is_prefill:
                # PREFILL PHASE for non-shared layers
                
                # IMPORTANT: For layers that provide KV for sharing, store full-length KV BEFORE compression
                if self.store_full_length_kv:
                    if not hasattr(past_key_values, "shared_layers"):
                        past_key_values.shared_layers = {}
                    past_key_values.shared_layers[self.layer_idx] = (key_states.clone(), value_states.clone())
                
                # Compress KV and store to cache
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
                past_key_values.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
                
                # Use original (uncompressed) KV for attention computation during prefill
                key_states_attn = key_states
                value_states_attn = value_states
            else:
                # DECODE PHASE for non-shared layers
                device = key_states.device
                
                # Initialize head metadata if needed - get actual length from cache
                if not hasattr(self.kv_cluster, 'head_lens') or self.kv_cluster.head_lens is None:
                    cache_key = past_key_values.key_cache[self.layer_idx]
                    total_cache_len = cache_key.shape[0]
                    head_len = total_cache_len // num_kv_heads
                    
                    self.kv_cluster.head_lens = torch.full((num_kv_heads,), head_len, dtype=torch.int32, device=device)
                    self.kv_cluster.klen_sum = total_cache_len
                    self.kv_cluster.max_seqlen_k = head_len
                    self.kv_cluster.cu_klen = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device) * head_len
                    self.kv_cluster.cu_qlen = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device)
                    self.kv_cluster.cu_offset = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device)

                cache_kwargs["head_lens"] = self.kv_cluster.head_lens
                cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
                key_states_updated, value_states_updated = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

                # CRITICAL FIX: Update shared_layers in decode phase for store_full_length_kv layers
                if self.store_full_length_kv:
                    if not hasattr(past_key_values, "shared_layers"):
                        past_key_values.shared_layers = {}
                    if self.layer_idx in past_key_values.shared_layers:
                        prev_key, prev_value = past_key_values.shared_layers[self.layer_idx]
                        prev_key = prev_key.to(key_states.device)
                        prev_value = prev_value.to(value_states.device)
                        updated_key = torch.cat([prev_key, key_states], dim=2)
                        updated_value = torch.cat([prev_value, value_states], dim=2)
                        past_key_values.shared_layers[self.layer_idx] = (updated_key, updated_value)
                    else:
                        past_key_values.shared_layers[self.layer_idx] = (key_states.clone(), value_states.clone())

                self.kv_cluster.klen_sum += num_kv_heads
                self.kv_cluster.max_seqlen_k += 1
                self.kv_cluster.cu_klen.add_(self.kv_cluster.cu_offset)
                self.kv_cluster.head_lens.add_(1)

                num_key_value_groups = num_heads // num_kv_heads
                query_states_varlen = query_states.view(-1, num_key_value_groups, head_dim)
                key_states_varlen = key_states_updated.view(-1, 1, head_dim)
                value_states_varlen = value_states_updated.view(-1, 1, head_dim)

                # CRITICAL: Gemma uses scaling=1.0 because Q and K are already normalized
                # by q_norm and k_norm. We must pass softmax_scale=1.0 to match this behavior.
                attn_output = sparsemm_flash_attn_varlen_func(
                    query_states_varlen, key_states_varlen, value_states_varlen,
                    self.kv_cluster.cu_qlen, self.kv_cluster.cu_klen,
                    1, self.kv_cluster.max_seqlen_k, causal=True,
                    softmax_scale=1.0
                )
                attn_output = attn_output.reshape(bsz, num_heads, q_len, head_dim)
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None
        elif past_key_values is not None and not is_split_head_flatten_cache:
            # Fallback: use standard DynamicCache with standard attention
            # NOTE: This branch is ONLY for standard DynamicCache, NOT for DynamicCacheSplitHeadFlatten
            # DynamicCacheSplitHeadFlatten returns 2D tensors which are incompatible with repeat_kv
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states_attn = repeat_kv(key_states, self.num_key_value_groups)
            value_states_attn = repeat_kv(value_states, self.num_key_value_groups)
        else:
            # No cache or DynamicCacheSplitHeadFlatten that wasn't handled above - use original KV
            key_states_attn = key_states
            value_states_attn = value_states

        # Standard attention for prefill (or no cache case)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states_attn,
            value_states_attn,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3nTextAttentionMixKV(Gemma3nTextAttention):
    """
    Gemma3n text attention module with MixKV integration.
    MixKV uses mixed scoring strategies (attention, key similarity, value norm) for token selection.
    
    Following Qwen's implementation pattern:
    - Prefill: compress KV and store to cache, but use original KV for attention
    - Decode: use variable-length flash attention with compressed KV from cache
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # CRITICAL: Handle KV shared layers (Gemma3n specific)
        # For layers with shared KV, reuse KV from previous non-shared layer
        if self.is_kv_shared_layer and past_key_values is not None:
            if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
            else:
                # Fallback: compute KV normally if shared layer data not available
                key_states = self.k_proj(hidden_states).view(hidden_shape)
                key_states = self.k_norm(key_states)
                key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                key_states = key_states.transpose(1, 2)

                value_states = self.v_proj(hidden_states).view(hidden_shape)
                value_states = self.v_norm(value_states)
                value_states = value_states.transpose(1, 2)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv_heads = self.config.num_key_value_heads
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "sliding_window": self.sliding_window} if past_key_values is not None else None

        # Initialize MixKV cluster (only for non-shared layers)
        if not self.is_kv_shared_layer:
            if not hasattr(self, "kv_cluster") or self.kv_cluster is None:
                init_mixkv(self)

        is_prefill = q_len != 1

        # Check for DynamicCacheSplitHeadFlatten - use both isinstance and class name check
        # to handle cases where the class is imported from different modules
        is_split_head_flatten_cache = (
            past_key_values is not None and (
                isinstance(past_key_values, DynamicCacheSplitHeadFlatten) or
                type(past_key_values).__name__ == "DynamicCacheSplitHeadFlatten"
            )
        )

        if is_split_head_flatten_cache:
            # IMPORTANT: Handle KV shared layers properly with MixKV
            if self.is_kv_shared_layer:
                # For KV shared layers: use shared_layers KV directly
                if not is_prefill:
                    # DECODE PHASE for KV shared layers
                    if hasattr(past_key_values, "shared_layers") and self.kv_shared_layer_index in past_key_values.shared_layers:
                        key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
                        key_states = key_states.to(query_states.device)
                        value_states = value_states.to(query_states.device)
                
                # Repeat KV for GQA and set key_states_attn/value_states_attn for attention computation
                key_states_attn = repeat_kv(key_states, self.num_key_value_groups)
                value_states_attn = repeat_kv(value_states, self.num_key_value_groups)
                # Skip MixKV processing - use standard attention below
            elif is_prefill:
                # PREFILL PHASE for non-shared layers
                
                # IMPORTANT: For layers that provide KV for sharing, store full-length KV BEFORE compression
                if self.store_full_length_kv:
                    if not hasattr(past_key_values, "shared_layers"):
                        past_key_values.shared_layers = {}
                    past_key_values.shared_layers[self.layer_idx] = (key_states.clone(), value_states.clone())
                
                # CRITICAL FIX: If budget is a ratio, dynamically adjust max_capacity_prompt before compression
                # This matches AdaKV's implementation to ensure budget_ratio is correctly applied
                kv_seq_len = key_states.shape[-2]
                budget_ratio = getattr(self.config, 'budget_ratio', None)
                if budget_ratio is not None and 0 < budget_ratio <= 1.0:
                    # Compute dynamic budget based on actual sequence length
                    dynamic_budget = max(64, int(kv_seq_len * budget_ratio))
                    dynamic_max_capacity = dynamic_budget
                    if dynamic_max_capacity <= self.kv_cluster.window_size:
                        dynamic_max_capacity = self.kv_cluster.window_size + 1
                    # Update max_capacity_prompt for this sample
                    old_max_capacity = self.kv_cluster.max_capacity_prompt
                    self.kv_cluster.max_capacity_prompt = dynamic_max_capacity
                    # Reset pyram_init to allow recalculation if pyram_mode is used
                    if hasattr(self.kv_cluster, 'pyram_mode') and self.kv_cluster.pyram_mode:
                        self.kv_cluster.pyram_init = False
                    
                    # Debug: Print dynamic budget adjustment (only once globally for first layer)
                    if self.layer_idx == 0 and not getattr(Gemma3nTextAttentionMixKV, '_global_debug_printed', False):
                        print(f"[MixKV DEBUG] Dynamic budget adjustment: kv_seq_len={kv_seq_len}, "
                              f"budget_ratio={budget_ratio}, dynamic_budget={dynamic_budget}, "
                              f"max_capacity_prompt: {old_max_capacity} -> {dynamic_max_capacity}")
                        Gemma3nTextAttentionMixKV._global_debug_printed = True
                
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states)
                past_key_values.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
                
                # Use original (uncompressed) KV for attention computation during prefill
                key_states_attn = key_states
                value_states_attn = value_states
            else:
                # Decode phase: use variable-length flash attention with compressed KV
                device = key_states.device
                
                # Initialize head metadata if needed
                if not hasattr(self.kv_cluster, 'head_lens') or self.kv_cluster.head_lens is None:
                    cache_key = past_key_values.key_cache[self.layer_idx]
                    total_cache_len = cache_key.shape[0]
                    head_len = total_cache_len // num_kv_heads
                    
                    self.kv_cluster.head_lens = torch.full((num_kv_heads,), head_len, dtype=torch.int32, device=device)
                    self.kv_cluster.klen_sum = total_cache_len
                    self.kv_cluster.max_seqlen_k = head_len
                    self.kv_cluster.cu_klen = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device) * head_len
                    self.kv_cluster.cu_qlen = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device)
                    self.kv_cluster.cu_offset = torch.arange(0, num_kv_heads + 1, dtype=torch.int32, device=device)

                cache_kwargs["head_lens"] = self.kv_cluster.head_lens
                cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
                key_states_updated, value_states_updated = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

                # CRITICAL FIX: Update shared_layers in decode phase for store_full_length_kv layers
                if self.store_full_length_kv:
                    if not hasattr(past_key_values, "shared_layers"):
                        past_key_values.shared_layers = {}
                    if self.layer_idx in past_key_values.shared_layers:
                        prev_key, prev_value = past_key_values.shared_layers[self.layer_idx]
                        prev_key = prev_key.to(key_states.device)
                        prev_value = prev_value.to(value_states.device)
                        updated_key = torch.cat([prev_key, key_states], dim=2)
                        updated_value = torch.cat([prev_value, value_states], dim=2)
                        past_key_values.shared_layers[self.layer_idx] = (updated_key, updated_value)
                    else:
                        past_key_values.shared_layers[self.layer_idx] = (key_states.clone(), value_states.clone())

                self.kv_cluster.klen_sum += num_kv_heads
                self.kv_cluster.max_seqlen_k += 1
                self.kv_cluster.cu_klen.add_(self.kv_cluster.cu_offset)
                self.kv_cluster.head_lens.add_(1)

                num_key_value_groups = num_heads // num_kv_heads
                query_states_varlen = query_states.view(-1, num_key_value_groups, head_dim)
                key_states_varlen = key_states_updated.view(-1, 1, head_dim)
                value_states_varlen = value_states_updated.view(-1, 1, head_dim)

                # CRITICAL: Gemma uses scaling=1.0 because Q and K are already normalized
                # by q_norm and k_norm. We must pass softmax_scale=1.0 to match this behavior.
                attn_output = sparsemm_flash_attn_varlen_func(
                    query_states_varlen, key_states_varlen, value_states_varlen,
                    self.kv_cluster.cu_qlen, self.kv_cluster.cu_klen,
                    1, self.kv_cluster.max_seqlen_k, causal=True,
                    softmax_scale=1.0
                )
                attn_output = attn_output.reshape(bsz, num_heads, q_len, head_dim)
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None
        elif past_key_values is not None and not is_split_head_flatten_cache:
            # Fallback: use standard DynamicCache with standard attention
            # NOTE: This branch is ONLY for standard DynamicCache, NOT for DynamicCacheSplitHeadFlatten
            # DynamicCacheSplitHeadFlatten returns 2D tensors which are incompatible with repeat_kv
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_states_attn = repeat_kv(key_states, self.num_key_value_groups)
            value_states_attn = repeat_kv(value_states, self.num_key_value_groups)
        else:
            # No cache or DynamicCacheSplitHeadFlatten that wasn't handled above - use original KV
            key_states_attn = key_states
            value_states_attn = value_states

        # Standard attention for prefill (or no cache case)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states_attn,
            value_states_attn,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3nTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3nTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3nTextAttention(config, layer_idx)
        self.mlp = Gemma3nTextMLP(config, layer_idx=layer_idx)
        self.input_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.act_fn = ACT2FN[config.hidden_activation]

        self.altup = Gemma3nTextAltUp(config)
        self.laurel = Gemma3nTextLaurelBlock(config)
        self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
        self.post_per_layer_input_norm = Gemma3nRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        per_layer_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        attn, self_attn_weights = self.self_attn(
            hidden_states=active_prediction_normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx].clone()
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # per_layer_input_gate adapted from jax.numpy.einsum("btd,dp->btp", ...)
        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        first_prediction = torch.multiply(first_prediction, per_layer_input)

        # per_layer_projection adapted from jax.numpy.einsum("btp,pd->btd", ...)
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)
        corrected_predictions[1:] += first_prediction

        outputs = (corrected_predictions,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class Gemma3nPreTrainedModel(PreTrainedModel):
    config: Gemma3nConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma3nTextDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Gemma3nTextDecoderLayer,
        "attentions": Gemma3nTextAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Gemma3nAudioCumulativeGroupNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Gemma3nAudioAttention):
            module.per_dim_scale.data.zero_()
        elif isinstance(module, Gemma3nTextAltUp):
            module.correct_output_scale.data.zero_()


@auto_docstring(custom_intro="The base Gemma 3n language model without a language modeling head.")
class Gemma3nTextModel(Gemma3nPreTrainedModel):
    config: Gemma3nTextConfig

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3n downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3nTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3nTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Gemma3nRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3nTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO (raushan): Fix this after RoPE refactor. For now we hack it by
        # reassigning thetas when we want to create a local RoPE layer. Config
        # defaults should hold values for global RoPE.
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3nTextRotaryEmbedding(config=config)

        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            self.padding_idx,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = nn.Linear(
            self.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        self.per_layer_projection_norm = Gemma3nRMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        self.altup_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.altup_unembed_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.register_buffer("per_layer_projection_scale", torch.tensor(self.hidden_size**-0.5), persistent=False)
        self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        per_layer_inputs (torch.Tensor, *optional*, defaults to None):
            Pre-computed per-layer embeddings. If None, they are derived from input_ids if provided.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.get_per_layer_inputs(input_ids)

        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states_0 = inputs_embeds

        # Initialize RoPE embeddings
        position_embeddings_global = self.rotary_emb(hidden_states_0, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states_0, position_ids)

        # Expand hidden_states to support per-layer inputs
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5)

        temp_hidden_states = [hidden_states_0]
        for i in range(1, self.config.altup_num_inputs):
            # altup_proj adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_proj = self.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states, dim=0)  # [num_altup_inputs, batch, seq_len, hidden_size]

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            causal_mask = causal_mask_mapping[decoder_layer.attention_type]
            per_layer_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global,
                position_embeddings_local,
                per_layer_input,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer (but before reprojecting to stay consistent with layer output)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Per-layer inputs to single output
        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        temp_hidden_states = [hidden_states[0]]
        for i in range(1, self.config.altup_num_inputs):
            # altup_unembed_projections adapted from jax.numpy.einsum("btp,pd->btd", ...)
            altup_unemb_proj: torch.Tensor = self.altup_unembed_projections[i - 1](hidden_states[i])
            current_hidden_state = altup_unemb_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states)
        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_layer_projection: torch.Tensor = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self.per_layer_projection_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            # per-layer inputs are sometimes padded with zeros, slice the relevant embeddings.
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )


@auto_docstring(custom_intro="The base Gemma 3n language model with a language modeling head.")
class Gemma3nForCausalLM(Gemma3nPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma3nTextConfig
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {"model.language_model": "model"}

    def __init__(self, config: Gemma3nTextConfig):
        super().__init__(config)
        self.model = Gemma3nTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma3nForCausalLM

        >>> model = Gemma3nForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Gemma3nMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: Union[Gemma3nAudioConfig, Gemma3nVisionConfig],
        text_config: Gemma3nTextConfig,
    ):
        super().__init__()

        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = multimodal_config.vocab_offset
        self.vocab_size = multimodal_config.vocab_size
        self.text_hidden_size = text_config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.multimodal_hidden_size)
        self.hard_embedding_norm = Gemma3nRMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.soft_embedding_norm = Gemma3nRMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, self.text_hidden_size, bias=False)
        self.embedding_post_projection_norm = Gemma3nRMSNorm(self.text_hidden_size, eps=self.eps, with_scale=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.

        Args:
            input_ids: A torch.LongTensor containing the token ids to embed. Values should be in the range
                `[vocab_offset, vocab_offset + vocab_size)`.
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.

        Returns:
            A torch.Tensor of embeddings with  shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is not None:
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:
            hard_emb = self.embedding(input_ids - self.vocab_offset)
            emb_norm = self.hard_embedding_norm(hard_emb)

        emb_norm_proj = self.embedding_projection(emb_norm)
        return self.embedding_post_projection_norm(emb_norm_proj)


@auto_docstring(
    custom_intro="""
    The base Gemma 3n model comprising a vision backbone, an audio backbone, and a language model without a
    language modeling head.
    """
)
class Gemma3nModel(Gemma3nPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        # Disable flash attention for vision_tower (TimmWrapperModel doesn't support it)
        vision_config = config.vision_config
        if hasattr(vision_config, '_attn_implementation_internal'):
            vision_config._attn_implementation_internal = "eager"
        self.vision_tower = AutoModel.from_config(
            config=vision_config,
            attn_implementation="eager"
        )
        self.vocab_size = config.text_config.vocab_size

        language_model = AutoModel.from_config(config=config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input
        # Disable flash attention for audio_tower if it uses TimmWrapperModel
        audio_config = config.audio_config
        if hasattr(audio_config, '_attn_implementation_internal'):
            audio_config._attn_implementation_internal = "eager"
        self.audio_tower = AutoModel.from_config(
            config=audio_config,
            attn_implementation="eager"
        )
        self.embed_vision = Gemma3nMultimodalEmbedder(config.vision_config, config.text_config)
        self.embed_audio = Gemma3nMultimodalEmbedder(config.audio_config, config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values, do_pooling=False, return_dict=True
        ).last_hidden_state
        # Convert from (batch, channels, height, width) to (batch, height * width, channels) where:
        # height == width and height * width == Gemma3nConfig.vision_soft_tokens_per_image.
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).permute(0, 2, 1)
        # Normalize and embed the soft tokens into language model space.
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(inputs_embeds=vision_outputs)

    def get_placeholder_mask(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0] * image_features.shape[1]}"
            )

        n_audio_tokens = special_audio_mask.sum()
        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if audio_features is not None and inputs_embeds[special_audio_mask].numel() != audio_features.numel():
            raise ValueError(
                f"Audio features and image tokens do not match: tokens: {n_audio_tokens}, features {audio_features.shape[0] * audio_features.shape[1]}"
            )

        return special_image_mask, special_audio_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text inputs
        pixel_values: Optional[torch.FloatTensor] = None,  # vision inputs
        input_features: Optional[torch.FloatTensor] = None,  # audio inputs
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **lm_kwargs,
    ) -> Gemma3nCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        >>> model = Gemma3nForConditionalGeneration.from_pretrained("google/gemma3n2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/gemma3n2-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # Prepare per-layer inputs from inputs_ids
            per_layer_inputs_mask = torch.logical_and(input_ids >= 0, input_ids < self.vocab_size_per_layer_input)
            per_layer_inputs_tokens = torch.where(per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids))
            per_layer_inputs = self.language_model.get_per_layer_inputs(per_layer_inputs_tokens)

            # Handle vision tokens (>= embed_vision.vocab_offset and < embed_audio.vocab_offset)
            vision_mask = torch.logical_and(
                input_ids >= self.embed_vision.vocab_offset, input_ids < self.embed_audio.vocab_offset
            )
            dummy_vision_token_id = self.embed_vision.vocab_offset + self.embed_vision.vocab_size - 1
            vision_input_ids = torch.where(vision_mask, input_ids, dummy_vision_token_id).to(inputs_embeds.device)
            vision_embeds = self.embed_vision(input_ids=vision_input_ids)
            expanded_vision_mask = vision_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(expanded_vision_mask, vision_embeds, inputs_embeds)

            # Handle audio tokens (>= embed_audio.vocab_offset)
            audio_mask = input_ids >= self.embed_audio.vocab_offset
            dummy_audio_token_id = self.embed_audio.vocab_offset + self.embed_audio.vocab_size - 1
            audio_input_ids = torch.where(audio_mask, input_ids, dummy_audio_token_id).to(inputs_embeds.device)
            audio_embeds = self.embed_audio(input_ids=audio_input_ids)
            expanded_audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(expanded_audio_mask, audio_embeds, inputs_embeds)
        else:
            per_layer_inputs = None

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Merge text and audio
        if input_features is not None and input_features_mask is not None:
            audio_features, audio_mask = self.get_audio_features(input_features, ~input_features_mask)

            # The Gemma3nProcessor expects all audio will be 30s in length and inserts 188 audio soft tokens into the
            # text to account for this. However, the audio preprocessing and encoder do not gurarantee they will
            # produce 188 soft tokens; they will produce at most that many tokens, but they may produce fewer tokens
            # depending on the length of the longest audio input in the batch. When we encounter this situation, we pad
            # the audio feature out to 188 soft tokens with the emebedding of the last token in the embed_audio vocab.
            audio_padding_toks = torch.tensor([[self.vocab_size - 1]], dtype=torch.long, device=audio_features.device)
            audio_padding_embs = self.embed_audio(input_ids=audio_padding_toks)
            audio_features = torch.where(audio_mask.unsqueeze(-1), audio_padding_embs, audio_features)

            audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape
            extra_padding_tokens = self.config.audio_soft_tokens_per_image - audio_seq_len
            extra_padding_features = audio_padding_embs.expand(audio_batch_size, extra_padding_tokens, audio_embed_dim)

            audio_features = torch.cat((audio_features, extra_padding_features), dim=1)
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, special_audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, audio_features=audio_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self.language_model(
            input_ids=None,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

        return Gemma3nModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            audio_hidden_states=audio_features if input_features is not None else None,
        )

    def get_audio_features(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the last hidden state from the audio encoder into language model space.

        Args:
            input_features (`torch.FloatTensor]` of shape `(num_images, seq_length, num_features)`):
               The tensors corresponding to the input audio.
            input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
               The attention mask for the input audio.

        Returns:
            audio_features (`torch.Tensor`): Audio feature tensor of shape `(num_images, audio_length, embed_dim)`).
        """
        audio_outputs, audio_mask = self.audio_tower(input_features, input_features_mask)
        return self.embed_audio(inputs_embeds=audio_outputs), audio_mask


@auto_docstring(
    custom_intro="""
    The base Gemma 3n model comprising a vision backbone, an audio backbone, a language model, and a language modeling
    head.
    """
)
class Gemma3nForConditionalGeneration(Gemma3nPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    base_model_prefix = "model"

    def __init__(self, config: Gemma3nConfig):
        super().__init__(config)
        self.model = Gemma3nModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        raise AttributeError("Use embed_vision instead of multi_modal_projector.")

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text inputs
        pixel_values: Optional[torch.FloatTensor] = None,  # vision inputs
        input_features: Optional[torch.FloatTensor] = None,  # audio inputs
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Gemma3nCausalLMOutputWithPast:
        r"""
        input_features_mask (torch.Tensor, *optional*, defaults to None):
            The attention mask for the input audio.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in
            `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
        >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

        >>> messages = [
        ...     {
        ...         "role": "system",
        ...         "content": [
        ...             {"type": "text", "text": "You are a helpful assistant."}
        ...         ]
        ...     },
        ...     {
        ...         "role": "user", "content": [
        ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Where is the cat standing?"},
        ...         ]
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages,
        ...     tokenizer=True,
        ...     return_dict=True,
        ...     return_tensors="pt",
        ...     add_generation_prompt=True
        ... )
        >>> # Generate
        >>> generate_ids = model.generate(**inputs)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **lm_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if (final_logit_softcapping := self.config.get_text_config().final_logit_softcapping) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        return Gemma3nCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # If we're in cached decoding stage, multimodal inputs should be None because input ids do not contain special
        # tokens anymore. Otherwise multimodal inputs should be passed to model.
        # NOTE: use_cache=False always needs pixel_values, input_features, and input_features_mask
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["input_features"] = input_features
            model_inputs["input_features_mask"] = input_features_mask

        return model_inputs

    @property
    def audio_tower(self):
        return self.model.audio_tower

    def enable_sparsemm_attention(
        self,
        head_score_data: Dict[str, Any],
        sparsemm_config: Dict[str, Any]
    ):
        """
        Replace attention modules with SparseMM version for adaptive KVCache.
        
        Args:
            head_score_data: Dictionary containing head scores with keys:
                - 'scores': List of head scores [layer0_head0, layer0_head1, ..., layerN_headM]
                - 'tensor': torch.Tensor of normalized head scores
            sparsemm_config: Dictionary containing SparseMM configuration:
                - 'budget': KV cache budget
                - 'ratio': Compression ratio
                - 'window_size': Window size for SparseMM
                - 'kernel_size': Kernel size for SparseMM
                - 'pooling': Pooling method ('maxpool', 'avgpool', etc.)
        """
        if not SPARSEM_AVAILABLE:
            raise ImportError("sparsemm_utils is not available. Please ensure sparsemm_utils.py is in the same directory.")
        
        # Find base model
        base_model = self.model.language_model  # Gemma3nTextModel
        # Get model config
        model_config = base_model.config
        num_hidden_layers = model_config.num_hidden_layers
        num_attention_heads = model_config.num_attention_heads
        num_key_value_heads = getattr(model_config, 'num_key_value_heads', num_attention_heads)
        num_key_value_groups = num_attention_heads // num_key_value_heads
        
        # Set head_score in config for SparseMM initialization
        def audio_load_head_score(model_type, num_hidden_layers=None):
            # Return format: dict with "layer-head" keys and score lists
            # Note: num_hidden_layers parameter is accepted for interface compatibility,
            # but we use the outer scope's num_hidden_layers variable
            head_scores_dict = {}
            head_score_list = head_score_data['scores']
            
            idx = 0
            for layer_idx in range(num_hidden_layers):
                for head_idx in range(num_attention_heads):
                    key = f"{layer_idx}-{head_idx}"
                    head_scores_dict[key] = [head_score_list[idx]]
                    idx += 1
            
            return head_scores_dict
        
        # Temporarily replace load_head_score in sparsemm_utils
        import sparsemm_utils as sparsemm_utils
        original_load_head_score = getattr(sparsemm_utils, 'load_head_score', None)
        sparsemm_utils.load_head_score = audio_load_head_score
        
        # Apply SparseMM configuration to model config
        model_config.head_score = 'audio'
        model_config.model_type = 'gemma'  # Ensure model_type is set
        model_config.window_size = sparsemm_config.get('window_size', 64)
        budget_value = sparsemm_config.get('budget', 4096)
        budget_ratio = sparsemm_config.get('budget_ratio', None)  # Store ratio if provided
        model_config.max_capacity_prompt = budget_value
        model_config.budget_ratio = budget_ratio  # Store ratio for dynamic adjustment
        model_config.kernel_size = sparsemm_config.get('kernel_size', 7)
        model_config.pooling = sparsemm_config.get('pooling', 'maxpool')
        model_config.ratio = sparsemm_config.get('ratio', 0.2)
        # Set gqa_func for Gemma models
        if not hasattr(model_config, 'gqa_func'):
            model_config.gqa_func = 'mean'
        
        patched_count = 0
        for layer_idx, layer in enumerate(base_model.layers):
            # Get the current attention module
            old_attn = layer.self_attn  # Gemma3nTextAttention()
            old_device = next(old_attn.parameters()).device
            old_dtype = next(old_attn.parameters()).dtype
            new_attn = Gemma3nTextAttentionSparseMM(
                config=model_config,
                layer_idx=layer_idx
            )  # Gemma3nTextAttentionSparseMM()
            new_attn = new_attn.to(device=old_device, dtype=old_dtype)
            new_attn.q_proj.load_state_dict(old_attn.q_proj.state_dict())
            new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
            new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
            new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
            new_attn.q_norm.load_state_dict(old_attn.q_norm.state_dict())
            new_attn.k_norm.load_state_dict(old_attn.k_norm.state_dict())
            new_attn.v_norm.load_state_dict(old_attn.v_norm.state_dict())
            
            # Copy rotary embedding if it exists
            if hasattr(old_attn, 'rotary_emb') and hasattr(new_attn, 'rotary_emb'):
                try:
                    new_attn.rotary_emb.load_state_dict(old_attn.rotary_emb.state_dict())
                except Exception as e:
                    print(f"[SparseMM WARNING] Could not copy rotary_emb for layer {layer_idx}: {e}")
            
            init_sparsemm(new_attn)
            
            if hasattr(new_attn, 'kv_cluster') and hasattr(new_attn.kv_cluster, 'head_adaptive_capacity'):
                new_attn.head_adaptive_capacity = new_attn.kv_cluster.head_adaptive_capacity
            
            # Replace the attention module
            layer.self_attn = new_attn
            patched_count += 1
        
        # Restore original load_head_score if it existed
        if original_load_head_score is not None:
            sparsemm_utils.load_head_score = original_load_head_score
        
        return True

    def enable_snapkv_attention(
        self,
        sparsemm_config: Dict[str, Any]
    ):
        """
        Replace attention modules with SnapKV version for KVCache compression.
        
        Args:
            sparsemm_config: Dictionary containing configuration:
                - 'budget': KV cache budget
                - 'budget_ratio': Budget ratio for dynamic adjustment (optional)
                - 'window_size': Window size
                - 'kernel_size': Kernel size for pooling
                - 'pooling': Pooling method
        """
        if not SPARSEM_AVAILABLE:
            raise ImportError("sparsemm_utils is not available.")
        
        base_model = self.model.language_model
        model_config = base_model.config
        
        # Apply configuration to model config
        model_config.window_size = sparsemm_config.get('window_size', 64)
        model_config.max_capacity_prompt = sparsemm_config.get('budget', 4096)
        model_config.kernel_size = sparsemm_config.get('kernel_size', 7)
        model_config.pooling = sparsemm_config.get('pooling', 'maxpool')
        # CRITICAL: Store budget_ratio for dynamic adjustment
        budget_ratio = sparsemm_config.get('budget_ratio', None)
        model_config.budget_ratio = budget_ratio
        
        if budget_ratio is not None:
            print(f"[SnapKV] budget_ratio={budget_ratio} enabled for dynamic budget adjustment")
        
        patched_count = 0
        for layer_idx, layer in enumerate(base_model.layers):
            old_attn = layer.self_attn
            old_device = next(old_attn.parameters()).device
            old_dtype = next(old_attn.parameters()).dtype
            new_attn = Gemma3nTextAttentionSnapKV(config=model_config, layer_idx=layer_idx)
            new_attn = new_attn.to(device=old_device, dtype=old_dtype)
            
            # Copy weights
            new_attn.q_proj.load_state_dict(old_attn.q_proj.state_dict())
            new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
            new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
            new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
            new_attn.q_norm.load_state_dict(old_attn.q_norm.state_dict())
            new_attn.k_norm.load_state_dict(old_attn.k_norm.state_dict())
            new_attn.v_norm.load_state_dict(old_attn.v_norm.state_dict())
            
            init_snapkv(new_attn)
            layer.self_attn = new_attn
            patched_count += 1
        
        print(f"[SnapKV] Total layers patched: {patched_count}/{len(base_model.layers)}")
        return True

    def enable_adakv_attention(
        self,
        sparsemm_config: Dict[str, Any]
    ):
        """
        Replace attention modules with AdaKV version for adaptive KVCache compression.
        
        Args:
            sparsemm_config: Dictionary containing configuration:
                - 'budget': KV cache budget
                - 'budget_ratio': Budget ratio for dynamic adjustment (optional)
                - 'window_size': Window size
                - 'kernel_size': Kernel size for pooling
                - 'pooling': Pooling method
        """
        if not SPARSEM_AVAILABLE:
            raise ImportError("sparsemm_utils is not available.")
        
        base_model = self.model.language_model
        model_config = base_model.config
        
        # Apply configuration to model config
        model_config.window_size = sparsemm_config.get('window_size', 64)
        model_config.max_capacity_prompt = sparsemm_config.get('budget', 4096)
        model_config.kernel_size = sparsemm_config.get('kernel_size', 7)
        model_config.pooling = sparsemm_config.get('pooling', 'maxpool')
        # CRITICAL: Store budget_ratio for dynamic adjustment in forward pass
        budget_ratio = sparsemm_config.get('budget_ratio', None)
        model_config.budget_ratio = budget_ratio
        
        if budget_ratio is not None:
            print(f"[AdaKV] budget_ratio={budget_ratio} enabled for dynamic budget adjustment")
        
        patched_count = 0
        for layer_idx, layer in enumerate(base_model.layers):
            old_attn = layer.self_attn
            old_device = next(old_attn.parameters()).device
            old_dtype = next(old_attn.parameters()).dtype
            new_attn = Gemma3nTextAttentionAdaKV(config=model_config, layer_idx=layer_idx)
            new_attn = new_attn.to(device=old_device, dtype=old_dtype)
            
            # Copy weights
            new_attn.q_proj.load_state_dict(old_attn.q_proj.state_dict())
            new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
            new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
            new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
            new_attn.q_norm.load_state_dict(old_attn.q_norm.state_dict())
            new_attn.k_norm.load_state_dict(old_attn.k_norm.state_dict())
            new_attn.v_norm.load_state_dict(old_attn.v_norm.state_dict())
            
            init_adakv(new_attn)
            layer.self_attn = new_attn
            patched_count += 1
        
        print(f"[AdaKV] Total layers patched: {patched_count}/{len(base_model.layers)}")
        return True

    def enable_pyramidkv_attention(
        self,
        sparsemm_config: Dict[str, Any]
    ):
        """
        Replace attention modules with PyramidKV version for layer-wise pyramidal KVCache.
        
        Args:
            sparsemm_config: Dictionary containing configuration:
                - 'budget': KV cache budget
                - 'budget_ratio': Budget ratio for dynamic adjustment (optional)
                - 'window_size': Window size
                - 'kernel_size': Kernel size for pooling
                - 'pooling': Pooling method
        """
        if not SPARSEM_AVAILABLE:
            raise ImportError("sparsemm_utils is not available.")
        
        base_model = self.model.language_model
        model_config = base_model.config
        
        # Apply configuration to model config
        model_config.window_size = sparsemm_config.get('window_size', 64)
        model_config.max_capacity_prompt = sparsemm_config.get('budget', 4096)
        model_config.kernel_size = sparsemm_config.get('kernel_size', 7)
        model_config.pooling = sparsemm_config.get('pooling', 'maxpool')
        # CRITICAL: Store budget_ratio for dynamic adjustment
        budget_ratio = sparsemm_config.get('budget_ratio', None)
        model_config.budget_ratio = budget_ratio
        
        if budget_ratio is not None:
            print(f"[PyramidKV] budget_ratio={budget_ratio} enabled for dynamic budget adjustment")
        # add
        if not hasattr(model_config, 'gqa_func'):
            model_config.gqa_func = 'mean'
        patched_count = 0
        for layer_idx, layer in enumerate(base_model.layers):
            old_attn = layer.self_attn
            old_device = next(old_attn.parameters()).device
            old_dtype = next(old_attn.parameters()).dtype
            new_attn = Gemma3nTextAttentionPyramidKV(config=model_config, layer_idx=layer_idx)
            new_attn = new_attn.to(device=old_device, dtype=old_dtype)
            
            # Copy weights
            new_attn.q_proj.load_state_dict(old_attn.q_proj.state_dict())
            new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
            new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
            new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
            new_attn.q_norm.load_state_dict(old_attn.q_norm.state_dict())
            new_attn.k_norm.load_state_dict(old_attn.k_norm.state_dict())
            new_attn.v_norm.load_state_dict(old_attn.v_norm.state_dict())
            
            init_pyramidkv(new_attn)
            layer.self_attn = new_attn
            patched_count += 1
        
        print(f"[PyramidKV] Total layers patched: {patched_count}/{len(base_model.layers)}")
        return True

    def enable_mixkv_attention(
        self,
        sparsemm_config: Dict[str, Any]
    ):
        """
        Replace attention modules with MixKV version for mixed-strategy KVCache compression.
        
        Args:
            sparsemm_config: Dictionary containing configuration:
                - 'budget': KV cache budget
                - 'budget_ratio': Budget ratio for dynamic adjustment (optional)
                - 'window_size': Window size
                - 'kernel_size': Kernel size for pooling
                - 'pooling': Pooling method
                - 'select_method': Selection method ('attn', 'keydiff', 'mixkv', etc.)
        """
        if not SPARSEM_AVAILABLE:
            raise ImportError("sparsemm_utils is not available.")
        
        base_model = self.model.language_model
        model_config = base_model.config
        
        # Apply configuration to model config
        model_config.window_size = sparsemm_config.get('window_size', 64)
        model_config.max_capacity_prompt = sparsemm_config.get('budget', 4096)
        model_config.kernel_size = sparsemm_config.get('kernel_size', 7)
        model_config.pooling = sparsemm_config.get('pooling', 'maxpool')
        model_config.select_method = sparsemm_config.get('select_method', 'mixkv')
        # CRITICAL: Store budget_ratio for dynamic adjustment in forward pass
        budget_ratio = sparsemm_config.get('budget_ratio', None)
        model_config.budget_ratio = budget_ratio
        
        if budget_ratio is not None:
            print(f"[MixKV] budget_ratio={budget_ratio} enabled for dynamic budget adjustment")
        
        patched_count = 0
        for layer_idx, layer in enumerate(base_model.layers):
            old_attn = layer.self_attn
            old_device = next(old_attn.parameters()).device
            old_dtype = next(old_attn.parameters()).dtype
            new_attn = Gemma3nTextAttentionMixKV(config=model_config, layer_idx=layer_idx)
            new_attn = new_attn.to(device=old_device, dtype=old_dtype)
            
            # Copy weights
            new_attn.q_proj.load_state_dict(old_attn.q_proj.state_dict())
            new_attn.k_proj.load_state_dict(old_attn.k_proj.state_dict())
            new_attn.v_proj.load_state_dict(old_attn.v_proj.state_dict())
            new_attn.o_proj.load_state_dict(old_attn.o_proj.state_dict())
            new_attn.q_norm.load_state_dict(old_attn.q_norm.state_dict())
            new_attn.k_norm.load_state_dict(old_attn.k_norm.state_dict())
            new_attn.v_norm.load_state_dict(old_attn.v_norm.state_dict())
            
            init_mixkv(new_attn)
            layer.self_attn = new_attn
            patched_count += 1
        
        print(f"[MixKV] Total layers patched: {patched_count}/{len(base_model.layers)}")
        return True


__all__ = [
    "Gemma3nAudioEncoder",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
    "Gemma3nModel",
    "Gemma3nPreTrainedModel",
    "Gemma3nTextModel",
    "Gemma3nTextAttentionSparseMM",
    "Gemma3nTextAttentionSnapKV",
    "Gemma3nTextAttentionAdaKV",
    "Gemma3nTextAttentionPyramidKV",
    "Gemma3nTextAttentionMixKV",
]
