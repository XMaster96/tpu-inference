# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict

import jax
import jax.numpy as jnp


def _get_rope_type(rope_scaling: Dict[str, Any] | None) -> str | None:
    if rope_scaling is None:
        return None
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type is None:
        return None
    return str(rope_type)


def _yarn_get_mscale(scale_factor: float, mscale_value: float = 1.0) -> float:
    if scale_factor <= 1.0:
        return 1.0
    return 0.1 * mscale_value * math.log(scale_factor) + 1.0


def _resolve_yarn_attention_factor(rope_scaling: Dict[str, Any]) -> float:
    attn_factor = rope_scaling.get("attn_factor")
    attention_factor = rope_scaling.get("attention_factor")
    if attn_factor is not None and attention_factor is not None and attn_factor != attention_factor:
        raise ValueError(
            "YaRN rope_scaling defines conflicting attn_factor and attention_factor values."
        )
    if attn_factor is not None:
        return float(attn_factor)
    if attention_factor is not None:
        return float(attention_factor)

    factor = float(rope_scaling["factor"])
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")
    if mscale is not None and mscale_all_dim is not None:
        return _yarn_get_mscale(factor, float(mscale)) / _yarn_get_mscale(
            factor, float(mscale_all_dim))
    return _yarn_get_mscale(factor)


def _apply_yarn_inv_freq(
    inv_freq: jax.Array,
    *,
    rope_theta: float,
    head_dim: int,
    rope_scaling: Dict[str, Any],
) -> tuple[jax.Array, float]:
    factor = float(rope_scaling["factor"])
    if factor <= 0.0:
        raise ValueError("YaRN rope_scaling factor must be positive.")

    original_max_position_value = rope_scaling.get(
        "original_max_position_embeddings")
    if original_max_position_value is None:
        raise ValueError(
            "YaRN rope_scaling requires original_max_position_embeddings.")
    original_max_position_embeddings = float(original_max_position_value)

    beta_fast = float(rope_scaling.get("beta_fast", 32.0))
    beta_slow = float(rope_scaling.get("beta_slow", 1.0))
    truncate = bool(rope_scaling.get("truncate", True))

    def find_correction_dim(num_rotations: float, dimensions: int, base: float,
                            max_positions: float) -> float:
        return (dimensions * math.log(max_positions /
                                      (num_rotations * 2.0 * math.pi))) / (
                                          2.0 * math.log(base))

    def find_correction_range(
        low_rotations: float,
        high_rotations: float,
        dimensions: int,
        base: float,
        max_positions: float,
    ) -> tuple[float, float]:
        low_value = find_correction_dim(low_rotations, dimensions, base,
                                        max_positions)
        high_value = find_correction_dim(high_rotations, dimensions, base,
                                         max_positions)
        if truncate:
            low_value = math.floor(low_value)
            high_value = math.ceil(high_value)
        return max(low_value, 0.0), min(high_value, float(dimensions - 1))

    def linear_ramp_factor(minimum_value: float, maximum_value: float,
                           dimensions: int) -> jax.Array:
        if math.isclose(minimum_value, maximum_value):
            maximum_value = minimum_value + 0.001
        linear_function = (jnp.arange(dimensions, dtype=jnp.float32) -
                           minimum_value) / (maximum_value - minimum_value)
        return jnp.clip(linear_function, 0.0, 1.0)

    low_value, high_value = find_correction_range(beta_fast, beta_slow,
                                                  head_dim, rope_theta,
                                                  original_max_position_embeddings)

    inv_freq_extrapolation = inv_freq
    inv_freq_interpolation = inv_freq / factor
    inv_freq_extrapolation_factor = 1.0 - linear_ramp_factor(
        low_value, high_value, head_dim // 2)
    inv_freq_result = inv_freq_interpolation * (
        1.0 - inv_freq_extrapolation_factor
    ) + inv_freq_extrapolation * inv_freq_extrapolation_factor

    attention_factor = _resolve_yarn_attention_factor(rope_scaling)
    return inv_freq_result, attention_factor


def apply_rope(
    # (seq_len, num_heads, head_dim)
    inputs: jax.Array,
    # (3, seq_len) for M-RoPE, otherwise (seq_len,)
    positions: jax.Array,
    head_dim: int,
    rope_theta: float = 10000,
    rope_scaling: Dict[str, Any] = None,
    rope_input_ordering: str = "split",
) -> jax.Array:
    """
    Applies Rotary Positional Embedding using the sine and cosine strategy.

    This implementation assumes the input tensor has a shape that might include
    padding on the last dimension (head_dim).
    RoPE is applied only to the first `head_dim` features, and the result is
    padded back to the original dimension if necessary.
    If rope_input_ordering is "split", then the input pairs for rotation are taken one from the
    first and one from the second half of the head_dim. If it is "interleaved" then
    adjacent values are used as inputs for rotation.
    """

    # M-RoPE support for Qwen2.5-VL
    if positions.ndim == 2 and positions.shape[0] == 3:
        mrope_section = rope_scaling.get("mrope_section",
                                         None) if rope_scaling else None
        # NOTE: We assume mrope_section is always available
        # as Qwen2.5-VL is the only model using mrope
        assert mrope_section is not None

        split_indices = [mrope_section[0], mrope_section[0] + mrope_section[1]]

        # Indices for the features to be rotated (first half of head_dim)
        all_freq_indices = jnp.arange(head_dim // 2)

        # Split the indices according to mrope_section. This is valid because split_indices are static.
        freq_indices_split = jnp.split(all_freq_indices, split_indices)
        # freq_indices_split is a list of 3 JAX arrays.

        cos_list = []
        sin_list = []

        for i in range(3):  # For each of the 3 position dimensions
            current_indices = freq_indices_split[i]

            if current_indices.size == 0:
                # This section is empty, skip.
                continue

            # inv_freq shape: (mrope_section[i],)
            inv_freq = 1.0 / (rope_theta**(current_indices * 2.0 / head_dim))

            # positions[i]: (seq_len,)
            # freqs shape: (seq_len, mrope_section[i])
            freqs = jnp.outer(positions[i].astype(jnp.float32), inv_freq)

            cos_list.append(jnp.cos(freqs))
            sin_list.append(jnp.sin(freqs))

        # Concatenate along the feature dimension
        # cos, sin shape: (seq_len, head_dim//2)
        cos = jnp.concatenate(cos_list, axis=1)
        sin = jnp.concatenate(sin_list, axis=1)

        # Add num_heads dimension for broadcasting
        cos = cos[:, jnp.newaxis, :]  # Shape: (seq_len, 1, head_dim//2)
        sin = sin[:, jnp.newaxis, :]  # Shape: (seq_len, 1, head_dim//2)

        # Apply rotation
        inputs_real = inputs[..., :head_dim // 2]
        inputs_imag = inputs[..., head_dim // 2:head_dim]

        outputs_real = inputs_real * cos - inputs_imag * sin
        outputs_imag = inputs_real * sin + inputs_imag * cos

        out = jnp.concatenate([outputs_real, outputs_imag], axis=-1)

    # Standard RoPE
    else:
        # Calculate inverse frequencies (timescale)
        fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
        timescale = 1.0 / (rope_theta**fraction)
        attention_factor = 1.0

        # Apply scaling if provided
        if rope_scaling:
            if _get_rope_type(rope_scaling) == "yarn":
                timescale, attention_factor = _apply_yarn_inv_freq(
                    timescale,
                    rope_theta=rope_theta,
                    head_dim=head_dim,
                    rope_scaling=rope_scaling,
                )
            else:
                timescale = apply_rope_scaling(timescale, rope_scaling)

        # Prepare for rotation by calculating sin and cos values
        # `sinusoid_inp` gets shape (batch * seq_len, head_dim/2)
        sinusoid_inp = positions.astype(timescale.dtype)[..., jnp.newaxis] * timescale[
            jnp.newaxis, :]

        # Broadcast over the 'heads' dimension, assuming shape (batch*seq, heads, head_dim)
        sinusoid_inp = sinusoid_inp[:, jnp.newaxis, ...]
        sin = jnp.sin(sinusoid_inp) * attention_factor
        cos = jnp.cos(sinusoid_inp) * attention_factor

        if rope_input_ordering == "interleaved":
            # Reshape to group adjacent features for rotation, matching new_apply_rope
            rotary_inputs = inputs[
                ..., :head_dim]  # Take just the non-padded amount.
            reshaped_inputs = rotary_inputs.reshape(*rotary_inputs.shape[:-1],
                                                    -1, 2)

            # Apply the rotation
            first_half = reshaped_inputs[..., 0]
            second_half = reshaped_inputs[..., 1]
        else:
            first_half = inputs[..., :head_dim // 2]
            second_half = inputs[..., head_dim // 2:head_dim]

        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin

        # Combine the rotated parts and reshape back
        if rope_input_ordering == "interleaved":
            out_stacked = jnp.stack([first_part, second_part], axis=-1)
            out = out_stacked.reshape(rotary_inputs.shape)
        else:
            out = jnp.concatenate([first_part, second_part], axis=-1)

    # If the original input was padded, pad the output with zeros to match.
    padded_head_dim = inputs.shape[-1]
    if padded_head_dim > head_dim:
        pad_width = padded_head_dim - head_dim
        pad_config = [(0, 0)] * (out.ndim - 1) + [(0, pad_width)]
        out = jnp.pad(out, pad_config)

    return out.astype(inputs.dtype)


def apply_longrope(
    inputs: jax.Array,
    positions: jax.Array,
    head_dim: int,
    rope_scaling: Dict[str, Any],
    original_max_position_embeddings: int,
    max_position_embeddings: int,
    rope_theta: float = 10000,
) -> jax.Array:
    # LongRoPE implementation specific to Phi-3
    # Implementation based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi3/modeling_phi3.py#L197-L235

    scale = max_position_embeddings / original_max_position_embeddings
    if scale <= 1.0:
        mscale = 1.0
    else:
        mscale = jnp.sqrt(1 + (jnp.log(scale) /
                               jnp.log(original_max_position_embeddings)))

    seq_len = inputs.shape[0]
    if seq_len > original_max_position_embeddings:
        long_factor = jnp.array(rope_scaling.get("long_factor"))
        timescale = 1.0 / (long_factor * (rope_theta**(
            (2 * jnp.arange(0, head_dim // 2)) / head_dim)))
    else:
        short_factor = jnp.array(rope_scaling.get("short_factor"))
        timescale = 1.0 / (short_factor * (rope_theta**(
            (2 * jnp.arange(0, head_dim // 2)) / head_dim)))

    # Calculate RoPE positions
    sinusoid_inp = positions.astype(timescale.dtype)[..., jnp.newaxis] * timescale[
        jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[:, jnp.newaxis, ...]
    sin = jnp.sin(sinusoid_inp) * mscale
    cos = jnp.cos(sinusoid_inp) * mscale

    # Padding logic
    padded_head_dim = inputs.shape[-1]

    # Apply RoPE mechanism
    first_half = inputs[..., :head_dim // 2]
    second_half = inputs[..., head_dim // 2:head_dim]

    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)

    if padded_head_dim > head_dim:
        out = jnp.pad(out, ((0, 0), (0, 0), (0, padded_head_dim - head_dim)))

    return out.astype(inputs.dtype)


def apply_rope_scaling(freqs: jax.Array, rope_scaling: Dict[str,
                                                            Any]) -> jax.Array:
    # Values obtained from grid search
    scale_factor = rope_scaling.get("scale_factor", 8.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling.get("original_max_position_embeddings",
                                       8192)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / freqs
    smooth = (old_context_len / wavelen -
              low_freq_factor) / (high_freq_factor - low_freq_factor)

    high_freqs = jnp.where(wavelen < high_freq_wavelen, freqs, 0)
    low_freqs = jnp.where(wavelen > low_freq_wavelen, freqs / scale_factor, 0)
    mid_freqs = jnp.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * freqs / scale_factor + smooth * freqs,
        0,
    )
    new_freqs = high_freqs + low_freqs + mid_freqs
    return new_freqs
