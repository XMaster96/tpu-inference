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
"""Utilities for downloading model weights from HuggingFace."""

import functools
import glob
import importlib
import math
import os
import re
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import jax
import jax.numpy as jnp
import torch
import torchax
from etils import epath
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import SingleDeviceSharding
from safetensors import safe_open
from torchax.ops.mappings import t2j
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import AutoWeightsLoader

from tpu_inference import envs, utils
from tpu_inference.layers.jax import JaxModule
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils import file_utils

logger = init_logger(__name__)

HF_WEIGHTS_FORMAT = "*.safetensors"

DTYPE_VIEW_MAP = {
    jnp.dtype(jnp.float8_e4m3fn): torch.uint8,
    jnp.dtype(jnp.bfloat16): torch.uint16,
    jnp.dtype(jnp.float32): torch.uint32,
}

_MODLAX_ROUTING_FLAGS = (
    "use_qkv_routing_experts",
    "use_attention_out_routing_experts",
    "use_mlp_gate_routing_experts",
    "use_mlp_up_routing_experts",
    "use_mlp_down_routing_experts",
    "use_norm_routing_experts",
)

_MODLAX_ADAPTER_CONFIG_FILENAMES = ("adapter_config.yml",
                                    "adapter_config.yaml")
_MODLAX_ADAPTER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


def _normalize_modlax_adapter_config(adapter_config: Any,
                                     source: str) -> dict[str, Any]:
    if isinstance(adapter_config, dict):
        name = adapter_config.get("name")
        iffm = adapter_config.get("iffm")
    else:
        name = getattr(adapter_config, "name", None)
        iffm = getattr(adapter_config, "iffm", None)

    if not isinstance(name, str):
        raise ValueError(f"{source} adapter name must be a string.")
    if not _MODLAX_ADAPTER_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"{source} adapter name {name!r} must match "
                         f"{_MODLAX_ADAPTER_NAME_PATTERN.pattern}.")
    try:
        iffm_value = float(iffm)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} adapter iffm must be a positive number.") from exc
    if iffm_value <= 0.0:
        raise ValueError(f"{source} adapter iffm must be positive.")
    return {"name": name, "iffm": iffm_value}


def normalize_modlax_adapter_configs(adapters: Any,
                                     source: str = "adapters"
                                     ) -> list[dict[str, Any]]:
    if adapters is None:
        return []
    if not isinstance(adapters, (list, tuple)):
        raise ValueError(f"{source} must be a list of adapter configs.")

    normalized = [
        _normalize_modlax_adapter_config(adapter, source)
        for adapter in adapters
    ]
    duplicate_names = sorted({
        adapter["name"]
        for adapter in normalized
        if sum(other["name"] == adapter["name"] for other in normalized) > 1
    })
    if duplicate_names:
        raise ValueError("Duplicate Modlax adapter names are not allowed: "
                         f"{', '.join(duplicate_names)}")
    return normalized


def get_single_modlax_adapter_config(adapters: Any,
                                     source: str = "adapters"
                                     ) -> dict[str, Any] | None:
    normalized = normalize_modlax_adapter_configs(adapters, source)
    if len(normalized) > 1:
        raise ValueError(
            "The TPU online RL server supports at most one Modlax adapter; "
            f"{source} contains {len(normalized)} adapters.")
    return normalized[0] if normalized else None


def modlax_adapter_configs_match(left: dict[str, Any] | None,
                                 right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        return left is right
    return left["name"] == right["name"] and math.isclose(
        float(left["iffm"]),
        float(right["iffm"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def resolve_single_modlax_adapter_config(
    base_adapters: Any,
    override_adapters: Any,
    *,
    base_source: str = "checkpoint adapters",
    override_source: str = "runtime adapter override",
) -> dict[str, Any] | None:
    base_adapter = get_single_modlax_adapter_config(base_adapters, base_source)
    override_adapter = get_single_modlax_adapter_config(
        override_adapters, override_source)
    if (base_adapter is not None and override_adapter is not None and
            not modlax_adapter_configs_match(base_adapter, override_adapter)):
        raise ValueError(
            "Configured Modlax adapter does not match checkpoint adapter: "
            f"{override_source}={override_adapter}, {base_source}={base_adapter}"
        )
    return override_adapter or base_adapter


def _import_orbax_checkpoint():
    try:
        return importlib.import_module("orbax.checkpoint")
    except ImportError as exc:
        raise ImportError(
            "orbax.checkpoint is required to load Modlax Orbax checkpoints."
        ) from exc


def _canonical_jax_dtype(dtype_name: str) -> jnp.dtype:
    normalized = str(dtype_name).lower().replace("torch.", "")
    dtype_map: dict[str, jnp.dtype] = {
        "bfloat16": jnp.bfloat16,
        "bf16": jnp.bfloat16,
        "float16": jnp.float16,
        "f16": jnp.float16,
        "float32": jnp.float32,
        "f32": jnp.float32,
        "float64": jnp.float64,
        "f64": jnp.float64,
    }
    if normalized not in dtype_map:
        raise ValueError(
            f"Unsupported Modlax checkpoint dtype '{dtype_name}'. "
            f"Supported values: {sorted(dtype_map.keys())}")
    return dtype_map[normalized]


def _split_gcs_uri(path: str) -> tuple[str, str] | None:
    if not path.startswith("gs://"):
        return None
    uri_body = path[5:].strip("/")
    if not uri_body:
        return None
    bucket_name, *object_parts = uri_body.split("/", 1)
    if not bucket_name:
        return None
    object_path = object_parts[0] if object_parts else ""
    return bucket_name, object_path


def _gcs_blob_exists(blob_uri: str) -> bool:
    parsed = _split_gcs_uri(blob_uri)
    if parsed is None:
        return False
    bucket_name, object_path = parsed
    if not object_path:
        return False
    from google.cloud import storage

    client = storage.Client()
    return client.bucket(bucket_name).blob(object_path).exists()


def _read_gcs_text(blob_uri: str, encoding: str = "utf-8") -> str:
    parsed = _split_gcs_uri(blob_uri)
    if parsed is None:
        raise ValueError(f"Expected gs:// URI, got: {blob_uri}")
    bucket_name, object_path = parsed
    if not object_path:
        raise ValueError(
            f"Expected gs:// URI with object path, got: {blob_uri}")
    from google.cloud import storage

    client = storage.Client()
    return client.bucket(bucket_name).blob(object_path).download_as_text(
        encoding=encoding)


def _checkpoint_member_path(checkpoint_path: str, member_name: str) -> str:
    return f"{checkpoint_path.rstrip('/')}/{member_name}"


def _checkpoint_path_candidates(path: str) -> list[str]:
    normalized = path.rstrip("/")
    candidates = [normalized]
    if normalized.endswith("/final_model"):
        parent = normalized.rsplit("/", 1)[0]
        if parent:
            candidates.append(parent)
    else:
        candidates.append(f"{normalized}/final_model")
    deduped_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped_candidates:
            deduped_candidates.append(candidate)
    return deduped_candidates


def _path_exists(path: str) -> bool:
    exists = False
    try:
        exists = epath.Path(path).exists()
    except Exception:
        exists = False

    if exists:
        return True
    if not path.startswith("gs://"):
        return False

    try:
        return _gcs_blob_exists(path)
    except Exception:
        return False


def _read_text(path: str, encoding: str = "utf-8") -> str:
    try:
        return epath.Path(path).read_text(encoding=encoding)
    except Exception:
        if not path.startswith("gs://"):
            raise
    return _read_gcs_text(path, encoding=encoding)


def _load_yaml_dict(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to parse Modlax model_config.yml.") from exc

    payload = yaml.safe_load(_read_text(path, encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected YAML mapping at {path}, got {type(payload).__name__}.")
    return payload


def _is_modlax_orbax_checkpoint(path: str | None) -> bool:
    return _resolve_modlax_orbax_checkpoint_path(path) is not None


def _is_modlax_adapter_orbax_checkpoint(path: str | None) -> bool:
    return _resolve_modlax_adapter_orbax_checkpoint_path(path) is not None


def _classify_modlax_orbax_checkpoint(path: str | None) -> str | None:
    if _resolve_modlax_orbax_checkpoint_path(path) is not None:
        return "full_model"
    if _resolve_modlax_adapter_orbax_checkpoint_path(path) is not None:
        return "adapter_only"
    return None


def _resolve_modlax_orbax_checkpoint_path(path: str | None) -> str | None:
    if not path:
        return None
    for candidate in _checkpoint_path_candidates(path):
        config_path = _checkpoint_member_path(candidate, "model_config.yml")
        metadata_path = _checkpoint_member_path(candidate,
                                                "_CHECKPOINT_METADATA")
        if _path_exists(config_path) and _path_exists(metadata_path):
            return candidate
    return None


def _resolve_modlax_adapter_orbax_checkpoint_path(
        path: str | None) -> str | None:
    if not path:
        return None
    for candidate in _checkpoint_path_candidates(path):
        metadata_path = _checkpoint_member_path(candidate,
                                                "_CHECKPOINT_METADATA")
        if not _path_exists(metadata_path):
            continue
        for config_filename in _MODLAX_ADAPTER_CONFIG_FILENAMES:
            config_path = _checkpoint_member_path(candidate, config_filename)
            if _path_exists(config_path):
                return candidate
    return None


def _load_modlax_orbax_checkpoint_config(
        checkpoint_path: str) -> dict[str, Any]:
    resolved_checkpoint_path = _resolve_modlax_orbax_checkpoint_path(
        checkpoint_path)
    if resolved_checkpoint_path is None:
        raise FileNotFoundError(
            f"Missing model_config.yml in Modlax Orbax checkpoint path: {checkpoint_path}"
        )
    model_config_path = _checkpoint_member_path(resolved_checkpoint_path,
                                                "model_config.yml")
    return _load_yaml_dict(model_config_path)


def _load_modlax_adapter_checkpoint_config(
        checkpoint_path: str) -> dict[str, Any]:
    resolved_checkpoint_path = _resolve_modlax_adapter_orbax_checkpoint_path(
        checkpoint_path)
    if resolved_checkpoint_path is None:
        expected = ", ".join(_MODLAX_ADAPTER_CONFIG_FILENAMES)
        raise FileNotFoundError(
            f"Missing {expected} in Modlax adapter checkpoint path: {checkpoint_path}"
        )
    for config_filename in _MODLAX_ADAPTER_CONFIG_FILENAMES:
        adapter_config_path = _checkpoint_member_path(resolved_checkpoint_path,
                                                      config_filename)
        if _path_exists(adapter_config_path):
            return _normalize_modlax_adapter_config(
                _load_yaml_dict(adapter_config_path),
                f"adapter checkpoint config {adapter_config_path}",
            )
    raise AssertionError(
        "adapter checkpoint path resolved without a config file")


def _get_model_axis_name(mesh: Mesh) -> str:
    if "model" in mesh.axis_names:
        return "model"
    if "tp" in mesh.axis_names:
        return "tp"
    if not mesh.axis_names:
        raise ValueError(
            "Mesh has no axis names; cannot shard Modlax checkpoint restore template."
        )
    return str(mesh.axis_names[-1])


def _build_modlax_llama_restore_template(
    checkpoint_config: dict[str, Any],
    mesh: Mesh,
) -> dict[str, Any]:
    model_axis = _get_model_axis_name(mesh)
    param_dtype = _canonical_jax_dtype(
        str(checkpoint_config.get("dtype", "bfloat16")))

    hidden_size = int(checkpoint_config["hidden_size"])
    intermediate_size = int(checkpoint_config["intermediate_size"])
    num_layers = int(checkpoint_config["num_hidden_layers"])
    num_attention_heads = int(checkpoint_config["num_attention_heads"])
    num_key_value_heads = int(checkpoint_config["num_key_value_heads"])
    head_dim = int(checkpoint_config["head_dim"])
    vocab_size = int(checkpoint_config["vocab_size"])
    tie_word_embeddings = bool(
        checkpoint_config.get("tie_word_embeddings", True))
    num_routing_experts = checkpoint_config.get("num_routing_experts")
    uses_routing = bool(num_routing_experts) or any(
        bool(checkpoint_config.get(flag, False))
        for flag in _MODLAX_ROUTING_FLAGS)
    if uses_routing:
        raise NotImplementedError(
            "Modlax Orbax loading currently supports non-routing Llama checkpoints only."
        )

    adapter_config = get_single_modlax_adapter_config(
        checkpoint_config.get("adapters"),
        "Modlax checkpoint adapters",
    )
    adapter_hidden_size = None
    if adapter_config is not None:
        adapter_hidden_size = int(float(adapter_config["iffm"]) * hidden_size)

    def _leaf(shape: tuple[int, ...], spec: P) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            shape=shape,
            dtype=param_dtype,
            sharding=NamedSharding(mesh, spec),
        )

    template: dict[str, Any] = {
        "token_embed": _leaf((vocab_size, hidden_size), P(None, model_axis)),
    }
    if not tie_word_embeddings:
        template["lm_head"] = {
            "weight": _leaf((vocab_size, hidden_size), P(None, model_axis)),
        }

    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    for layer_index in range(num_layers):
        layer_key = f"layers_{layer_index}"
        layer_entry: dict[str,
                          Any] = {
                              "input_layernorm": {
                                  "weight": _leaf((hidden_size, ), P(None)),
                              },
                              "self_attn": {
                                  "q_proj": {
                                      "kernel":
                                      _leaf((hidden_size, q_size),
                                            P(None, model_axis)),
                                  },
                                  "k_proj": {
                                      "kernel":
                                      _leaf((hidden_size, kv_size),
                                            P(None, model_axis)),
                                  },
                                  "v_proj": {
                                      "kernel":
                                      _leaf((hidden_size, kv_size),
                                            P(None, model_axis)),
                                  },
                                  "o_proj": {
                                      "kernel":
                                      _leaf((q_size, hidden_size),
                                            P(model_axis, None)),
                                  },
                              },
                              "post_attention_layernorm": {
                                  "weight": _leaf((hidden_size, ), P(None)),
                              },
                              "mlp": {
                                  "gate": {
                                      "kernel":
                                      _leaf((hidden_size, intermediate_size),
                                            P(None, model_axis)),
                                  },
                                  "up": {
                                      "kernel":
                                      _leaf((hidden_size, intermediate_size),
                                            P(None, model_axis)),
                                  },
                                  "down": {
                                      "kernel":
                                      _leaf((intermediate_size, hidden_size),
                                            P(model_axis, None)),
                                  },
                              },
                          }

        if adapter_config is not None and adapter_hidden_size is not None:
            adapter_name = str(adapter_config["name"])
            layer_entry["adapters"] = {
                adapter_name: {
                    "attention": {
                        "in_proj": {
                            "kernel":
                            _leaf((hidden_size, adapter_hidden_size),
                                  P(None, model_axis)),
                        },
                        "out_proj": {
                            "kernel":
                            _leaf((adapter_hidden_size, hidden_size),
                                  P(model_axis, None)),
                        },
                    },
                    "mlp": {
                        "in_proj": {
                            "kernel":
                            _leaf((hidden_size, adapter_hidden_size),
                                  P(None, model_axis)),
                        },
                        "out_proj": {
                            "kernel":
                            _leaf((adapter_hidden_size, hidden_size),
                                  P(model_axis, None)),
                        },
                    },
                }
            }

        template[layer_key] = layer_entry

    template["final_norm"] = {
        "weight": _leaf((hidden_size, ), P(None)),
    }
    return template


def _iter_modlax_llama_hf_weights(
    params: dict[str, Any],
    checkpoint_config: dict[str, Any],
) -> Generator[tuple[str, jax.Array], None, None]:
    num_layers = int(checkpoint_config["num_hidden_layers"])
    tie_word_embeddings = bool(
        checkpoint_config.get("tie_word_embeddings", True))
    adapter_config = get_single_modlax_adapter_config(
        checkpoint_config.get("adapters"),
        "Modlax checkpoint adapters",
    )

    yield "model.embed_tokens.weight", params["token_embed"]
    for layer_index in range(num_layers):
        layer_key = f"layers_{layer_index}"
        layer_params = params[layer_key]
        prefix = f"model.layers.{layer_index}."
        yield f"{prefix}input_layernorm.weight", layer_params[
            "input_layernorm"]["weight"]

        attn_params = layer_params["self_attn"]
        yield f"{prefix}self_attn.q_proj.weight", jnp.transpose(
            attn_params["q_proj"]["kernel"])
        yield f"{prefix}self_attn.k_proj.weight", jnp.transpose(
            attn_params["k_proj"]["kernel"])
        yield f"{prefix}self_attn.v_proj.weight", jnp.transpose(
            attn_params["v_proj"]["kernel"])
        yield f"{prefix}self_attn.o_proj.weight", jnp.transpose(
            attn_params["o_proj"]["kernel"])

        yield (f"{prefix}post_attention_layernorm.weight",
               layer_params["post_attention_layernorm"]["weight"])

        mlp_params = layer_params["mlp"]
        yield f"{prefix}mlp.gate_proj.weight", jnp.transpose(
            mlp_params["gate"]["kernel"])
        yield f"{prefix}mlp.up_proj.weight", jnp.transpose(
            mlp_params["up"]["kernel"])
        yield f"{prefix}mlp.down_proj.weight", jnp.transpose(
            mlp_params["down"]["kernel"])

        if adapter_config is not None:
            adapter_name = str(adapter_config["name"])
            adapters_params = layer_params.get("adapters")
            if adapters_params is None or adapter_name not in adapters_params:
                raise KeyError(
                    "Modlax checkpoint config specifies adapter "
                    f"{adapter_name!r}, but adapter parameters are missing.")
            adapter_params = adapters_params[adapter_name]
            attn_adapter = adapter_params["attention"]
            mlp_adapter = adapter_params["mlp"]
            yield (
                f"{prefix}attn_adapter.in_proj.weight",
                attn_adapter["in_proj"]["kernel"],
            )
            yield (
                f"{prefix}attn_adapter.out_proj.weight",
                attn_adapter["out_proj"]["kernel"],
            )
            yield (
                f"{prefix}mlp_adapter.in_proj.weight",
                mlp_adapter["in_proj"]["kernel"],
            )
            yield (
                f"{prefix}mlp_adapter.out_proj.weight",
                mlp_adapter["out_proj"]["kernel"],
            )

    yield "model.norm.weight", params["final_norm"]["weight"]
    if not tie_word_embeddings:
        lm_head = params.get("lm_head", {}).get("weight")
        if lm_head is None:
            raise KeyError(
                "Expected lm_head weights in non-tied Modlax Llama checkpoint."
            )
        yield "lm_head.weight", lm_head


def _select_modlax_orbax_checkpoint_path(vllm_config: VllmConfig,
                                         model_path: str) -> str | None:
    model_weights_override = getattr(vllm_config.model_config, "model_weights",
                                     None)
    resolved_override = _resolve_modlax_orbax_checkpoint_path(
        model_weights_override)
    if resolved_override is not None:
        return resolved_override
    resolved_model_path = _resolve_modlax_orbax_checkpoint_path(model_path)
    if resolved_model_path is not None:
        return resolved_model_path
    return None


def _model_config_jax_dtype(model_config: Any) -> jnp.dtype:
    dtype = getattr(model_config, "dtype", jnp.bfloat16)
    try:
        return jnp.dtype(dtype)
    except TypeError:
        return _canonical_jax_dtype(str(dtype))


def _runtime_llama_adapter_config(
        vllm_config: VllmConfig) -> dict[str, Any] | None:
    hf_config = vllm_config.model_config.hf_config
    adapters = getattr(hf_config, "adapters", None)
    if adapters is None or not isinstance(adapters, (list, tuple)):
        return None
    return get_single_modlax_adapter_config(
        adapters,
        "runtime Llama hf_config.adapters",
    )


def _validate_checkpoint_runtime_adapter_match(
    vllm_config: VllmConfig,
    checkpoint_adapter_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    runtime_adapter_config = _runtime_llama_adapter_config(vllm_config)
    if checkpoint_adapter_config is not None and runtime_adapter_config is None:
        raise ValueError(
            "Checkpoint contains a Modlax adapter, but the running model was "
            "not configured with an adapter.")
    if (checkpoint_adapter_config is not None
            and runtime_adapter_config is not None
            and not modlax_adapter_configs_match(checkpoint_adapter_config,
                                                 runtime_adapter_config)):
        raise ValueError(
            "Checkpoint adapter does not match running model adapter: "
            f"checkpoint={checkpoint_adapter_config}, "
            f"runtime={runtime_adapter_config}")
    return runtime_adapter_config


def _adapter_hidden_size(hidden_size: int, adapter_config: dict[str,
                                                                Any]) -> int:
    adapter_hidden_size = int(float(adapter_config["iffm"]) * hidden_size)
    if adapter_hidden_size <= 0:
        raise ValueError(
            f"Adapter {adapter_config['name']!r} resolves to hidden size "
            f"{adapter_hidden_size}; increase iffm.")
    return adapter_hidden_size


def _build_modlax_llama_adapter_restore_template(
    vllm_config: VllmConfig,
    adapter_config: dict[str, Any],
    mesh: Mesh,
) -> dict[str, Any]:
    model_axis = _get_model_axis_name(mesh)
    hf_config = vllm_config.model_config.hf_config
    param_dtype = _model_config_jax_dtype(vllm_config.model_config)
    hidden_size = int(hf_config.hidden_size)
    adapter_hidden_size = _adapter_hidden_size(hidden_size, adapter_config)
    num_layers = int(hf_config.num_hidden_layers)
    adapter_name = str(adapter_config["name"])

    def _leaf(shape: tuple[int, ...], spec: P) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            shape=shape,
            dtype=param_dtype,
            sharding=NamedSharding(mesh, spec),
        )

    template: dict[str, Any] = {}
    for layer_index in range(num_layers):
        template[f"layers_{layer_index}"] = {
            "adapters": {
                adapter_name: {
                    "attention": {
                        "in_proj": {
                            "kernel":
                            _leaf((hidden_size, adapter_hidden_size),
                                  P(None, model_axis)),
                        },
                        "out_proj": {
                            "kernel":
                            _leaf((adapter_hidden_size, hidden_size),
                                  P(model_axis, None)),
                        },
                    },
                    "mlp": {
                        "in_proj": {
                            "kernel":
                            _leaf((hidden_size, adapter_hidden_size),
                                  P(None, model_axis)),
                        },
                        "out_proj": {
                            "kernel":
                            _leaf((adapter_hidden_size, hidden_size),
                                  P(model_axis, None)),
                        },
                    },
                }
            }
        }
    return template


def _iter_modlax_llama_adapter_hf_weights(
    params: dict[str, Any],
    adapter_config: dict[str, Any],
    num_layers: int,
) -> Generator[tuple[str, jax.Array], None, None]:
    adapter_name = str(adapter_config["name"])
    for layer_index in range(num_layers):
        layer_params = params[f"layers_{layer_index}"]
        adapters_params = layer_params.get("adapters")
        if adapters_params is None or adapter_name not in adapters_params:
            raise KeyError(
                f"Adapter checkpoint is missing layer {layer_index} adapter "
                f"{adapter_name!r}.")
        adapter_params = adapters_params[adapter_name]
        prefix = f"model.layers.{layer_index}."
        attn_adapter = adapter_params["attention"]
        mlp_adapter = adapter_params["mlp"]
        yield f"{prefix}attn_adapter.in_proj.weight", attn_adapter["in_proj"][
            "kernel"]
        yield f"{prefix}attn_adapter.out_proj.weight", attn_adapter[
            "out_proj"]["kernel"]
        yield f"{prefix}mlp_adapter.in_proj.weight", mlp_adapter["in_proj"][
            "kernel"]
        yield f"{prefix}mlp_adapter.out_proj.weight", mlp_adapter["out_proj"][
            "kernel"]


def _zero_initialize_runtime_llama_adapter_weights(
    vllm_config: VllmConfig,
    params: nnx.State,
    shardings: Any,
    metadata_map: "MetadataMap",
    mesh: Mesh,
    runtime_adapter_config: dict[str, Any] | None,
    keep_hf_weight_suffix_when_match: list[str],
    pp_missing_layers: list[str] | None = None,
) -> None:
    if runtime_adapter_config is None:
        return

    hf_config = vllm_config.model_config.hf_config
    hidden_size = int(hf_config.hidden_size)
    adapter_hidden_size = _adapter_hidden_size(hidden_size,
                                               runtime_adapter_config)
    param_dtype = _model_config_jax_dtype(vllm_config.model_config)
    num_layers = int(hf_config.num_hidden_layers)
    shapes = {
        "attn_adapter.in_proj.weight": (hidden_size, adapter_hidden_size),
        "attn_adapter.out_proj.weight": (adapter_hidden_size, hidden_size),
        "mlp_adapter.in_proj.weight": (hidden_size, adapter_hidden_size),
        "mlp_adapter.out_proj.weight": (adapter_hidden_size, hidden_size),
    }
    for layer_index in range(num_layers):
        for suffix, shape in shapes.items():
            _load_and_shard_weight(
                vllm_config,
                params,
                shardings,
                metadata_map,
                mesh,
                f"model.layers.{layer_index}.{suffix}",
                jnp.zeros(shape, dtype=param_dtype),
                keep_original_dtype_keys_regex=None,
                pp_missing_layers=pp_missing_layers,
                keep_hf_weight_suffix_when_match=
                keep_hf_weight_suffix_when_match,
            )


def _has_unloaded_runtime_llama_adapter_weights(
    params: nnx.State,
    runtime_adapter_config: dict[str, Any] | None,
    num_layers: int,
) -> bool:
    if runtime_adapter_config is None:
        return False
    suffixes = (
        "attn_adapter.in_proj.kernel",
        "attn_adapter.out_proj.kernel",
        "mlp_adapter.in_proj.kernel",
        "mlp_adapter.out_proj.kernel",
    )
    for layer_index in range(num_layers):
        for suffix in suffixes:
            try:
                param = get_param(params,
                                  f"model.layers.{layer_index}.{suffix}")
            except ValueError:
                continue
            if isinstance(param, nnx.Param) and isinstance(
                    param.value, jax.ShapeDtypeStruct):
                return True
    return False


def _load_modlax_orbax_llama_weights(
    vllm_config: VllmConfig,
    params: nnx.State,
    shardings: Any,
    metadata_map: "MetadataMap",
    mesh: Mesh,
    checkpoint_path: str,
    keep_hf_weight_suffix_when_match: list[str],
    filter_regex: Optional[str] = None,
    keep_original_dtype_keys_regex: Optional[list[str]] = None,
    pp_missing_layers: list[str] | None = None,
) -> None:
    architectures = getattr(vllm_config.model_config.hf_config,
                            "architectures", [])
    if "LlamaForCausalLM" not in architectures:
        raise NotImplementedError(
            "Modlax Orbax loading currently supports LlamaForCausalLM only.")

    resolved_checkpoint_path = _resolve_modlax_orbax_checkpoint_path(
        checkpoint_path)
    if resolved_checkpoint_path is None:
        raise FileNotFoundError(
            f"Modlax Orbax checkpoint markers were not found at: {checkpoint_path}"
        )

    checkpoint_config = _load_modlax_orbax_checkpoint_config(
        resolved_checkpoint_path)
    checkpoint_adapter_config = get_single_modlax_adapter_config(
        checkpoint_config.get("adapters"),
        "Modlax checkpoint adapters",
    )
    runtime_adapter_config = _validate_checkpoint_runtime_adapter_match(
        vllm_config, checkpoint_adapter_config)
    restore_template = _build_modlax_llama_restore_template(
        checkpoint_config, mesh)
    ocp = _import_orbax_checkpoint()
    logger.info("Loading Modlax Orbax Llama checkpoint from %s",
                resolved_checkpoint_path)
    with ocp.StandardCheckpointer() as checkpointer:
        restored_params = checkpointer.restore(resolved_checkpoint_path,
                                               target=restore_template)

    # Ensure arrays are materialized before transformation.
    restored_params = jax.tree_util.tree_map(jnp.asarray, restored_params)

    for hf_key, hf_weight in _iter_modlax_llama_hf_weights(
            restored_params, checkpoint_config):
        if filter_regex and not re.match(filter_regex, hf_key):
            continue
        _load_and_shard_weight(
            vllm_config,
            params,
            shardings,
            metadata_map,
            mesh,
            hf_key,
            hf_weight,
            keep_original_dtype_keys_regex=keep_original_dtype_keys_regex,
            pp_missing_layers=pp_missing_layers,
            keep_hf_weight_suffix_when_match=keep_hf_weight_suffix_when_match,
        )

    if checkpoint_adapter_config is None:
        _zero_initialize_runtime_llama_adapter_weights(
            vllm_config=vllm_config,
            params=params,
            shardings=shardings,
            metadata_map=metadata_map,
            mesh=mesh,
            runtime_adapter_config=runtime_adapter_config,
            pp_missing_layers=pp_missing_layers,
            keep_hf_weight_suffix_when_match=keep_hf_weight_suffix_when_match,
        )


def load_modlax_orbax_llama_adapter_weights(
    vllm_config: VllmConfig,
    params: nnx.State,
    shardings: Any,
    metadata_map: "MetadataMap",
    mesh: Mesh,
    checkpoint_path: str,
    keep_hf_weight_suffix_when_match: list[str],
    pp_missing_layers: list[str] | None = None,
) -> None:
    architectures = getattr(vllm_config.model_config.hf_config,
                            "architectures", [])
    if "LlamaForCausalLM" not in architectures:
        raise NotImplementedError(
            "Modlax adapter-only loading currently supports LlamaForCausalLM only."
        )

    resolved_checkpoint_path = _resolve_modlax_adapter_orbax_checkpoint_path(
        checkpoint_path)
    if resolved_checkpoint_path is None:
        raise FileNotFoundError(
            "Modlax adapter checkpoint markers were not found at: "
            f"{checkpoint_path}")

    adapter_config = _load_modlax_adapter_checkpoint_config(
        resolved_checkpoint_path)
    runtime_adapter_config = _validate_checkpoint_runtime_adapter_match(
        vllm_config, adapter_config)
    if runtime_adapter_config is None:
        raise ValueError(
            "Adapter-only reload requires the running model to be configured "
            "with exactly one Modlax adapter.")

    restore_template = _build_modlax_llama_adapter_restore_template(
        vllm_config, adapter_config, mesh)
    ocp = _import_orbax_checkpoint()
    logger.info("Loading Modlax Orbax Llama adapter checkpoint from %s",
                resolved_checkpoint_path)
    with ocp.StandardCheckpointer() as checkpointer:
        restored_params = checkpointer.restore(resolved_checkpoint_path,
                                               target=restore_template)

    restored_params = jax.tree_util.tree_map(jnp.asarray, restored_params)
    num_layers = int(vllm_config.model_config.hf_config.num_hidden_layers)
    for hf_key, hf_weight in _iter_modlax_llama_adapter_hf_weights(
            restored_params, adapter_config, num_layers):
        _load_and_shard_weight(
            vllm_config,
            params,
            shardings,
            metadata_map,
            mesh,
            hf_key,
            hf_weight,
            keep_original_dtype_keys_regex=None,
            pp_missing_layers=pp_missing_layers,
            keep_hf_weight_suffix_when_match=keep_hf_weight_suffix_when_match,
        )


@dataclass
class MetadataMap:
    name_map: dict[str, str] = field(default_factory=dict)
    transpose_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    reshape_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    bias_reshape_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    pad_map: dict[str, tuple[int, ...]] = field(default_factory=dict)
    bias_pad_map: dict[str, tuple[int, ...]] = field(default_factory=dict)


############ START Used by llama4, deepseek only for now START ############


def print_param_info(param: nnx.Param, name: str):
    logger.warning(f"Global shape for {name}: {param.value.shape}")
    logger.warning(f"Sharding for {name}: {param.sharding}")

    logger.warning(
        f"Shape of {name} on a single device: {param.value.addressable_shards[0].data.shape}"
    )


def transpose_params(param_key: str, param_tensor: jax.Array, transpose_map):
    for key, value in transpose_map.items():
        if key in param_key:
            return jnp.transpose(param_tensor, value)
    return param_tensor  # Base case / no-op


def reshape_params(param_key: str, param_tensor: jax.Array, shape_map):
    for key, new_shape in shape_map.items():
        if key in param_key:
            try:
                #TODO:(gpolovets) Add validation on whether reshape preserves data layout.
                return jnp.reshape(param_tensor, new_shape)
            except TypeError:
                raise TypeError(
                    f"Cannot reshape for key={key}, new_shape={new_shape}, param_shape={param_tensor.shape}"
                )
    return param_tensor  # Base case / no-op


def model_file_generator(
        model_name_or_path: str,
        download_dir: Optional[str]) -> Generator[str, None, None]:
    weights_files = get_model_weights_files(model_name_or_path, download_dir)
    for st_file in weights_files:
        yield st_file


def model_weights_generator(
    model_name_or_path: str,
    framework: str,
    filter_regex: Optional[str] = None,
    download_dir: Optional[str] = None,
) -> Generator[tuple, None, None]:
    for st_file in model_file_generator(model_name_or_path, download_dir):
        for name, weight_tensor in model_weights_single_file_generator(
                st_file, framework, filter_regex):
            yield name, weight_tensor


def convert_torch_to_jax_with_view(loaded_weight: torch.Tensor,
                                   cast_type: jnp.dtype) -> jax.Array:
    """
    Converts a PyTorch tensor to a JAX array by reinterpreting its
    bit representation using a dtype view map.
    """
    torch_view_type = DTYPE_VIEW_MAP.get(jnp.dtype(cast_type))
    loaded_weight = jnp.array(
        loaded_weight.view(torch_view_type).numpy()).view(cast_type)
    return loaded_weight


############ END Used by llama4, deepseek only for now END ############


def get_model_weights_files(
        model_name_or_path: str,
        download_dir: Optional[str]) -> tuple[list[str], str]:
    """
    Helper to get weight files and their location.
    """

    if os.path.isdir(model_name_or_path):
        logger.info(f"Found weights from local: {model_name_or_path}")
        weights_files = glob.glob(
            os.path.join(model_name_or_path, HF_WEIGHTS_FORMAT))
    elif file_utils.is_hf_repo(model_name_or_path):
        logger.info(f"Downloading weights from HF {model_name_or_path}")
        weights_files = file_utils.download_model_weights_from_hf(
            model_name_or_path, download_dir, HF_WEIGHTS_FORMAT)
    else:
        raise ValueError(
            f"{model_name_or_path} must be a local directory, or a Huggingface model id."
        )

    if not weights_files:
        raise RuntimeError(
            f"Cannot find any {HF_WEIGHTS_FORMAT} files in {model_name_or_path}."
        )

    weights_files.sort()
    return weights_files


def model_weights_single_file_generator(
    weights_file: str,
    framework: str,
    filter_regex: Optional[str] = None,
) -> Generator[tuple, None, None]:
    logger.info(f"Loading weights from {weights_file}")
    # NOTE: We enforce loading tensors on CPU here.
    # Because otherwise the tensor will be loaded on TPU:0 by default,
    # although the tensor would eventually be sharded across multiple TPUs,
    # it would lead to OOM on TPU:0 for large models.
    with jax.default_device(jax.devices("cpu")[0]):
        with safe_open(weights_file, framework=framework) as f:
            for name in f.keys():
                if filter_regex is not None and not re.match(
                        filter_regex, name):
                    continue
                weight_tensor = f.get_tensor(name)
                yield name, weight_tensor


def get_param(params: nnx.State, path: str) -> nnx.State:
    keys = path.split(".")
    plevel = params
    for key in keys:
        if key.isdigit():
            plevel = plevel[int(key)]
        else:
            if key in plevel:
                plevel = plevel[key]
            else:
                raise ValueError(f"{path} is not a valid param path")
    return plevel


def get_param_and_sharding(params: nnx.State, shardings: Any,
                           path: str) -> tuple[nnx.State, nnx.State]:
    keys = path.split(".")
    plevel = params
    slevel = shardings
    for key in keys:
        if key.isdigit():
            plevel = plevel[int(key)]
            slevel = slevel[int(key)]
        else:
            if key in plevel:
                plevel = plevel[key]
                slevel = slevel[key]
            else:
                raise ValueError(f"{path} is not a valid param path")
    return plevel, slevel.value


def shard_put(x: jax.Array, shardings,
              mesh: jax.sharding.Mesh | None) -> jax.Array:
    # Single device sharding requires this special handling
    # to avoid the recursive jit error.
    if mesh is None:
        return jax.device_put(x, shardings) if isinstance(shardings, P) \
            else jax.device_put(x, P(*shardings))

    if math.prod(mesh.axis_sizes) == 1:
        return jax.device_put(x, mesh.devices.flatten()[0])

    if isinstance(shardings, tuple):
        return jax.device_put(x, NamedSharding(mesh, P(*shardings)))
    else:
        return jax.device_put(x, shardings)


def get_default_maps(model_config, mesh: Mesh,
                     name_map: dict[str, str]) -> MetadataMap:
    """Load weights from one model weights file to the model, run on single thread."""
    sharding_size = mesh.shape["model"]

    hf_config = model_config.hf_config

    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    hidden_size = model_config.get_hidden_size()

    # Pad head_dim for kernel performance.
    head_dim_original = model_config.get_head_size()

    reshape_keys: dict[str, tuple[int, ...]] = {
        "q_proj": (num_heads, head_dim_original, hidden_size),
        "k_proj": (num_kv_heads, head_dim_original, hidden_size),
        "v_proj": (num_kv_heads, head_dim_original, hidden_size),
        "o_proj": (hidden_size, num_heads, head_dim_original),
    }
    bias_reshape_keys: dict[str, tuple[int, ...]] = {
        "q_proj.bias": (num_heads, head_dim_original),
        "k_proj.bias": (num_kv_heads, head_dim_original),
        "v_proj.bias": (num_kv_heads, head_dim_original)
    }
    transpose_keys: dict[str, tuple[int, ...]] = {
        "lm_head": (1, 0),
        "fc": (1, 0),
        "gate_proj": (1, 0),
        "up_proj": (1, 0),
        "down_proj": (1, 0),
        "q_proj": (2, 0, 1),
        "k_proj": (2, 0, 1),
        "v_proj": (2, 0, 1),
        "o_proj": (1, 2, 0),
    }

    # # get vision config
    if model_config.is_multimodal_model:
        # TODO: Wenlong: Do not consider padding for now
        transpose_keys.update({
            "attn.proj": (1, 0),
            "attn.qkv": (1, 0),
            "visual.merger.mlp": (1, 0),
            "visual.patch_embed.proj": (2, 3, 4, 1, 0),
        })

    # key: (padding_dim, padding_size)
    pad_keys: dict[str, tuple[int, ...]] = {
        "q_proj": (1, sharding_size // num_heads),
        "k_proj": (1, sharding_size // num_kv_heads),
        "v_proj": (1, sharding_size // num_kv_heads),
        "o_proj": (0, sharding_size // num_heads),
    }
    bias_pad_keys: dict[str, tuple[int, ...]] = {
        "q_proj.bias": (0, sharding_size // num_heads),
        "k_proj.bias": (0, sharding_size // num_kv_heads),
        "v_proj.bias": (0, sharding_size // num_kv_heads),
    }

    return MetadataMap(name_map=name_map,
                       reshape_map=reshape_keys,
                       bias_reshape_map=bias_reshape_keys,
                       transpose_map=transpose_keys,
                       pad_map=pad_keys,
                       bias_pad_map=bias_pad_keys)


def _load_and_shard_weight(vllm_config,
                           params: nnx.State,
                           shardings: Any,
                           metadata_map: MetadataMap,
                           mesh: Mesh,
                           hf_key: str,
                           hf_weight: jax.Array,
                           keep_hf_weight_suffix_when_match: list[str],
                           keep_original_dtype_keys_regex: list[str]
                           | None = None,
                           pp_missing_layers: list[str] | None = None):
    name_map = metadata_map.name_map
    reshape_keys = metadata_map.reshape_map
    bias_reshape_keys = metadata_map.bias_reshape_map
    transpose_keys = metadata_map.transpose_map
    pad_keys = metadata_map.pad_map
    bias_pad_keys = metadata_map.bias_pad_map

    shard = functools.partial(shard_put, mesh=mesh)

    model_config = vllm_config.model_config

    # Pad head_dim for kernel performance.
    head_dim_original = model_config.get_head_size()
    head_dim = utils.get_padded_head_dim(head_dim_original)
    head_dim_pad = head_dim - head_dim_original

    # Check if the key should retain its original dtype
    keep_original_dtype = False
    if keep_original_dtype_keys_regex:
        for pattern in keep_original_dtype_keys_regex:
            if re.match(pattern, hf_key):
                keep_original_dtype = True
                break

    # Converting to config's dtype
    if not keep_original_dtype and hf_weight.dtype != model_config.dtype:
        logger.warning(
            f"Converting dtype for {hf_key} from {hf_weight.dtype} to {model_config.dtype}"
        )
        hf_weight = hf_weight.astype(model_config.dtype)

    # For tensors whose name matches any string in `keep_hf_weight_suffix_when_match`, the
    # '.weight' suffix in HF keys will be kept.
    # Context: some models are being refactored to have identical parameter names as HF
    # models, so the suffix does not need to be removed for those parameters. Eventually
    # we want to get rid of the ".weight" suffix removal logic altogether.
    # TODO(#1479): remove this argument and related logic after the refactoring is done.
    if hf_key.endswith(".weight") and all(
            substr not in hf_key
            for substr in keep_hf_weight_suffix_when_match):
        hf_key = hf_key.removesuffix(".weight")

    # Find the corresponding model key using the HF key
    if "layers" in hf_key:
        layer_num = re.search(r"layers\.(\d+)", hf_key).group(1)
        layer_key = re.sub(r"layers\.\d+", "layers.*", hf_key)
        model_key = name_map.get(layer_key, layer_key)
        model_key = re.sub(r"layers\.\*", f"layers.{layer_num}", model_key)
    elif "blocks" in hf_key:
        layer_num = re.search(r"blocks\.(\d+)", hf_key).group(1)
        layer_key = re.sub(r"blocks\.\d+", "blocks.*", hf_key)
        model_key = name_map.get(layer_key, layer_key)
        model_key = re.sub(r"blocks\.\*", f"blocks.{layer_num}", model_key)
    else:
        if hf_key not in name_map and hf_key == "lm_head":
            logger.warning(f"Skip loading {hf_key} due to tie_word_embeddings")
            return
        if hf_key not in name_map and "t2d" in hf_key:
            logger.warning(
                f"Skip loading {hf_key} as it's not used in eagle-3 for now")
            return
        model_key = name_map.get(hf_key, hf_key)

    if pp_missing_layers and _is_pp_missing_layer(hf_key, pp_missing_layers):
        logger.warning(
            f"Skip loading {hf_key} as it doesn't belong to this PP stage.")
        return
    model_weight, model_sharding = get_param_and_sharding(
        params, shardings, model_key)

    logger.debug(
        "before transform | "
        f"{hf_key}: {hf_weight.shape} --> {model_key}: {model_weight.value.shape} {model_sharding}"
    )

    if hf_key.endswith(".bias"):
        for key in bias_reshape_keys:
            if key in hf_key:
                hf_weight = jnp.reshape(hf_weight, bias_reshape_keys[key])
                if head_dim_pad > 0:
                    hf_weight = jnp.pad(hf_weight, ((0, 0), (0, head_dim_pad)))
                break
    else:
        for key in reshape_keys:
            if key in hf_key:
                hf_weight = jnp.reshape(hf_weight, reshape_keys[key])
                if head_dim_pad > 0:
                    if "o_proj" in key:
                        hf_weight = jnp.pad(hf_weight, ((0, 0), (0, 0),
                                                        (0, head_dim_pad)))
                    else:
                        hf_weight = jnp.pad(hf_weight,
                                            ((0, 0), (0, head_dim_pad),
                                             (0, 0)))
                break
        for key in transpose_keys:
            if key in hf_key:
                hf_weight = jnp.transpose(hf_weight, transpose_keys[key])
                break

    # Pad num-kv-heads
    if hf_key.endswith(".bias"):
        for key, value in bias_pad_keys.items():
            dim = value[0]
            dim_size = value[1]
            if key in hf_key and dim_size != 0:
                hf_weight = jnp.repeat(hf_weight, dim_size, axis=dim)
                break
    else:
        for key, value in pad_keys.items():
            dim = value[0]
            dim_size = value[1]
            if key in hf_key and dim_size != 0:
                hf_weight = jnp.repeat(hf_weight, dim_size, axis=dim)
                break

    logger.debug(
        "after transform | "
        f"{hf_key}: {hf_weight.shape} --> {model_key}: {model_weight.value.shape} {model_sharding}"
    )

    if head_dim_pad == 0:
        assert model_weight.value.shape == hf_weight.shape, f"{hf_key}: {model_weight.value.shape} != {hf_weight.shape}"

    # Update the model weight
    spec = model_weight.sharding.spec if isinstance(
        model_weight.sharding, NamedSharding) else model_weight.sharding
    model_weight.value = shard(hf_weight, spec)


def _is_pp_missing_layer(hf_key: str, pp_missing_layers: list[str]) -> bool:
    has_digit = any(char.isdigit() for char in hf_key)
    # add the suffix after digits to avoid it matches "layers.10" with "layers.1"
    suffix = "." if has_digit else ""
    return any(f'{pp_missing_layer}{suffix}' in hf_key
               for pp_missing_layer in pp_missing_layers)


def _load_hf_weights_on_thread(
    vllm_config: VllmConfig,
    params: nnx.State,
    metadata_map: "MetadataMap",
    mesh: Mesh,
    weights_file: str,
    keep_hf_weight_suffix_when_match: list[str],
    filter_regex: Optional[str] = None,
    keep_original_dtype_keys_regex: Optional[list[str]] = None,
    pp_missing_layers: list[str] | None = None,
):
    """Loads weights from a single weights file."""
    try:
        shardings = nnx.get_named_sharding(params, mesh)
    except TypeError:
        shardings = params

    for hf_key, hf_weight in model_weights_single_file_generator(
            weights_file, framework="flax", filter_regex=filter_regex):
        _load_and_shard_weight(
            vllm_config,
            params,
            shardings,
            metadata_map,
            mesh,
            hf_key,
            hf_weight,
            keep_original_dtype_keys_regex=keep_original_dtype_keys_regex,
            pp_missing_layers=pp_missing_layers,
            keep_hf_weight_suffix_when_match=keep_hf_weight_suffix_when_match,
        )


def load_hf_weights(
    vllm_config: VllmConfig,
    model: nnx.Module,
    metadata_map: "MetadataMap",
    mesh: Mesh,
    filter_regex: Optional[str] = None,
    is_draft_model: bool = False,
    keep_original_dtype_keys_regex: Optional[list[str]] = None,
    pp_missing_layers: list[str] | None = None,
    keep_hf_weight_suffix_when_match: list[str] = [],
):
    """Load weights into a JAX model from either an iterator or files.

    For tensors whose name matches any string in `keep_hf_weight_suffix_when_match`, the
    '.weight' suffix in HF keys will be kept.
    Some models are being refactored to have identical parameter names as HF models, so the suffix
    does not need to be removed for those parameters. Eventually we want to get rid of
    the ".weight" suffix removal logic altogether.
    TODO(#1479): remove this argument and related logic after the refactoring is done.
    """
    params = nnx.state(model)
    try:
        shardings = nnx.get_named_sharding(params, mesh)
    except TypeError:
        shardings = params
    if is_draft_model:
        model_path = vllm_config.speculative_config.draft_model_config.model
    else:
        model_path = vllm_config.model_config.model

    modlax_checkpoint_path = None
    if not is_draft_model:
        modlax_checkpoint_path = _select_modlax_orbax_checkpoint_path(
            vllm_config, model_path)

    weights_iterator = None
    if hasattr(vllm_config.model_config, "runai_model_weights_iterator"):
        weights_iterator = vllm_config.model_config.runai_model_weights_iterator
    env = torchax.default_env()
    if modlax_checkpoint_path is not None:
        if weights_iterator is not None:
            logger.info(
                "Detected Modlax Orbax checkpoint at %s; bypassing RunAI iterator path.",
                modlax_checkpoint_path)
        _load_modlax_orbax_llama_weights(
            vllm_config=vllm_config,
            params=params,
            shardings=shardings,
            metadata_map=metadata_map,
            mesh=mesh,
            checkpoint_path=modlax_checkpoint_path,
            filter_regex=filter_regex,
            keep_original_dtype_keys_regex=keep_original_dtype_keys_regex,
            pp_missing_layers=pp_missing_layers,
            keep_hf_weight_suffix_when_match=keep_hf_weight_suffix_when_match,
        )
    # The weights_iterator is used in RunAI model streamer integration.
    elif weights_iterator is not None:
        for hf_key, hf_weight in weights_iterator:
            if filter_regex and not re.match(filter_regex, hf_key):
                continue

            # Since the weights_iterator yields Pytorch tensors (torch.Tensor),
            # we need to convert them to JAX arrays (jax.Array).
            hf_weight_jax = env.t2j_copy(hf_weight)

            _load_and_shard_weight(
                vllm_config,
                params,
                shardings,
                metadata_map,
                mesh,
                hf_key,
                hf_weight_jax,
                keep_original_dtype_keys_regex=keep_original_dtype_keys_regex,
                pp_missing_layers=pp_missing_layers,
                keep_hf_weight_suffix_when_match=
                keep_hf_weight_suffix_when_match,
            )
    else:
        # File-based path (multi-threaded)
        weights_files = get_model_weights_files(
            model_path, vllm_config.load_config.download_dir)
        max_workers = min(64, len(weights_files))
        # NOTE(xiang): Disable multi-threading mode if running on multi-host.
        # Because multi-threading would cause different JAX processes to load
        # different weights at the same time.
        if envs.TPU_MULTIHOST_BACKEND == "ray":
            max_workers = 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _load_hf_weights_on_thread,
                    vllm_config,
                    params,
                    metadata_map,
                    mesh,
                    weights_file,
                    filter_regex=filter_regex,
                    keep_original_dtype_keys_regex=
                    keep_original_dtype_keys_regex,
                    pp_missing_layers=pp_missing_layers,
                    keep_hf_weight_suffix_when_match=
                    keep_hf_weight_suffix_when_match,
                ) for weights_file in weights_files
            ]
            for future in futures:
                future.result()

    if not is_draft_model and "LlamaForCausalLM" in getattr(
            vllm_config.model_config.hf_config, "architectures", []):
        runtime_adapter_config = _runtime_llama_adapter_config(vllm_config)
        if _has_unloaded_runtime_llama_adapter_weights(
                params,
                runtime_adapter_config,
                int(vllm_config.model_config.hf_config.num_hidden_layers),
        ):
            _zero_initialize_runtime_llama_adapter_weights(
                vllm_config=vllm_config,
                params=params,
                shardings=shardings,
                metadata_map=metadata_map,
                mesh=mesh,
                runtime_adapter_config=runtime_adapter_config,
                pp_missing_layers=pp_missing_layers,
                keep_hf_weight_suffix_when_match=
                keep_hf_weight_suffix_when_match,
            )

    check_all_loaded(params)
    nnx.update(model, params)


def check_all_loaded(params: nnx.State):

    def _check(x: Any):
        if isinstance(x, nnx.Param) and isinstance(x.value,
                                                   jax.ShapeDtypeStruct):
            raise ValueError(f"The param does not load weights: {x}")

    jax.tree.map(_check, params)


def build_flat_dict(flat_state, mappings):
    """Build a new flat dictionary from the flat state using the provided mappings."""
    new_flat_dict = {}
    for keys, v in flat_state:
        path = '.'.join(str(key) for key in keys)
        mapped = False
        for src, (tgt, sharding) in mappings.items():
            regex = "^" + re.escape(tgt).replace("\\.\\*", r"\.(\d+)") + "$"
            matched = re.match(regex, path)
            if matched:
                # Extract wildcards if any
                wildcards = matched.groups()
                src_parts = []
                wc_index = 0
                for part in src.split("."):
                    if part == "*":
                        src_parts.append(wildcards[wc_index])
                        wc_index += 1
                    else:
                        src_parts.append(part)
                actual_src = ".".join(src_parts)
                new_flat_dict[actual_src] = v, sharding
                mapped = True
                break
        if not mapped:
            logger.info(f"!!! No mapping for flat state: {keys}")
    return new_flat_dict


def transfer_state_with_mappings(src_state,
                                 tgt_state,
                                 mappings,
                                 transpose_keys=None,
                                 shard=None):
    """Transfer state from src_state to tgt_state using the provided mappings."""
    src_flat = src_state.flat_state()
    tgt_flat = tgt_state.flat_state()

    new_src_dict = build_flat_dict(tgt_flat, mappings)
    logger.info(f"{mappings=}")
    logger.info(f"{transpose_keys=}")
    for src_keys, v in src_flat:
        flattened_src_keys = '.'.join(str(k) for k in src_keys)
        new_v = jnp.copy(v.value)
        logger.info(
            f"Processing source key: {flattened_src_keys} and value: {new_v.shape} {new_v.dtype}"
        )
        if flattened_src_keys not in new_src_dict:
            logger.info(f"!!! No mapping for source key: {flattened_src_keys}")
            continue
        sharding = new_src_dict[flattened_src_keys][1]

        # E.g. layers.*.attn.k_proj.w, layers.*.attn.k_proj.w_lora_a
        # E.g. layers.*.mlp.down_proj.kernel, layers.*.mlp.down_proj.kernel_lora_a
        if transpose_keys is not None \
          and ((src_keys[-1] in transpose_keys) and ('lora' not in src_keys[-1])):
            v_maybe_t = jnp.transpose(new_v, transpose_keys[src_keys[-1]])
        else:
            v_maybe_t = new_v

        to_update_value = new_src_dict[flattened_src_keys][0].value
        assert to_update_value.shape == v_maybe_t.shape, \
            f"Shape mismatch for {flattened_src_keys}: {to_update_value.shape} vs {v_maybe_t.shape}"

        if to_update_value.dtype != v_maybe_t.dtype:
            logger.info(
                f"Type mismatch between external model and vLLM model. Converting {v_maybe_t.dtype=} to {to_update_value.dtype=}"
            )
            v_maybe_t = v_maybe_t.astype(to_update_value.dtype)

        new_src_dict[flattened_src_keys][0].value = shard(
            v_maybe_t, sharding) if shard else v_maybe_t

    tgt_state = tgt_state.from_flat_path(tgt_flat)
    return tgt_state


class BaseWeightLoader:

    def __init__(self, vllm_config: VllmConfig, **kwargs):
        self.vllm_config = vllm_config
        self.names_and_weights_generator = model_weights_generator(
            model_name_or_path=vllm_config.model_config.model,
            download_dir=vllm_config.load_config.download_dir,
            **kwargs,
        )

    def get_weights_iterator(self):
        weights_iterator = getattr(self.vllm_config.model_config,
                                   "runai_model_weights_iterator", None)
        if weights_iterator:
            return weights_iterator
        else:
            return self.names_and_weights_generator


class StandardWeightLoader(BaseWeightLoader):

    def __init__(self, vllm_config: VllmConfig, mesh: Mesh):
        super().__init__(vllm_config, framework="pt")
        self.vllm_config = vllm_config
        self.mesh = mesh

    def load_weights(self,
                     model: nnx.Module,
                     mappings: dict | MetadataMap,
                     keep_hf_weight_suffix_when_match: list[str] = []):
        """
        Calls the generic load_hf_weights utility, passing the correct
        weights iterator.

        `mappings` can be either a MetadataMap or a dict mapping, if it's
        * a dict, the default MetadataMap will be created with get_default_maps.
        * a MetadataMap, it will be used directly. This is useful for cases
        where caller needs to customize the reshape/transpose/pad maps, e.g.
        update the key of tranpose_map from "q_proj" to "q_proj.weight".

        Context: some models are being refactored to have identical parameter names as HF
        models, so these parameters keeps ".weight" suffix. Eventually
        we want to get rid of the ".weight" suffix removal logic altogether.
        TODO(#1479): remove this argument and related logic after the refactoring is done.
        """
        if isinstance(mappings, MetadataMap):
            metadata_map = mappings
        else:
            metadata_map = get_default_maps(self.vllm_config.model_config,
                                            self.mesh, mappings)

        load_hf_weights(
            vllm_config=self.vllm_config,
            model=model,
            metadata_map=metadata_map,
            mesh=self.mesh,
            pp_missing_layers=getattr(model, 'pp_missing_layers', []),
            keep_hf_weight_suffix_when_match=keep_hf_weight_suffix_when_match)

    def load_adapter_weights(
        self,
        model: nnx.Module,
        mappings: dict | MetadataMap,
        checkpoint_path: str,
        keep_hf_weight_suffix_when_match: list[str] = [],
    ):
        if isinstance(mappings, MetadataMap):
            metadata_map = mappings
        else:
            metadata_map = get_default_maps(self.vllm_config.model_config,
                                            self.mesh, mappings)

        params = nnx.state(model)
        try:
            shardings = nnx.get_named_sharding(params, self.mesh)
        except TypeError:
            shardings = params

        load_modlax_orbax_llama_adapter_weights(
            vllm_config=self.vllm_config,
            params=params,
            shardings=shardings,
            metadata_map=metadata_map,
            mesh=self.mesh,
            checkpoint_path=checkpoint_path,
            pp_missing_layers=getattr(model, 'pp_missing_layers', []),
            keep_hf_weight_suffix_when_match=keep_hf_weight_suffix_when_match,
        )
        check_all_loaded(params)
        nnx.update(model, params)


def load_nnx_param_from_reshaped_torch(
        jax_param: nnx.Param,
        torch_weight: torch.Tensor,
        *,
        reshape_dims: Optional[tuple[int, ...]] = None,
        permute_dims: Optional[tuple[int, ...]] = None,
        param_name: str = "Unknown"):
    """Load a nnx.Param from a torch.Tensor with reshaping and transposing.
    
    HuggingFace model almost always store linear layer weights with contracting dimension
    last, and only support 1D/2D weight tensors. This function reshapes then transposes
    the torch weight to match the jax_param shape before loading.

    Args:
        jax_param: The target nnx.Param to load the weight into.
        torch_weight: The source torch.Tensor weight.
        reshape_dims: Optional tuple specifying the shape to reshape the torch weight to before permutation. If None, no reshaping is applied.
        permute_dims: Optional tuple specifying the permutation of dimensions. If None, no-op for 1D tensors and transpose for 2D tensors is applied.
    """
    if reshape_dims is not None:
        torch_weight = torch_weight.reshape(reshape_dims)
    if permute_dims is None and torch_weight.ndim == 2:
        permute_dims = (1, 0)
    if permute_dims is not None:
        torch_weight = torch_weight.permute(*permute_dims)

    assert tuple(torch_weight.shape) == jax_param.value.shape, \
        f"Shape mismatch when loading weight '{param_name}': torch {torch_weight.shape} vs jax {jax_param.value.shape}"

    spec = jax_param.sharding
    if isinstance(jax_param.sharding, NamedSharding):
        spec = jax_param.sharding.spec
    elif isinstance(jax_param.sharding, SingleDeviceSharding):
        spec = ()
    mesh = getattr(jax_param, 'mesh', None)

    jax_param.value = shard_put(t2j(torch_weight, use_dlpack=False).astype(
        jax_param.value.dtype),
                                spec,
                                mesh=mesh)


class JaxAutoWeightsLoader(AutoWeightsLoader):
    """A weights loader for JAX models."""

    def __init__(self, model, **kwargs):
        assert isinstance(model, JaxModule)

        for name, param in model.named_parameters():
            if not hasattr(param, "weight_loader"):
                # Following are common patterns in standard transformers. To add pattern for modules
                # beyond standard transformers, please consider setting weight_loader.
                reshape_dims = None
                permute_dims = None
                if any(substr in name for substr in
                       ["q_proj.weight", "k_proj.weight", "v_proj.weight"]):
                    D, N, H = param.value.shape
                    reshape_dims = (N, H, D)
                    permute_dims = (2, 0, 1)
                elif any(substr in name for substr in
                         ["q_proj.bias", "k_proj.bias", "v_proj.bias"]):
                    permute_dims = (0, 1)
                elif "o_proj.weight" in name:
                    N, H, D = param.value.shape
                    reshape_dims = (D, N, H)
                    permute_dims = (1, 2, 0)
                elif "embed_tokens.weight" in name:
                    permute_dims = (0, 1)
                elif "lm_head" in name:
                    permute_dims = (1, 0)

                setattr(
                    param, "weight_loader",
                    functools.partial(load_nnx_param_from_reshaped_torch,
                                      reshape_dims=reshape_dims,
                                      permute_dims=permute_dims,
                                      param_name=name))

        super().__init__(model, **kwargs)


class LoadableWithIterator:
    """Mixin for models that support loading weights with an iterator.
    
    This is replicating what vLLM does for most models, e.g. https://github.com/vllm-project/vllm/blob/8e2a469b3b2f67bc900ed72724fe3f05e3564994/vllm/model_executor/models/gemma3_mm.py#L644-L646
    """

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        if not isinstance(weights, Iterable):
            # Use next parent class in MRO.
            return super().load_weights(weights)

        loader = JaxAutoWeightsLoader(self)
        return loader.load_weights(weights)
