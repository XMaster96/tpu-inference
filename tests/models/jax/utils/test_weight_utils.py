# Copyright 2026 Google LLC
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

import os
import tempfile
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from safetensors.torch import save_file
from torch import nn
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator,
    MetadataMap,
    _build_modlax_llama_restore_template,
    _is_modlax_orbax_checkpoint,
    _iter_modlax_llama_hf_weights,
    _load_modlax_orbax_checkpoint_config,
    _resolve_modlax_orbax_checkpoint_path,
    load_hf_weights,
)


class TorchMLP(nn.Module):
    """MLP implemented with PyTorch."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(2, 6)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(6, 2, bias=True)

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class JaxMLP(JaxModule, LoadableWithIterator):
    """MLP implemented with JAX."""

    def __init__(self, rngs):
        super().__init__()
        self.w1 = JaxLinear(2, 6, rngs)
        self.act = nnx.relu
        self.w2 = JaxLinear(6, 2, rngs, use_bias=True)

    def __call__(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class TestJaxAutoWeightsLoader:

    def test_load_from_safetensors(self):
        """Load weights from a safetensors file saved from a PyTorch model.
        """
        torch_model = TorchMLP()
        with torch.no_grad():
            torch_model.w1.weight.fill_(1.1)
            torch_model.w2.weight.fill_(0.9)
            torch_model.w2.bias.fill_(0.1)

        # Save the PyTorch model weights to a safetensors file. Load them
        # into the JAX model.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file_path = os.path.join(tmpdir, "model.safetensors")
            save_file(torch_model.state_dict(), tmp_file_path)

            devices = jax.local_devices()
            mesh = Mesh(devices, axis_names=('p', ))
            with jax.set_mesh(mesh):
                jax_model = JaxMLP(rngs=nnx.Rngs(0))

                model_config = MagicMock()
                model_config.quantization = None
                model_config.model = tmpdir
                model_config.revision = None

                loader = get_model_loader(
                    LoadConfig(load_format="safetensors"))
                loader.load_weights(jax_model, model_config)

        np.testing.assert_allclose(torch_model.w1.weight.T.detach().numpy(),
                                   jax_model.w1.weight.value)
        np.testing.assert_allclose(torch_model.w2.weight.T.detach().numpy(),
                                   jax_model.w2.weight.value)

        # Forward pass to verify correctness.
        input_values = [[0.1, 0.2], [0.3, 0.4]]
        torch_input = torch.tensor(input_values)
        jax_input = np.array(input_values)
        torch_output = torch_model(torch_input).detach().numpy()
        jax_output = jax_model(jax_input)
        np.testing.assert_allclose(torch_output,
                                   jax_output,
                                   rtol=1e-3,
                                   atol=1e-2)


class TestModlaxOrbaxLlamaSupport:

    def test_build_modlax_llama_restore_template(self):
        devices = jax.local_devices()
        mesh = Mesh(np.array(devices[:1]), axis_names=("model", ))
        checkpoint_config = {
            "dtype": "bfloat16",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "tie_word_embeddings": False,
            "adapter_iffm": None,
            "num_routing_experts": None,
        }

        template = _build_modlax_llama_restore_template(checkpoint_config, mesh)

        assert template["token_embed"].shape == (32, 8)
        assert template["token_embed"].dtype == jnp.bfloat16
        assert template["token_embed"].sharding.spec == P(None, "model")
        assert template["layers_0"]["self_attn"]["q_proj"]["kernel"].shape == (8, 8)
        assert template["layers_0"]["self_attn"]["o_proj"]["kernel"].shape == (8, 8)
        assert template["layers_0"]["mlp"]["gate"]["kernel"].shape == (8, 16)
        assert template["layers_0"]["mlp"]["down"]["kernel"].shape == (16, 8)
        assert template["final_norm"]["weight"].shape == (8, )
        assert template["lm_head"]["weight"].shape == (32, 8)

    def test_iter_modlax_llama_hf_weights(self):
        checkpoint_config = {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "tie_word_embeddings": False,
            "adapter_iffm": None,
        }
        params = {
            "token_embed": jnp.arange(32 * 8, dtype=jnp.float32).reshape(32, 8),
            "layers_0": {
                "input_layernorm": {
                    "weight": jnp.ones((8, ), dtype=jnp.float32),
                },
                "self_attn": {
                    "q_proj": {
                        "kernel": jnp.arange(8 * 8, dtype=jnp.float32).reshape(8, 8),
                    },
                    "k_proj": {
                        "kernel": jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4),
                    },
                    "v_proj": {
                        "kernel": jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4),
                    },
                    "o_proj": {
                        "kernel": jnp.arange(8 * 8, dtype=jnp.float32).reshape(8, 8),
                    },
                },
                "post_attention_layernorm": {
                    "weight": jnp.ones((8, ), dtype=jnp.float32),
                },
                "mlp": {
                    "gate": {
                        "kernel": jnp.arange(8 * 16, dtype=jnp.float32).reshape(8, 16),
                    },
                    "up": {
                        "kernel": jnp.arange(8 * 16, dtype=jnp.float32).reshape(8, 16),
                    },
                    "down": {
                        "kernel": jnp.arange(16 * 8, dtype=jnp.float32).reshape(16, 8),
                    },
                },
            },
            "final_norm": {
                "weight": jnp.ones((8, ), dtype=jnp.float32),
            },
            "lm_head": {
                "weight": jnp.arange(32 * 8, dtype=jnp.float32).reshape(32, 8),
            },
        }

        hf_weights = dict(_iter_modlax_llama_hf_weights(params, checkpoint_config))
        assert "model.embed_tokens.weight" in hf_weights
        assert "model.layers.0.self_attn.q_proj.weight" in hf_weights
        assert "model.layers.0.self_attn.o_proj.weight" in hf_weights
        assert "model.layers.0.mlp.down_proj.weight" in hf_weights
        assert "model.norm.weight" in hf_weights
        assert "lm_head.weight" in hf_weights

        np.testing.assert_allclose(
            np.asarray(hf_weights["model.layers.0.self_attn.q_proj.weight"]),
            np.asarray(params["layers_0"]["self_attn"]["q_proj"]["kernel"].T),
        )
        np.testing.assert_allclose(
            np.asarray(hf_weights["model.layers.0.self_attn.o_proj.weight"]),
            np.asarray(params["layers_0"]["self_attn"]["o_proj"]["kernel"].T),
        )

    def test_modlax_path_takes_priority_over_runai_iterator(self):
        devices = np.array(jax.local_devices()[:1])
        mesh = Mesh(devices, axis_names=("model", ))
        model = JaxMLP(rngs=nnx.Rngs(0))

        vllm_config = MagicMock()
        vllm_config.model_config.model = "unused-model-path"
        vllm_config.model_config.runai_model_weights_iterator = [
            ("unused.key", torch.ones((1, )))
        ]
        vllm_config.model_config.hf_config.architectures = ["LlamaForCausalLM"]
        vllm_config.load_config.download_dir = None
        vllm_config.speculative_config = None

        with patch(
                "tpu_inference.models.jax.utils.weight_utils._select_modlax_orbax_checkpoint_path",
                return_value="gs://dummy-orbax-checkpoint"), patch(
                    "tpu_inference.models.jax.utils.weight_utils._load_modlax_orbax_llama_weights"
                ) as modlax_loader, patch(
                    "tpu_inference.models.jax.utils.weight_utils._load_and_shard_weight",
                    side_effect=AssertionError(
                        "RunAI iterator path should not be used for Modlax Orbax checkpoints."
                    )):
            load_hf_weights(vllm_config=vllm_config,
                            model=model,
                            metadata_map=MetadataMap(),
                            mesh=mesh)

        modlax_loader.assert_called_once()

    @patch("google.cloud.storage.Client")
    def test_is_modlax_orbax_checkpoint_gcs_fallback(self, mock_storage_client):
        mock_bucket = MagicMock()
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        def _blob(name):
            mock_blob = MagicMock()
            mock_blob.exists.return_value = (
                name.endswith("model_config.yml")
                or name.endswith("_CHECKPOINT_METADATA"))
            return mock_blob

        mock_bucket.blob.side_effect = _blob

        with patch("tpu_inference.models.jax.utils.weight_utils.epath.Path",
                   side_effect=RuntimeError("no gcs filesystem backend")):
            assert _is_modlax_orbax_checkpoint(
                "gs://bucket/some/checkpoint/path")

    @patch("google.cloud.storage.Client")
    def test_load_modlax_orbax_checkpoint_config_gcs_fallback(
            self, mock_storage_client):
        mock_bucket = MagicMock()
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client

        model_config_yaml = "\n".join([
            "hidden_size: 16",
            "intermediate_size: 32",
            "num_hidden_layers: 1",
            "num_attention_heads: 2",
            "num_key_value_heads: 2",
            "head_dim: 8",
            "vocab_size: 32000",
            "max_position_embeddings: 2048",
            "rope_theta: 10000.0",
        ])

        def _blob(name):
            mock_blob = MagicMock()
            mock_blob.exists.return_value = (
                name.endswith("model_config.yml")
                or name.endswith("_CHECKPOINT_METADATA"))
            if name.endswith("model_config.yml"):
                mock_blob.download_as_text.return_value = model_config_yaml
            return mock_blob

        mock_bucket.blob.side_effect = _blob

        with patch("tpu_inference.models.jax.utils.weight_utils.epath.Path",
                   side_effect=RuntimeError("no gcs filesystem backend")):
            config = _load_modlax_orbax_checkpoint_config(
                "gs://bucket/some/checkpoint/path")
        assert config["hidden_size"] == 16

    def test_resolve_modlax_orbax_checkpoint_from_parent_path(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "ckpt")
            final_model_path = os.path.join(checkpoint_path, "final_model")
            os.makedirs(final_model_path, exist_ok=True)

            with open(os.path.join(final_model_path, "_CHECKPOINT_METADATA"),
                      "w",
                      encoding="utf-8") as f:
                f.write("{}")
            with open(os.path.join(final_model_path, "model_config.yml"),
                      "w",
                      encoding="utf-8") as f:
                f.write("hidden_size: 16\n")

            resolved_path = _resolve_modlax_orbax_checkpoint_path(
                checkpoint_path)
            assert resolved_path == final_model_path

    def test_resolve_modlax_orbax_checkpoint_from_final_model_path(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "ckpt")
            os.makedirs(checkpoint_path, exist_ok=True)

            with open(os.path.join(checkpoint_path, "_CHECKPOINT_METADATA"),
                      "w",
                      encoding="utf-8") as f:
                f.write("{}")
            with open(os.path.join(checkpoint_path, "model_config.yml"),
                      "w",
                      encoding="utf-8") as f:
                f.write("hidden_size: 16\n")

            resolved_path = _resolve_modlax_orbax_checkpoint_path(
                os.path.join(checkpoint_path, "final_model"))
            assert resolved_path == checkpoint_path
