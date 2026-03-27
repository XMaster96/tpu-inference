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

from unittest.mock import MagicMock

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec

import tpu_inference.models.vllm.vllm_model_wrapper as vllm_model_wrapper
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName)


def test_jit_step_func_shards_hidden_states_over_attention_data(monkeypatch):
    monkeypatch.setenv("NEW_MODEL_DESIGN", "1")

    captured = {}

    def fake_jit(*args, **kwargs):
        captured["out_shardings"] = kwargs["out_shardings"]

        def decorator(fn):
            return fn

        return decorator

    monkeypatch.setattr(vllm_model_wrapper.jax, "jit", fake_jit)

    wrapper = vllm_model_wrapper.VllmModelWrapper.__new__(
        vllm_model_wrapper.VllmModelWrapper)
    wrapper.mesh = Mesh(np.array(jax.local_devices()[:1]).reshape((1, 1, 1,
                                                                   1)),
                        MESH_AXIS_NAMES)
    wrapper.vllm_config = MagicMock()

    wrapper.jit_step_func()

    assert captured["out_shardings"][1].spec == PartitionSpec(
        ShardingAxisName.ATTN_DATA, None)
