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

from tpu_inference.layers.common.sharding import ShardingAxisName


def test_sharding_axis_names_follow_new_model_design_env(monkeypatch):
    monkeypatch.delenv("NEW_MODEL_DESIGN", raising=False)
    monkeypatch.delenv("USE_2D_TP", raising=False)
    assert ShardingAxisName.ATTN_DATA == "data"
    assert ShardingAxisName.ATTN_HEAD == "model"

    monkeypatch.setenv("NEW_MODEL_DESIGN", "1")
    assert ShardingAxisName.ATTN_DATA == ("data", "attn_dp")
    assert ShardingAxisName.ATTN_HEAD == ("model", "expert")
