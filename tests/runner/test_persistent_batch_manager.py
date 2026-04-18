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

import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from tpu_inference.runner.input_batch import CachedRequestState, InputBatch
from tpu_inference.runner.persistent_batch_manager import \
    PersistentBatchManager


class TestPersistentBatchManager(unittest.TestCase):

    def test_update_states_pp_non_last_rank(self):
        """
        the current rank is not the last rank.

        This test verifies that when new tokens are received from the scheduler,
        the internal state of the PersistentBatchManager (including request
        states and the input batch) is correctly updated.
        """

        req_id = 101
        initial_output_tokens = [10, 20]

        req_state = MagicMock()
        req_state.num_tokens = 2
        req_state.output_token_ids = list(initial_output_tokens)

        requests = {req_id: req_state}

        input_batch = MagicMock()
        input_batch.req_id_to_index = {req_id: 0}
        input_batch.num_prompt_tokens = np.array([2], dtype=np.int32)
        input_batch.token_ids_cpu = np.zeros((1, 10), dtype=np.int32)
        input_batch.num_tokens = np.array([2], dtype=np.int32)
        input_batch.num_tokens_no_spec = np.array([2], dtype=np.int32)
        input_batch.num_reqs = 1

        encoder_cache = MagicMock()
        model_config = MagicMock()

        manager = PersistentBatchManager(requests,
                                         input_batch,
                                         encoder_cache,
                                         False,
                                         model_config,
                                         is_last_rank=False)

        scheduler_output = MagicMock()
        req_data = MagicMock()
        req_data.req_ids = [req_id]
        req_data.num_computed_tokens = [2]
        new_token_id = [30]
        req_data.new_token_ids = [new_token_id]
        req_data.new_block_ids = [None]
        req_data.num_output_tokens = [len(initial_output_tokens) + 1]
        scheduler_output.scheduled_cached_reqs = req_data
        scheduler_output.scheduled_spec_decode_tokens = {}

        manager.update_states(scheduler_output, None)

        expected_output_token_ids = initial_output_tokens + new_token_id
        self.assertEqual(req_state.output_token_ids, expected_output_token_ids)

        np.testing.assert_array_equal(
            manager.input_batch.token_ids_cpu[0, 2:3],
            np.array(new_token_id, dtype=np.int32))

        self.assertEqual(manager.input_batch.num_tokens[0], 3)
        self.assertEqual(manager.input_batch.num_tokens_no_spec[0], 3)

    @pytest.mark.online_rl_server_related
    def test_update_states_recovers_from_stale_finished_request_index(self):
        req_id = "req-1"
        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[1, 2],
            mm_features=[],
            sampling_params=SamplingParams(temperature=0.0),
            pooling_params=None,
            block_ids=[[1]],
            num_computed_tokens=0,
            lora_request=None,
            output_token_ids=[],
        )

        input_batch = InputBatch(
            max_num_reqs=4,
            max_model_len=16,
            max_num_batched_tokens=16,
            pin_memory=False,
            vocab_size=32,
            block_sizes=[16],
        )
        input_batch.add_request(req_state)
        input_batch.req_id_to_index[req_id] = 7

        manager = PersistentBatchManager(
            requests={req_id: req_state},
            input_batch=input_batch,
            encoder_cache={},
            uses_mrope=False,
            model_config=MagicMock(),
            is_last_rank=True,
        )

        scheduler_output = MagicMock()
        scheduler_output.finished_req_ids = [req_id]
        scheduler_output.free_encoder_mm_hashes = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.scheduled_new_reqs = []
        scheduler_output.scheduled_spec_decode_tokens = {}
        scheduler_output.scheduled_cached_reqs = MagicMock(
            req_ids=[],
            num_computed_tokens=[],
            new_block_ids=[],
            resumed_req_ids=set(),
            num_output_tokens=[],
        )

        batch_changed = manager.update_states(scheduler_output, None)

        self.assertFalse(batch_changed)
        self.assertEqual(input_batch.num_reqs, 0)
