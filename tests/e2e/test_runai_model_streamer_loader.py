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

# This file contains end-to-end tests for the RunAI Model Streamer loader.
#
# The RunAI Model Streamer is a high-performance model loader that serves as an
# alternative to the default Hugging Face loader. Instead of downloading a model
# to local disk, it streams the weights from object storage (like GCS) into
# GPU memory. This streaming process is significantly faster than the
# traditional disk-based loading method.

# The tests in this file verify that loading model weights using the
# streamer produces the same results as loading the same model using the
# standard Hugging Face loader. This ensures the correctness of the streamer
# integration.

# The tests are performed by:
# 1. Loading a model from Google Cloud Storage using the `runai_streamer` format.
# 2. Generating output with this model.
# 3. Loading the same model from Hugging Face using the default loader.
# 4. Generating output with this second model.
# 5. Asserting that the outputs from both models are identical.

from __future__ import annotations

import time

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = (
    "/home/jan/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/"
    "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)


@pytest.fixture
def sampling_config():
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=True)


def test_correctness_jax_uni_proc_executor(
    sampling_config: SamplingParams,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Compare the outputs of a model loaded from a local path via
    runai_model_streamer and the default HF loader. The outputs should be the
    same.
    '''
    prompt = "Hello, my name is"

    streamer_llm = LLM(model=MODEL_NAME,
                       load_format="runai_streamer",
                       max_model_len=128,
                       max_num_seqs=16,
                       max_num_batched_tokens=256)
    streamer_outputs = streamer_llm.generate([prompt], sampling_config)
    streamer_output_text = streamer_outputs[0].outputs[0].text
    del streamer_llm
    time.sleep(10)  # Wait for TPUs to be released

    # Test with Hugging Face model
    hf_llm = LLM(model=MODEL_NAME,
                 max_model_len=128,
                 max_num_seqs=16,
                 max_num_batched_tokens=256)
    hf_outputs = hf_llm.generate([prompt], sampling_config)
    hf_output_text = hf_outputs[0].outputs[0].text
    del hf_llm
    time.sleep(10)  # Wait for TPUs to be released

    assert streamer_output_text == hf_output_text, (
        f"Outputs do not match! "
        f"Streamer output: {streamer_output_text}, HF output: {hf_output_text}")


def test_correctness_torchax_uni_proc_executor(
    sampling_config: SamplingParams,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Compare the outputs of a model loaded from a local path via
    runai_model_streamer and a model loaded via the default loader.
    """
    prompt = "def fibonacci("

    streamer_llm = LLM(model=MODEL_NAME,
                       load_format="runai_streamer",
                       max_model_len=128,
                       max_num_seqs=16,
                       max_num_batched_tokens=256)
    streamer_outputs = streamer_llm.generate([prompt], sampling_config)
    streamer_output_text = streamer_outputs[0].outputs[0].text
    del streamer_llm
    time.sleep(10)  # Wait for TPUs to be released

    # Test with Hugging Face model
    hf_llm = LLM(model=MODEL_NAME,
                 max_model_len=128,
                 max_num_seqs=16,
                 max_num_batched_tokens=256)
    hf_outputs = hf_llm.generate([prompt], sampling_config)
    hf_output_text = hf_outputs[0].outputs[0].text
    del hf_llm
    time.sleep(10)  # Wait for TPUs to be released

    assert streamer_output_text == hf_output_text, (
        f"Outputs do not match! "
        f"Streamer output: {streamer_output_text}, HF output: {hf_output_text}")


def test_correctness_torchax_ray_distributed_executor(
    sampling_config: SamplingParams,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Compare the outputs of a local-path model loaded via runai_model_streamer
    and the default loader, both using TP=2. The outputs should be the same.
    """
    prompt = "def fibonacci("

    streamer_llm = LLM(model=MODEL_NAME,
                       load_format="runai_streamer",
                       tensor_parallel_size=2,
                       max_model_len=128,
                       max_num_seqs=16,
                       max_num_batched_tokens=256)
    streamer_outputs = streamer_llm.generate([prompt], sampling_config)
    streamer_output_text = streamer_outputs[0].outputs[0].text
    del streamer_llm
    time.sleep(10)  # Wait for TPUs to be released

    # Test with Hugging Face model
    hf_llm = LLM(model=MODEL_NAME,
                 tensor_parallel_size=2,
                 max_model_len=128,
                 max_num_seqs=16,
                 max_num_batched_tokens=256)
    hf_outputs = hf_llm.generate([prompt], sampling_config)
    hf_output_text = hf_outputs[0].outputs[0].text
    del hf_llm
    time.sleep(10)  # Wait for TPUs to be released

    assert streamer_output_text == hf_output_text, (
        f"Outputs do not match! "
        f"Streamer output: {streamer_output_text}, HF output: {hf_output_text}")
