# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from dataclasses import asdict
from unittest.mock import patch

import pytest
import vllm.envs as vllm_envs
from vllm import LLM, EngineArgs, SamplingParams

from tpu_inference.core.core_tpu import DisaggEngineCore, DisaggEngineCoreProc

MODEL_NAME = (
    "/home/jan/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/"
    "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
DISAGG_SLICE_TP_SIZE = 2


@pytest.fixture
def test_prompts():
    """Simple test prompts for disaggregated serving testing."""
    return [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
        "The president of the United States is",
        "How many players are on a standard soccer team on the field at one time?",
        "In Greek mythology, who is the god of the sea?",
        "In what year did the Titanic sink?",
        "In which museum is the Mona Lisa displayed?",
        "Mount Everest is located in which mountain range?",
        "What ancient empire was ruled by Julius Caesar?",
        "What are the four fundamental forces of nature?",
        'What does "CPU" stand for?',
        'What does "HTML" stand for?',
        "What is the capital of Australia?",
        "What is the chemical symbol for gold?",
        "What is the currency of Switzerland?",
        "What is the distance from the Earth to the Sun called?",
        "What is the freezing point of water in Celsius?",
        "What is the hardest known natural substance on Earth?",
        "What is the largest planet in our solar system?",
        "What is the longest river in the world?",
        "What is the main function of the kidneys in the human body?",
        "What is the main ingredient in guacamole?",
        "What is the most spoken language in the world by number of native speakers?",
        "What is the process by which plants use sunlight to create food?",
        "Which country is known as the Land of the Rising Sun?",
        "Who developed the theory of general relativity?",
        'Who directed the original "Star Wars" trilogy?',
        "Who is credited with inventing the telephone?",
        "Who painted the ceiling of the Sistine Chapel?",
        "Who was the first female Prime Minister of the United Kingdom?",
        "Who was the first person to walk on the moon?",
        "Who wrote the American Declaration of Independence?",
        'Who wrote the novel "Pride and Prejudice"?',
    ]


@pytest.fixture
def sampling_params():
    """Standard sampling parameters for testing."""
    return SamplingParams(
        temperature=0.0,
        max_tokens=32,
        ignore_eos=True,
        logprobs=1,
    )


def test_disaggregated_serving(test_prompts, sampling_params):
    """
    Test disaggregated serving end-to-end.

    Equivalent to:
    PREFILL_SLICES=2 DECODE_SLICES=2 python examples/offline_inference.py \
        --model=<cached Qwen2.5-1.5B-Instruct> --task=generate \
        --max_model_len=2048 --tensor_parallel_size 2
    """
    # Set environment variables for disaggregated serving
    # On a 4-chip machine the split must satisfy prefill + decode <= 4.
    # Use an even 2+2 partition so the test exercises disagg without asking for
    # more chips than the host actually has.

    # We need to mock the environment variables for this test
    with patch.dict(
            os.environ, {
                "PREFILL_SLICES": "2",
                "DECODE_SLICES": "2",
                "SKIP_JAX_PRECOMPILE": "1",
                "VLLM_XLA_CHECK_RECOMPILATION": "0"
            }):
        # Patch the EngineCore classes to use Disagg versions
        with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), \
             patch("vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):

            model_name = MODEL_NAME
            os.system(f"rm -rf {vllm_envs.VLLM_XLA_CACHE_PATH}/*")
            engine_args = EngineArgs(
                model=model_name,
                max_model_len=2048,
                tensor_parallel_size=DISAGG_SLICE_TP_SIZE,
                gpu_memory_utilization=0.90,
                enforce_eager=False,
            )

            llm = LLM(**asdict(engine_args))

            try:
                outputs = llm.generate(test_prompts, sampling_params)

                # Verify outputs
                assert len(outputs) == len(test_prompts)
                for output in outputs:
                    assert len(output.outputs) > 0
                    assert len(output.outputs[0].text.strip()) > 0
                    print(f"Prompt: {output.prompt!r}")
                    print(f"Generated: {output.outputs[0].text!r}")

            finally:
                del llm
                time.sleep(10)
                pass


def _run_inference(model_name: str,
                   test_prompts: list,
                   sampling_params: SamplingParams,
                   tensor_parallel_size: int = DISAGG_SLICE_TP_SIZE,
                   is_disagg: bool = False,
                   prefill_slices: str = "2",
                   decode_slices: str = "2") -> list:
    """Helper function to run inference with specified configuration."""

    # Define the inner execution logic
    def run_inner():
        engine_args = EngineArgs(
            model=model_name,
            max_model_len=2048,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.90,
            enforce_eager=False,
        )

        llm = LLM(**asdict(engine_args))
        try:
            return llm.generate(test_prompts, sampling_params)
        finally:
            del llm
            time.sleep(10)
            pass

    if is_disagg:
        # Mock environment variables and patch classes for disagg
        with patch.dict(
                os.environ, {
                    "PREFILL_SLICES": prefill_slices,
                    "DECODE_SLICES": decode_slices,
                    "SKIP_JAX_PRECOMPILE": "1",
                    "VLLM_XLA_CHECK_RECOMPILATION": "0"
                }):
            with patch("vllm.v1.engine.core.EngineCore", DisaggEngineCore), \
                 patch("vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):
                return run_inner()
    else:
        # Run standard inference
        # We still set some env vars to ensure consistent behavior if needed
        # but for baseline we want it as standard as possible.
        # However, to match the disagg run's potential jax settings:
        with patch.dict(os.environ, {
                "SKIP_JAX_PRECOMPILE": "1",
                "VLLM_XLA_CHECK_RECOMPILATION": "0"
        }):
            return run_inner()


def test_disaggregated_serving_correctness(test_prompts, sampling_params):
    """
    Test that disaggregated serving produces consistent results compared to a baseline.
    """
    model_name = MODEL_NAME
    # Use a smaller subset of prompts for correctness testing
    small_prompts = test_prompts[:20]
    sampling_params.max_tokens = 16

    # Run baseline (standard execution) with the same per-engine TP size the
    # disaggregated prefill/decode slices will actually use on a 2+2 split.
    print("Running Baseline Inference...")
    baseline_outputs = _run_inference(model_name=model_name,
                                      test_prompts=small_prompts,
                                      sampling_params=sampling_params,
                                      tensor_parallel_size=DISAGG_SLICE_TP_SIZE,
                                      is_disagg=False)

    # Run disaggregated inference
    os.system(f"rm -rf {vllm_envs.VLLM_XLA_CACHE_PATH}/*")
    print("Running Disaggregated Inference...")

    disagg_outputs = _run_inference(model_name=model_name,
                                    test_prompts=small_prompts,
                                    sampling_params=sampling_params,
                                    tensor_parallel_size=DISAGG_SLICE_TP_SIZE,
                                    is_disagg=True,
                                    prefill_slices="2",
                                    decode_slices="2")

    assert len(baseline_outputs) == len(disagg_outputs)

    text_matches = 0
    text_mismatches = 0
    token_mismatches = 0

    for i, (baseline, disagg) in enumerate(zip(baseline_outputs,
                                               disagg_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        disagg_text = disagg.outputs[0].text.strip()

        if baseline_text == disagg_text:
            text_matches += 1
        else:
            text_mismatches += 1
            print(f"Text mismatch found in prompt {i}:")
            print(f"  Baseline: {baseline_text}")
            print(f"  Disagg: {disagg_text}")

        baseline_tokens = baseline.outputs[0].token_ids
        disagg_tokens = disagg.outputs[0].token_ids
        if baseline_tokens != disagg_tokens:
            token_mismatches += 1
            print(f"Token mismatch found in prompt {i}:")
            print(f"  Baseline: {baseline_tokens}")
            print(f"  Disagg: {disagg_tokens}")

    print("✓ Disaggregated correctness test results:")
    print(f"  Text: {text_matches} matches, {text_mismatches} mismatches")
    print(f"  Token mismatches: {token_mismatches}")

    text_match_rate = text_matches / len(baseline_outputs)
    assert text_match_rate >= 0.8, f"Text match rate {text_match_rate:.2%} is too low"
    assert token_mismatches <= 4, (
        f"Too many token-level mismatches: {token_mismatches}/{len(baseline_outputs)}")
