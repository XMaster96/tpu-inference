# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import difflib
import os
import time
from dataclasses import asdict

import pytest
from vllm import LLM, EngineArgs, SamplingParams

MODEL_NAME = (
    "/home/jan/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/"
    "snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
)


@pytest.fixture
def model_name():
    return MODEL_NAME


@pytest.fixture
def test_prompts():
    return [
        "Hello, my name is",
        "The capital of France is",
        "The colors of the rainbow are",
        "The future of AI is",
        "The president of the United States is",
        "How many players are on a standard soccer team?",
        "In Greek mythology, who is the god of the sea?",
        "What is the capital of Australia?",
        "What is the largest planet in our solar system?",
        "Who developed the theory of general relativity?",
    ]


@pytest.fixture
def sampling_params():
    return SamplingParams(
        temperature=0.0,
        max_tokens=32,
        ignore_eos=True,
        logprobs=1,
    )


def _run_inference_with_config(model_name: str,
                               test_prompts: list,
                               sampling_params: SamplingParams,
                               tensor_parallel_size: int = 1,
                               use_ep_kernel_flag: bool = False) -> list:
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    os.environ['VLLM_XLA_CHECK_RECOMPILATION'] = '0'

    previous_model_impl = os.environ.get("MODEL_IMPL_TYPE")
    previous_use_moe_ep_kernel = os.environ.get("USE_MOE_EP_KERNEL")
    os.environ['MODEL_IMPL_TYPE'] = 'vllm'
    if use_ep_kernel_flag:
        os.environ['USE_MOE_EP_KERNEL'] = '1'
    else:
        os.environ.pop('USE_MOE_EP_KERNEL', None)

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=128,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        enable_prefix_caching=False,
        kv_cache_dtype="auto",
        enable_expert_parallel=False,
    )

    try:
        llm = LLM(**asdict(engine_args))
        return llm.generate(test_prompts, sampling_params)
    finally:
        if 'llm' in locals():
            del llm
        if previous_model_impl is None:
            os.environ.pop('MODEL_IMPL_TYPE', None)
        else:
            os.environ['MODEL_IMPL_TYPE'] = previous_model_impl
        if previous_use_moe_ep_kernel is None:
            os.environ.pop('USE_MOE_EP_KERNEL', None)
        else:
            os.environ['USE_MOE_EP_KERNEL'] = previous_use_moe_ep_kernel
        time.sleep(5)


def _verify_correctness(baseline_outputs, experiment_outputs, label: str):
    assert len(baseline_outputs) == len(experiment_outputs)

    text_matches = 0
    text_mismatches = 0
    max_logprob_diff = 0.0

    for i, (baseline, experiment) in enumerate(
            zip(baseline_outputs, experiment_outputs)):
        baseline_text = baseline.outputs[0].text.strip()
        experiment_text = experiment.outputs[0].text.strip()

        if baseline_text == experiment_text:
            text_matches += 1
        else:
            similarity = difflib.SequenceMatcher(None, baseline_text,
                                                 experiment_text).ratio()
            if similarity >= 0.95:
                text_matches += 1
            else:
                text_mismatches += 1
                print(f"Text mismatch for prompt {i}:")
                print(f"  Baseline: {baseline_text}")
                print(f"  {label}: {experiment_text}")

        baseline_logprobs = baseline.outputs[0].logprobs
        experiment_logprobs = experiment.outputs[0].logprobs
        if baseline_logprobs is None or experiment_logprobs is None:
            continue

        assert len(baseline_logprobs) == len(experiment_logprobs)
        for base_lp_dict, exp_lp_dict in zip(baseline_logprobs,
                                             experiment_logprobs):
            if not base_lp_dict or not exp_lp_dict:
                continue
            base_token, base_lp = next(iter(base_lp_dict.items()))
            exp_token, exp_lp = next(iter(exp_lp_dict.items()))
            if base_token == exp_token:
                    diff = abs(base_lp.logprob - exp_lp.logprob)
                    max_logprob_diff = max(max_logprob_diff, diff)

    text_match_rate = text_matches / len(baseline_outputs)
    print("✓ Correctness test results:")
    print(f"  Text: {text_matches} matches, {text_mismatches} mismatches")
    print(f"  Max logprob difference: {max_logprob_diff:.6e}")

    # MoE TP can diverge slightly from single-chip greedy decoding due to
    # numerical differences in routing/logit accumulation. Match the tolerance
    # already used by the general TP E2E.
    assert text_match_rate >= 0.8
    assert max_logprob_diff < 1.5


def test_moe_tensor_parallelism_correctness(model_name: str,
                                            test_prompts: list,
                                            sampling_params: SamplingParams):
    baseline_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
    )
    tp_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=4,
    )
    _verify_correctness(baseline_outputs, tp_outputs, "MoE Tensor Parallel")


def test_moe_tensor_parallelism_with_ep_kernel_flag_falls_back(
        model_name: str, test_prompts: list,
        sampling_params: SamplingParams):
    baseline_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
    )
    tp_outputs = _run_inference_with_config(
        model_name=model_name,
        test_prompts=test_prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=4,
        use_ep_kernel_flag=True,
    )
    _verify_correctness(
        baseline_outputs,
        tp_outputs,
        "MoE Tensor Parallel with USE_MOE_EP_KERNEL=1 fallback",
    )
