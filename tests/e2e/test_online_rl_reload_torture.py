# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests


DEFAULT_ORBAX_CHECKPOINT = (
    "gs://ml-flops-checkpoints-us-central2/"
    "9k_books--batch_size=64-num_epochs=1-lr_milestones="
    "LR({in_stp=25;lr=1e-06;linear}->{lr=0;cosine})-"
    "mistral_24b-seq_lens=16384-use_stochastic_rounding_smooth-"
    "orpo_beta=1---run_1"
)
DEFAULT_HF_MODEL_DIR = (
    "/dev/shm/ml-flops-checkpoints-us-central2/"
    "9k_books--batch_size=64-num_epochs=1-lr_milestones="
    "LR({in_stp=25;lr=1e-06;linear}->{lr=0;cosine})-"
    "mistral_24b-seq_lens=16384-use_stochastic_rounding_smooth-"
    "orpo_beta=1---run_1"
)


def _e2e_enabled() -> bool:
    return os.environ.get("TPU_ONLINE_RL_E2E", "0") == "1"


pytestmark = pytest.mark.skipif(
    not _e2e_enabled(),
    reason=(
        "Set TPU_ONLINE_RL_E2E=1 to run live server torture tests. "
        "These tests are intentionally expensive."
    ),
)


@dataclass
class _E2EConfig:
    model_dir: str
    tokenizer_dir: str
    reload_checkpoints: list[str]
    startup_dummy_weights_path: str
    host: str
    port: int
    tensor_parallel_size: int
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    reload_timeout_seconds: int
    completion_timeout_seconds: int
    server_start_timeout_seconds: int
    inflight_grace_seconds: float
    max_tokens: int
    prompt_repeat_count: int
    reloads_per_generation: int
    reload_storm_rounds: int
    parallel_generations: int
    continuous_reload_interval_seconds: float
    continuous_reload_min_cycles: int
    continuous_reload_max_cycles: int


@dataclass
class _ServerContext:
    config: _E2EConfig
    process: subprocess.Popen
    base_url: str
    model_name: str
    stdout_path: str
    stderr_path: str


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_config() -> _E2EConfig:
    model_dir = os.environ.get("TPU_ONLINE_RL_E2E_MODEL_DIR",
                               DEFAULT_HF_MODEL_DIR)
    tokenizer_dir = os.environ.get("TPU_ONLINE_RL_E2E_TOKENIZER_DIR",
                                   model_dir)

    raw_checkpoints = os.environ.get("TPU_ONLINE_RL_E2E_RELOAD_CHECKPOINTS",
                                     DEFAULT_ORBAX_CHECKPOINT)
    reload_checkpoints = [p.strip() for p in raw_checkpoints.split(",") if p.strip()]
    if not reload_checkpoints:
        pytest.skip(
            "TPU_ONLINE_RL_E2E_RELOAD_CHECKPOINTS resolved to an empty list.")

    startup_dummy_weights_path = os.environ.get(
        "TPU_ONLINE_RL_E2E_STARTUP_DUMMY_WEIGHTS",
        "/tmp/tpu-online-rl-e2e-force-dummy-startup",
    )

    return _E2EConfig(
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        reload_checkpoints=reload_checkpoints,
        startup_dummy_weights_path=startup_dummy_weights_path,
        host="127.0.0.1",
        port=int(os.environ.get("TPU_ONLINE_RL_E2E_PORT", _pick_free_port())),
        tensor_parallel_size=int(os.environ.get("TPU_ONLINE_RL_E2E_TP", "4")),
        max_model_len=int(os.environ.get("TPU_ONLINE_RL_E2E_MAX_MODEL_LEN",
                                         "16384")),
        max_num_batched_tokens=int(
            os.environ.get("TPU_ONLINE_RL_E2E_MAX_BATCHED_TOKENS", "32768")),
        max_num_seqs=int(os.environ.get("TPU_ONLINE_RL_E2E_MAX_NUM_SEQS", "64")),
        reload_timeout_seconds=int(
            os.environ.get("TPU_ONLINE_RL_E2E_RELOAD_TIMEOUT_SECONDS", "7200")),
        completion_timeout_seconds=int(
            os.environ.get("TPU_ONLINE_RL_E2E_COMPLETION_TIMEOUT_SECONDS",
                           "7200")),
        server_start_timeout_seconds=int(
            os.environ.get("TPU_ONLINE_RL_E2E_SERVER_START_TIMEOUT_SECONDS",
                           "1800")),
        inflight_grace_seconds=float(
            os.environ.get("TPU_ONLINE_RL_E2E_INFLIGHT_GRACE_SECONDS", "15")),
        max_tokens=int(os.environ.get("TPU_ONLINE_RL_E2E_MAX_TOKENS", "2048")),
        prompt_repeat_count=int(
            os.environ.get("TPU_ONLINE_RL_E2E_PROMPT_REPEAT_COUNT", "10000")),
        reloads_per_generation=int(
            os.environ.get("TPU_ONLINE_RL_E2E_RELOADS_PER_GENERATION", "10")),
        reload_storm_rounds=int(
            os.environ.get("TPU_ONLINE_RL_E2E_RELOAD_STORM_ROUNDS", "20")),
        parallel_generations=int(
            os.environ.get("TPU_ONLINE_RL_E2E_PARALLEL_GENERATIONS", "4")),
        continuous_reload_interval_seconds=float(
            os.environ.get("TPU_ONLINE_RL_E2E_CONTINUOUS_RELOAD_INTERVAL_SECONDS",
                           "0.5")),
        continuous_reload_min_cycles=int(
            os.environ.get("TPU_ONLINE_RL_E2E_CONTINUOUS_RELOAD_MIN_CYCLES",
                           "30")),
        continuous_reload_max_cycles=int(
            os.environ.get("TPU_ONLINE_RL_E2E_CONTINUOUS_RELOAD_MAX_CYCLES",
                           "200")),
    )


def _read_tail(path: str, limit_bytes: int = 32_000) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(size - limit_bytes, 0), os.SEEK_SET)
        payload = f.read()
    return payload.decode("utf-8", errors="replace")


def _wait_for_http_or_crash(ctx: _ServerContext) -> None:
    deadline = time.time() + ctx.config.server_start_timeout_seconds
    status_url = f"{ctx.base_url}/status"
    while time.time() < deadline:
        if ctx.process.poll() is not None:
            stdout_tail = _read_tail(ctx.stdout_path)
            stderr_tail = _read_tail(ctx.stderr_path)
            raise AssertionError(
                "Online RL server exited during startup.\n"
                f"stdout tail:\n{stdout_tail}\n\nstderr tail:\n{stderr_tail}")
        try:
            resp = requests.get(status_url, timeout=2)
            if resp.status_code in (200, 503):
                return
        except requests.RequestException:
            pass
        time.sleep(1)

    raise AssertionError(
        f"Online RL server did not become reachable within "
        f"{ctx.config.server_start_timeout_seconds}s")


def _reload_weights(
    ctx: _ServerContext,
    checkpoint_path: str,
    *,
    wait_for_inflight_requests: bool,
    clear_cache: bool,
) -> dict[str, Any]:
    payload = {
        "checkpoint_path": checkpoint_path,
        "wait_for_inflight_requests": wait_for_inflight_requests,
        "clear_cache": clear_cache,
        "release_kv_cache": True,
        "timeout_seconds": float(ctx.config.reload_timeout_seconds),
    }
    response = requests.post(
        f"{ctx.base_url}/v1/reload_weights",
        json=payload,
        timeout=ctx.config.reload_timeout_seconds + 120,
    )
    assert response.status_code == 200, (
        f"/v1/reload_weights failed with {response.status_code}: "
        f"{response.text[:4000]}")
    return response.json()


def _resolve_model_name(ctx: _ServerContext) -> str:
    response = requests.get(f"{ctx.base_url}/v1/models", timeout=60)
    if response.status_code != 200:
        return ctx.config.model_dir
    payload = response.json()
    model_data = payload.get("data")
    if not isinstance(model_data, list) or not model_data:
        return ctx.config.model_dir
    model_id = model_data[0].get("id")
    return str(model_id) if model_id else ctx.config.model_dir


def _wait_until_ready(ctx: _ServerContext) -> None:
    deadline = time.time() + ctx.config.reload_timeout_seconds + 120
    while time.time() < deadline:
        response = requests.get(f"{ctx.base_url}/status", timeout=10)
        if response.status_code == 200:
            payload = response.json()
            if payload.get("ready") is True:
                return
        time.sleep(2)
    raise AssertionError("Server did not become ready after reload.")


def _make_long_prompt(repeat_count: int, *, seed: int) -> str:
    chunks = []
    for i in range(repeat_count):
        chunks.append(
            f"[{seed:03d}:{i:05d}] Keep structure stable across reloads. "
            f"Emit coherent technical prose with punctuation and line breaks."
        )
    return "\n".join(chunks)


def _make_complex_regex() -> str:
    # Keep the regex broad enough to exercise constrained decoding, but avoid
    # nested repetition that can trigger backend-specific lexer blowups.
    return r"^(?:[A-Za-z0-9 _.,;:!?()\[\]{}'\"/\-\n\t]{32,4096})$"


def _completion_request(
    ctx: _ServerContext,
    *,
    prompt: str,
    regex: str,
) -> dict[str, Any]:
    payload = {
        "model": ctx.model_name,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": ctx.config.max_tokens,
        "skip_special_tokens": False,
        "logprobs": 1,
        "structured_outputs": {
            "regex": regex
        },
    }
    response = requests.post(
        f"{ctx.base_url}/v1/completions",
        json=payload,
        timeout=ctx.config.completion_timeout_seconds,
    )
    assert response.status_code == 200, (
        f"/v1/completions failed with {response.status_code}: "
        f"{response.text[:4000]}")
    payload = response.json()
    choices = payload.get("choices")
    assert isinstance(choices, list) and choices, "Completion returned no choices."
    text = choices[0].get("text")
    assert isinstance(text, str) and text, "Completion returned empty text."
    return payload


@pytest.fixture(scope="module")
def online_rl_server_context() -> _ServerContext:
    cfg = _build_config()

    if cfg.parallel_generations < 1:
        pytest.skip("TPU_ONLINE_RL_E2E_PARALLEL_GENERATIONS must be >= 1")
    if cfg.reloads_per_generation < 1:
        pytest.skip("TPU_ONLINE_RL_E2E_RELOADS_PER_GENERATION must be >= 1")
    if cfg.continuous_reload_min_cycles < 1:
        pytest.skip("TPU_ONLINE_RL_E2E_CONTINUOUS_RELOAD_MIN_CYCLES must be >= 1")
    if cfg.continuous_reload_max_cycles < cfg.continuous_reload_min_cycles:
        pytest.skip(
            "TPU_ONLINE_RL_E2E_CONTINUOUS_RELOAD_MAX_CYCLES must be >= "
            "TPU_ONLINE_RL_E2E_CONTINUOUS_RELOAD_MIN_CYCLES")

    repo_root = Path(__file__).resolve().parents[2]
    stdout_file = tempfile.NamedTemporaryFile(prefix="online-rl-e2e-stdout-",
                                              suffix=".log",
                                              delete=False)
    stderr_file = tempfile.NamedTemporaryFile(prefix="online-rl-e2e-stderr-",
                                              suffix=".log",
                                              delete=False)
    stdout_path = stdout_file.name
    stderr_path = stderr_file.name
    stdout_file.close()
    stderr_file.close()

    command = [
        sys.executable,
        "-m",
        "tpu_inference.entrypoints.online_rl_server",
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--tensor-parallel-size",
        str(cfg.tensor_parallel_size),
        "--max-model-len",
        str(cfg.max_model_len),
        "--max-num-batched-tokens",
        str(cfg.max_num_batched_tokens),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--model",
        cfg.model_dir,
        "--tokenizer",
        cfg.tokenizer_dir,
        "--model-weights",
        cfg.startup_dummy_weights_path,
    ]

    env = os.environ.copy()
    env.setdefault("MODEL_IMPL_TYPE", "flax_nnx")
    env.setdefault("SKIP_JAX_PRECOMPILE", "1")
    env.setdefault("VLLM_XLA_CHECK_RECOMPILATION", "0")

    with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=stdout,
            stderr=stderr,
            preexec_fn=os.setsid,
        )

    ctx = _ServerContext(
        config=cfg,
        process=process,
        base_url=f"http://{cfg.host}:{cfg.port}",
        model_name=cfg.model_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    try:
        _wait_for_http_or_crash(ctx)
        _reload_weights(
            ctx,
            checkpoint_path=cfg.reload_checkpoints[0],
            wait_for_inflight_requests=True,
            clear_cache=True,
        )
        _wait_until_ready(ctx)
        ctx.model_name = _resolve_model_name(ctx)
        yield ctx
    finally:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_torture_single_generation_with_multiple_reloads(
    online_rl_server_context: _ServerContext,
) -> None:
    ctx = online_rl_server_context
    regex = _make_complex_regex()
    prompt = _make_long_prompt(ctx.config.prompt_repeat_count, seed=1)

    completion_payload: dict[str, Any] = {}
    completion_error: list[BaseException] = []
    request_started = threading.Event()

    def completion_worker() -> None:
        request_started.set()
        try:
            completion_payload.update(
                _completion_request(ctx, prompt=prompt, regex=regex))
        except BaseException as exc:  # noqa: BLE001
            completion_error.append(exc)

    completion_thread = threading.Thread(target=completion_worker, daemon=True)
    completion_thread.start()

    assert request_started.wait(timeout=10), "Completion worker did not start."
    time.sleep(ctx.config.inflight_grace_seconds)

    reloaded_while_inflight = False
    for i in range(ctx.config.reloads_per_generation):
        if completion_thread.is_alive():
            reloaded_while_inflight = True
        checkpoint_path = ctx.config.reload_checkpoints[i % len(
            ctx.config.reload_checkpoints)]
        _reload_weights(
            ctx,
            checkpoint_path=checkpoint_path,
            wait_for_inflight_requests=False,
            clear_cache=(i % 2 == 0),
        )

    completion_thread.join(timeout=ctx.config.completion_timeout_seconds)
    assert not completion_thread.is_alive(), (
        "Completion request did not finish within timeout.")
    assert not completion_error, f"Completion worker failed: {completion_error[0]}"
    assert reloaded_while_inflight, (
        "No reload happened while completion was still active; increase "
        "prompt/max_tokens for a stronger torture run.")

    completion_text = completion_payload["choices"][0]["text"]
    assert re.fullmatch(regex, completion_text) is not None, (
        "Completion did not match regex after multi-reload torture.")


def test_torture_parallel_generations_with_reload_storm(
    online_rl_server_context: _ServerContext,
) -> None:
    ctx = online_rl_server_context
    regex = _make_complex_regex()

    prompts = [
        _make_long_prompt(max(512, ctx.config.prompt_repeat_count // 2), seed=100 + i)
        for i in range(ctx.config.parallel_generations)
    ]

    with ThreadPoolExecutor(max_workers=ctx.config.parallel_generations) as pool:
        futures = [
            pool.submit(_completion_request, ctx, prompt=prompt, regex=regex)
            for prompt in prompts
        ]

        time.sleep(ctx.config.inflight_grace_seconds)
        saw_inflight = False
        for i in range(ctx.config.reload_storm_rounds):
            if any(not future.done() for future in futures):
                saw_inflight = True
            checkpoint_path = ctx.config.reload_checkpoints[i % len(
                ctx.config.reload_checkpoints)]
            _reload_weights(
                ctx,
                checkpoint_path=checkpoint_path,
                wait_for_inflight_requests=False,
                clear_cache=(i % 2 == 0),
            )

        results = [
            future.result(timeout=ctx.config.completion_timeout_seconds)
            for future in futures
        ]

    assert saw_inflight, (
        "Reload storm did not overlap with active completions; increase "
        "prompt/max_tokens to keep requests active longer.")

    for payload in results:
        text = payload["choices"][0]["text"]
        assert re.fullmatch(regex, text) is not None, (
            "At least one completion violated regex after reload storm.")


def test_torture_continuous_reload_soak(
    online_rl_server_context: _ServerContext,
) -> None:
    ctx = online_rl_server_context
    regex = _make_complex_regex()

    prompts = [
        _make_long_prompt(ctx.config.prompt_repeat_count, seed=200 + i)
        for i in range(ctx.config.parallel_generations)
    ]

    with ThreadPoolExecutor(max_workers=ctx.config.parallel_generations) as pool:
        futures = [
            pool.submit(_completion_request, ctx, prompt=prompt, regex=regex)
            for prompt in prompts
        ]

        time.sleep(ctx.config.inflight_grace_seconds)

        reload_cycles = 0
        overlap_cycles = 0
        while reload_cycles < ctx.config.continuous_reload_max_cycles:
            active_count = sum(1 for future in futures if not future.done())
            if active_count > 0:
                overlap_cycles += 1

            checkpoint_path = ctx.config.reload_checkpoints[
                reload_cycles % len(ctx.config.reload_checkpoints)]
            _reload_weights(
                ctx,
                checkpoint_path=checkpoint_path,
                wait_for_inflight_requests=False,
                clear_cache=(reload_cycles % 2 == 0),
            )
            reload_cycles += 1

            all_done = all(future.done() for future in futures)
            if all_done and reload_cycles >= ctx.config.continuous_reload_min_cycles:
                break
            time.sleep(ctx.config.continuous_reload_interval_seconds)

        assert reload_cycles >= ctx.config.continuous_reload_min_cycles, (
            "Continuous reload soak ended before minimum cycle target.")
        assert overlap_cycles > 0, (
            "Continuous reload soak never overlapped active generations.")

        results = [
            future.result(timeout=ctx.config.completion_timeout_seconds)
            for future in futures
        ]

    for payload in results:
        text = payload["choices"][0]["text"]
        assert re.fullmatch(regex, text) is not None, (
            "At least one completion violated regex after continuous reload soak.")
