# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.online_rl_server_related


_MODULE_PATH = (Path(__file__).resolve().parents[1] / "e2e" /
                "test_online_rl_production_harness.py")
_SPEC = importlib.util.spec_from_file_location(
    "test_online_rl_production_harness_module",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _Response:

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _MODULE.requests.exceptions.HTTPError(response=self)

    def json(self) -> dict[str, Any]:
        return self._payload


def test_call_llm_continues_across_intermediate_length_chunks(
    monkeypatch,
) -> None:
    prompts: list[str] = []

    def fake_post(_url: str, *, json: dict[str, Any], timeout: float) -> _Response:
        del timeout
        prompts.append(str(json["prompt"]))
        if len(prompts) == 1:
            return _Response(
                {
                    "choices": [{
                        "text": "seedabc",
                        "finish_reason": "length",
                        "logprobs": {
                            "tokens": ["a", "b", "c"],
                            "text_offset": [4, 5, 6],
                            "token_logprobs": [-0.1, -0.2, -0.3],
                        },
                    }],
                    "usage": {
                        "completion_tokens": 3
                    },
                })
        return _Response(
            {
                "choices": [{
                    "text": "seedabcdef",
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": ["d", "e", "f"],
                        "text_offset": [7, 8, 9],
                        "token_logprobs": [-0.1, -0.2, -0.3],
                    },
                }],
                "usage": {
                    "completion_tokens": 3
                },
            })

    monkeypatch.setattr(_MODULE.requests, "post", fake_post)

    text, finish_reason, generated_tokens = _MODULE.call_llm(
        "seed",
        max_new_tokens=6,
        block_length=3,
        max_retries=0,
        retry_delay=0.0,
        request_timeout=1.0,
        max_model_len=128,
        count_prompt_tokens=lambda _text: 1,
    )

    assert prompts == ["seed", "seedabc"]
    assert text == "abcdef"
    assert finish_reason == _MODULE.FinishReason.STOP
    assert generated_tokens == 6


def test_call_llm_caps_chunk_size_to_remaining_context(monkeypatch) -> None:
    seen_max_tokens: list[int] = []

    def fake_post(_url: str, *, json: dict[str, Any], timeout: float) -> _Response:
        del timeout
        seen_max_tokens.append(int(json["max_tokens"]))
        prompt = str(json["prompt"])
        return _Response(
            {
                "choices": [{
                    "text": prompt + "xy",
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": ["x", "y"],
                        "text_offset": [len(prompt), len(prompt) + 1],
                        "token_logprobs": [-0.1, -0.2],
                    },
                }],
                "usage": {
                    "completion_tokens": 2
                },
            })

    monkeypatch.setattr(_MODULE.requests, "post", fake_post)

    text, finish_reason, generated_tokens = _MODULE.call_llm(
        "prompt",
        max_new_tokens=100,
        max_retries=0,
        retry_delay=0.0,
        request_timeout=1.0,
        max_model_len=32,
        count_prompt_tokens=lambda _text: 30,
    )

    assert seen_max_tokens == [2]
    assert text == "xy"
    assert finish_reason == _MODULE.FinishReason.STOP
    assert generated_tokens == 2


def test_call_llm_returns_context_limit_when_prompt_fills_window(
    monkeypatch,
) -> None:
    call_count = 0

    def fake_post(_url: str, *, json: dict[str, Any], timeout: float) -> _Response:
        nonlocal call_count
        del timeout
        call_count += 1
        prompt = str(json["prompt"])
        return _Response(
            {
                "choices": [{
                    "text": prompt + "abc",
                    "finish_reason": "length",
                    "logprobs": {
                        "tokens": ["a", "b", "c"],
                        "text_offset": [
                            len(prompt),
                            len(prompt) + 1,
                            len(prompt) + 2,
                        ],
                        "token_logprobs": [-0.1, -0.2, -0.3],
                    },
                }],
                "usage": {
                    "completion_tokens": 3
                },
            })

    monkeypatch.setattr(_MODULE.requests, "post", fake_post)

    token_counts = iter([28, 31, 32])
    text, finish_reason, generated_tokens = _MODULE.call_llm(
        "seed",
        max_new_tokens=100,
        block_length=3,
        max_retries=0,
        retry_delay=0.0,
        request_timeout=1.0,
        max_model_len=32,
        count_prompt_tokens=lambda _text: next(token_counts),
    )

    assert call_count == 1
    assert text == "abc"
    assert finish_reason == _MODULE.FinishReason.CONTEXT_LIMIT
    assert generated_tokens == 3
