# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import random
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, cast

import pytest
import requests

pytestmark = pytest.mark.online_rl_server_related


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
LOCAL_MISTRAL3_14B_MODEL_DIR = (
    "/dev/shm/9k_books_new_arc_summary-stage_2-self_attn_plus_mlp_gate_only---"
    "batch_size=24-num_epochs=6-lr_milestones=LR({in_stp=25;lr=2e-05;linear}->"
    "{lr=0;cosine})-mistral3_14b_base-seq_lens=262144-"
    "qk_clip_max_attention_logit_score=85---run_1"
)
LOCAL_MISTRAL3_14B_ORBAX_CHECKPOINT = "/dev/shm/buffer--orbax"

def _debug_enabled() -> bool:
    return os.environ.get("TPU_ONLINE_RL_PROD_DEBUG", "0") == "1"


def _debug_stage(stage: str, *, finish_reason: FinishReason, generated: int,
                 cumulative_tokens: int) -> None:
    if _debug_enabled():
        print(
            "PROD_DEBUG",
            f"stage={stage}",
            f"finish_reason={finish_reason.value}",
            f"generated={generated}",
            f"cumulative_tokens={cumulative_tokens}",
            flush=True,
        )

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _default_model_dir() -> str:
    if os.path.exists(LOCAL_MISTRAL3_14B_MODEL_DIR):
        return LOCAL_MISTRAL3_14B_MODEL_DIR
    return DEFAULT_HF_MODEL_DIR


def _default_orbax_checkpoint() -> str:
    if os.path.exists(LOCAL_MISTRAL3_14B_ORBAX_CHECKPOINT):
        return LOCAL_MISTRAL3_14B_ORBAX_CHECKPOINT
    return DEFAULT_ORBAX_CHECKPOINT


def _tokenize_for_ngram(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _ngram_multiset_f1(reference: str, candidate: str, *, n: int) -> float:
    assert n > 0, "n must be > 0"
    ref_tokens = _tokenize_for_ngram(reference)
    cand_tokens = _tokenize_for_ngram(candidate)

    if len(ref_tokens) < n and len(cand_tokens) < n:
        return 1.0
    if len(ref_tokens) < n or len(cand_tokens) < n:
        return 0.0

    ref_ngrams = Counter(tuple(ref_tokens[i:i + n])
                         for i in range(len(ref_tokens) - n + 1))
    cand_ngrams = Counter(tuple(cand_tokens[i:i + n])
                          for i in range(len(cand_tokens) - n + 1))
    overlap = sum((ref_ngrams & cand_ngrams).values())
    ref_total = sum(ref_ngrams.values())
    cand_total = sum(cand_ngrams.values())
    if ref_total == 0 and cand_total == 0:
        return 1.0
    if ref_total == 0 or cand_total == 0:
        return 0.0
    precision = overlap / cand_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


BOOK_PREVIEW_MAX_NEW_TOKENS = _env_int(
    "TPU_ONLINE_RL_PROD_BOOK_PREVIEW_MAX_NEW_TOKENS", 256)
BOOK_PLAN_MAX_NEW_TOKENS = _env_int(
    "TPU_ONLINE_RL_PROD_BOOK_PLAN_MAX_NEW_TOKENS", 512)
FIRST_CHAPTER_PLAN_MAX_NEW_TOKENS = _env_int(
    "TPU_ONLINE_RL_PROD_FIRST_CHAPTER_PLAN_MAX_NEW_TOKENS", 256)
CHARACTERS_LIST_MAX_NEW_TOKENS = _env_int(
    "TPU_ONLINE_RL_PROD_CHARACTERS_LIST_MAX_NEW_TOKENS", 256)
SCENE_BREAKDOWN_MAX_NEW_TOKENS = _env_int(
    "TPU_ONLINE_RL_PROD_SCENE_BREAKDOWN_MAX_NEW_TOKENS", 192)
MAX_TOKENS = _env_int("TPU_ONLINE_RL_PROD_MAX_TOKENS", 2048)

BULLET_RE = r"- \S+(?:[^\S\n]+\S+){4,44}"
EMBEDDING_SPACE_RE = (
    r"action: [0-9]{1,2}, dialog: [0-9]{1,2}, world_building: [0-9]{1,2}, "
    r"exposition: [0-9]{1,2}, romantic: [0-9]{1,2}, erotic: [0-9]{1,2}, "
    r"pacing: [0-9]{1,2}\n"
)
NARRATIVE_FOCUS_NAME_RE = r"\S(?:[^\n;]{0,63}\S)?"
NARRATIVE_FOCUSES_RE = (
    NARRATIVE_FOCUS_NAME_RE +
    r"(?:; " + NARRATIVE_FOCUS_NAME_RE + r"){0,6}"
)
FIRST_CHAPTER_PLAN_SCENE_RE = (
    r"#### Scene (?:[1-9]|1[0-2]): (?:\S+(?:[^\S\n]+\S+){0,11})\n"
    r"\*\*Word Count:\*\* [0-9]{2,5}\n"
    r"\*\*Embedding Space:\*\* " + EMBEDDING_SPACE_RE +
    r"\*\*Narrative Focus:\*\*[^\S\n]+" + NARRATIVE_FOCUSES_RE + r"\n"
    r"\*\*Narrative Perspective:\*\* [^\n]{1,120}\n"
    r"\*\*Scene Summary:\*\*\n"
    r"(?:" + BULLET_RE + r"\n){0,16}" + BULLET_RE + r"\n?"
)
FIRST_CHAPTER_PLAN_RE = (
    r"\A" + NARRATIVE_FOCUSES_RE + r"\n"
    r"\*\*Chapter Summary:\*\*\n"
    r"(?:" + BULLET_RE + r"\n){0,99}" + BULLET_RE + r"\n\n"
    r"### Scene Breakdown\n"
    r"(?:" + FIRST_CHAPTER_PLAN_SCENE_RE + r"\n){0,11}"
    + FIRST_CHAPTER_PLAN_SCENE_RE +
    r"\z"
)
BOOK_PREVIEW_RE = (
    r"\A## Book Highlight\n((?:\S+\s+){0,149}\S+)\n\n"
    r"## Book Title\n((?:[^\s\n]+\s+){0,7}[^\s\n]+)\n\n"
    r"## Book Tags\n((?:- [^\r\n]{1,75}\n){7})\n"
    r"## Book Archetype\n((?:\S+\s+){0,199}\S+)\z"
)


class FinishReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    STOP_SEQUENCE = "stop_sequence"
    EMPTY_RESPONSE = "empty_response"
    CONTEXT_LIMIT = "context_limit"


def _normalize_stop_list(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        stop_list = [stop]
    elif isinstance(stop, list):
        stop_list = stop
    else:
        raise AssertionError("stop must be a string, list of strings, or None")

    normalized: list[str] = []
    for entry in stop_list:
        assert isinstance(entry, str), "each stop entry must be a string"
        assert entry, "stop entries must be non-empty strings"
        normalized.append(entry)
    return normalized


def _derive_finish_reason(
    *,
    hit_stop: bool,
    stop_reason: str | None,
    finish_tag: str | None,
    stop_list: list[str],
) -> FinishReason | None:
    if hit_stop:
        return FinishReason.STOP_SEQUENCE

    if stop_reason is not None:
        if stop_reason == "stop_sequence":
            return FinishReason.STOP_SEQUENCE
        if stop_reason == "length":
            return FinishReason.LENGTH
        if stop_reason == "stop":
            return FinishReason.STOP
        if stop_reason in stop_list:
            return FinishReason.STOP_SEQUENCE
        raise AssertionError(f"unexpected stop_reason value: {stop_reason!r}")

    if finish_tag is not None:
        if finish_tag == "stop":
            return FinishReason.STOP
        if finish_tag == "length":
            return FinishReason.LENGTH
        raise AssertionError(f"unexpected finish_reason value: {finish_tag!r}")
    return None


def _normalize_regex(regex: str | list[str] | None) -> str | None:
    if regex is None:
        return None
    if isinstance(regex, str):
        return regex
    assert isinstance(regex, list), "regex must be a string, list of strings, or None"
    assert regex, "regex list must be non-empty"
    assert all(isinstance(entry, str) and entry for entry in regex), (
        "regex list must contain only non-empty strings")
    return r"\A" + "".join(regex) + r"\z"


@lru_cache(maxsize=None)
def _get_prompt_token_counter(tokenizer_dir: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer_path = Path(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=tokenizer_path.exists(),
        trust_remote_code=False,
    )

    def count_prompt_tokens(text: str) -> int:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded["input_ids"]
        assert isinstance(input_ids, list), "tokenizer returned non-list input_ids"
        return len(input_ids)

    return count_prompt_tokens


@lru_cache(maxsize=None)
def _get_eos_token_info(tokenizer_dir: str) -> tuple[int, str]:
    from transformers import AutoTokenizer

    tokenizer_path = Path(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=tokenizer_path.exists(),
        trust_remote_code=False,
    )
    eos_token_id = tokenizer.eos_token_id
    assert isinstance(eos_token_id, int), "tokenizer eos_token_id must be an int"
    eos_text = tokenizer.decode([eos_token_id], skip_special_tokens=False)
    assert eos_text, "tokenizer EOS token must decode to non-empty text"
    return eos_token_id, eos_text


def call_llm(
    text: str,
    temperature: float = 0.0,
    max_new_tokens: int = 2000,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    best_of: int = 1,
    block_length: int | None | tuple[int, int] = None,
    stop: str | list[str] | None = None,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    regex: str | list[str] | None = None,
    server_ip: str = "localhost",
    server_port: int = 8100,
    model_name: str = "model",
    request_timeout: float = 10_000.0,
    max_model_len: int | None = None,
    count_prompt_tokens: Callable[[str], int] | None = None,
) -> tuple[str, FinishReason, int]:
    assert isinstance(text, str), "text must be a str"
    assert max_new_tokens > 0, "max_new_tokens must be > 0"
    assert max_retries >= 0, "max_retries must be >= 0"
    assert retry_delay >= 0, "retry_delay must be >= 0"
    assert best_of >= 1, "best_of must be >= 1"
    assert request_timeout > 0, "request_timeout must be > 0"
    if max_model_len is not None:
        assert max_model_len > 0, "max_model_len must be > 0 when provided"

    post_url = f"http://{server_ip}:{server_port}/v1/completions"
    stop_list = _normalize_stop_list(stop)
    normalized_regex = _normalize_regex(regex)

    def post_with_retry(payload: dict[str, Any]) -> dict[str, Any]:
        last_exception: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(post_url,
                                         json=payload,
                                         timeout=request_timeout)
                response.raise_for_status()
                return cast(dict[str, Any], response.json())
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as exc:
                last_exception = exc
            except requests.exceptions.HTTPError as exc:
                status_code = getattr(exc.response, "status_code", None)
                if status_code is not None and status_code >= 500:
                    last_exception = exc
                else:
                    raise
            if attempt < max_retries and retry_delay > 0:
                jitter = random.uniform(0.0, min(retry_delay, 0.25))
                time.sleep(retry_delay + jitter)
        if last_exception is None:
            raise RuntimeError("call_llm failed without raising an exception")
        raise last_exception

    prompt = text
    out_parts: list[str] = []
    remaining = max_new_tokens
    finish_reason = FinishReason.LENGTH
    generated_tokens = 0

    tuple_block: tuple[int, int] | None = None
    use_random_blocks = isinstance(block_length, tuple)
    if block_length is None:
        fixed_block = None
    elif isinstance(block_length, int):
        assert block_length > 0, "block_length must be > 0 when provided as int."
        fixed_block = block_length
    elif isinstance(block_length, tuple):
        assert len(block_length) == 2, "block_length tuple must be (min_len, max_len)."
        tuple_block = block_length
        lo, hi = tuple_block
        assert isinstance(lo, int) and isinstance(
            hi, int), "block_length tuple values must be ints."
        assert lo > 0 and hi > 0, "block_length tuple values must be > 0."
        assert lo <= hi, "block_length tuple must satisfy min_len <= max_len."
        fixed_block = None
    else:
        raise AssertionError("block_length must be None, int, or tuple[int, int].")

    while remaining > 0:
        if block_length is None:
            chunk_size = remaining
        elif use_random_blocks:
            assert tuple_block is not None
            lo, hi = tuple_block
            chunk_size = random.randint(lo, hi)
        else:
            assert fixed_block is not None
            chunk_size = fixed_block

        chunk_max = min(chunk_size, remaining)
        if max_model_len is not None:
            chunk_max = min(chunk_max, max_model_len)
            if count_prompt_tokens is not None:
                prompt_tokens = count_prompt_tokens(prompt)
                available_tokens = max_model_len - prompt_tokens
                if available_tokens <= 0:
                    finish_reason = FinishReason.CONTEXT_LIMIT
                    break
                chunk_max = min(chunk_max, available_tokens)
        if chunk_max <= 0:
            finish_reason = FinishReason.CONTEXT_LIMIT
            break

        old_prompt = prompt
        old_prompt_len = len(old_prompt)

        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": old_prompt,
            "stream": False,
            "temperature": temperature,
            "max_tokens": chunk_max,
            "skip_special_tokens": False,
            "logprobs": 1,
        }

        if normalized_regex is not None:
            payload["structured_outputs"] = {"regex": normalized_regex}
        if best_of > 1:
            payload["best_of"] = int(best_of)
        if top_k is not None:
            payload["top_k"] = top_k
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if stop_list:
            payload["stop"] = stop_list

        result = post_with_retry(payload)
        usage = result.get("usage")
        usage_completion_tokens = None
        if isinstance(usage, dict):
            completion_tokens_value = usage.get("completion_tokens")
            if isinstance(completion_tokens_value, int):
                usage_completion_tokens = completion_tokens_value

        choices = result.get("choices") or []
        if not choices:
            finish_reason = FinishReason.EMPTY_RESPONSE
            break

        best_choice = choices[0]
        best_score = float("-inf")
        for c in choices:
            ret_text = c.get("text", "") or ""
            prefix_len = old_prompt_len if ret_text.startswith(old_prompt) else 0
            lp_info = c.get("logprobs") or {}
            lp = lp_info.get("token_logprobs") or []
            offs = lp_info.get("text_offset") or []

            vals: list[float] = []
            if lp and offs and len(lp) == len(offs):
                for i in range(len(lp)):
                    if offs[i] < prefix_len:
                        continue
                    v = lp[i]
                    if isinstance(v, (int, float)) and not (
                            isinstance(v, float) and math.isnan(v)):
                        vals.append(float(v))
            else:
                for v in lp:
                    if isinstance(v, (int, float)) and not (
                            isinstance(v, float) and math.isnan(v)):
                        vals.append(float(v))

            score = (sum(vals) / len(vals)) if vals else float("-inf")
            if score > best_score:
                best_score = score
                best_choice = c

        stop_reason_raw = best_choice.get("stop_reason")
        stop_reason = stop_reason_raw if isinstance(stop_reason_raw,
                                                    str) else None
        finish_tag_raw = best_choice.get("finish_reason")
        finish_tag = finish_tag_raw if isinstance(finish_tag_raw,
                                                  str) else None

        ret_text = best_choice.get("text", "") or ""
        if ret_text.startswith(old_prompt):
            piece = ret_text[old_prompt_len:]
            prefix_len = old_prompt_len
        else:
            piece = ret_text
            prefix_len = 0

        hit_stop = False
        if stop_list:
            cut: int | None = None
            for s in stop_list:
                idx = piece.find(s)
                if idx != -1:
                    cut = idx if cut is None else min(cut, idx)
            if cut is not None:
                piece = piece[:cut]
                hit_stop = True

        if not piece:
            finish_override = _derive_finish_reason(
                hit_stop=hit_stop,
                stop_reason=stop_reason,
                finish_tag=finish_tag,
                stop_list=stop_list,
            )
            finish_reason = (finish_override
                             if finish_override is not None else
                             FinishReason.EMPTY_RESPONSE)
            produced = 0
            if usage_completion_tokens is not None:
                produced = usage_completion_tokens
            else:
                lp_info = best_choice.get("logprobs") or {}
                offs = lp_info.get("text_offset")
                toks = lp_info.get("tokens")
                if isinstance(offs, list) and isinstance(
                        toks, list) and len(offs) == len(toks):
                    for offset in offs:
                        if isinstance(offset, int) and offset >= prefix_len:
                            produced += 1
            assert produced > 0, "cannot infer generated token count from response payload"
            produced = min(produced, remaining)
            remaining -= produced
            generated_tokens += produced
            break

        out_parts.append(piece)
        prompt += piece
        if (max_model_len is not None and count_prompt_tokens is not None
                and remaining > 0 and count_prompt_tokens(prompt) >= max_model_len):
            finish_reason = FinishReason.CONTEXT_LIMIT

        trimmed_end = prefix_len + len(piece)
        produced = 0
        lp_info = best_choice.get("logprobs") or {}
        offs = lp_info.get("text_offset")
        toks = lp_info.get("tokens")
        if isinstance(offs, list) and isinstance(toks, list) and len(offs) == len(
                toks):
            for offset in offs:
                if isinstance(offset, int) and prefix_len <= offset < trimmed_end:
                    produced += 1
        if produced <= 0 and usage_completion_tokens is not None:
            produced = usage_completion_tokens
        assert produced > 0, "cannot infer generated token count from response payload"
        produced = min(produced, remaining)

        remaining -= produced
        generated_tokens += produced

        finish_override = _derive_finish_reason(
            hit_stop=hit_stop,
            stop_reason=stop_reason,
            finish_tag=finish_tag,
            stop_list=stop_list,
        )
        if finish_override in (FinishReason.STOP, FinishReason.STOP_SEQUENCE):
            finish_reason = finish_override
            break
        if finish_override == FinishReason.LENGTH:
            finish_reason = finish_override
            if remaining > 0:
                continue
            break
        if finish_reason == FinishReason.CONTEXT_LIMIT:
            break

    return "".join(out_parts), finish_reason, generated_tokens


def extract_chapters_info(text: str) -> list[dict[str, Any]]:
    def get_section(name: str) -> str:
        pattern = rf"^##\s+{re.escape(name)}\s*$([\s\S]*?)(?=^##\s+|\Z)"
        match = re.search(pattern, text, flags=re.M)
        return match.group(1) if match else ""

    chapter_names_section = get_section("Chapter Names")
    embedding_section = get_section("Chapters Embedding Space")
    word_count_section = get_section("Chapters Word Count")

    if not chapter_names_section:
        return []

    chapter_names = [
        name.strip()
        for name in re.findall(r"^\s*-\s+(.+?)\s*$", chapter_names_section,
                               flags=re.M)
    ]

    embedding_map = {}
    if embedding_section:
        embedding_map = {
            name.strip(): value.strip()
            for name, value in re.findall(
                r"^\s*####\s+(.+?)\s*$\n([^\n]+)",
                embedding_section,
                flags=re.M,
            )
        }

    word_count_map = {}
    if word_count_section:
        word_count_map = {
            name.strip(): int(value)
            for name, value in re.findall(
                r"^\s*####\s+(.+?)\s*$\n\s*(\d+)",
                word_count_section,
                flags=re.M,
            )
        }

    return [
        {
            "chapter_name": name,
            "chapter_word_count": int(word_count_map.get(name)),
            "chapter_embedding_space": embedding_map.get(name),
        }
        for name in chapter_names
    ]


def inverse_upper_bound(rounded: int) -> int:
    assert rounded >= 0, "rounded must be non-negative"

    def calc(value: int, step: int, lo: int, hi: int | float,
             pct: float) -> int | None:
        if value < lo or value > hi:
            return None
        if value % step != 0:
            return None

        q = value // step

        if q % 2 == 0:
            upper = math.floor((q + 0.5) * step)
        else:
            upper = math.ceil((q + 0.5) * step) - 1

        upper = min(upper, hi)
        if upper < lo:
            return None

        return math.ceil(upper * (1 + pct))

    candidates = [
        calc(rounded, 10, 0, 99, 0.50),
        calc(rounded, 50, 100, 999, 0.25),
        calc(rounded, 100, 1000, 9999, 0.15),
        calc(rounded, 500, 10000, float("inf"), 0.08),
    ]

    candidates = [x for x in candidates if x is not None]
    if candidates:
        return max(candidates)
    return rounded


def build_full_book_chapters_plan_regex(
        chapters_info: list[dict[str, Any]]) -> list[str]:
    def esc(value: Any) -> str:
        return re.escape(str(value))

    regexes: list[str] = []
    chapter_entries = chapters_info[1:]

    for idx, chapter_info in enumerate(chapter_entries):
        is_last = idx == len(chapter_entries) - 1
        separator = r"" if is_last else r"\n\n"

        regexes.append(
            r"### " + esc(chapter_info["chapter_name"]) + r"\n"
            r"\*\*Word Count:\*\* " + esc(
                chapter_info["chapter_word_count"]) + r"\n"
            r"\*\*Embedding Space:\*\* " + esc(
                chapter_info["chapter_embedding_space"]) + r"\n"
            r"\*\*Narrative Focuses:\*\* " + NARRATIVE_FOCUSES_RE + r"\n"
            r"\*\*Chapter Summary:\*\*\n"
            r"(?:" + BULLET_RE + r"\n){0,99}" + BULLET_RE + separator)

    return regexes


def process_prompt(
    prompt: str,
    temperature: float,
    *,
    llm_call: Callable[..., tuple[str, FinishReason, int]],
) -> tuple[str, bool]:
    def finish_due_to_stop_or_context_limit(reason: FinishReason) -> bool:
        return reason in (FinishReason.STOP, FinishReason.CONTEXT_LIMIT)

    def finish_due_to_budget_limit(
        reason: FinishReason,
        *,
        generated: int,
        requested_max_new_tokens: int,
    ) -> bool:
        return (reason == FinishReason.LENGTH
                and generated >= requested_max_new_tokens)

    all_eos_token_correct = True
    num_generated_tokens = 0
    prompt_prefix = (
        f"<|start_header_id|>prompt<|stop_header_id|>\n\n{prompt}<|eot_id|>"
    )

    carry = prompt_prefix + "<|start_header_id|>book_preview<|stop_header_id|>\n\n"

    book_preview_max_new_tokens = BOOK_PREVIEW_MAX_NEW_TOKENS
    book_preview, finish_reason, generated = llm_call(
        carry,
        temperature=temperature,
        max_new_tokens=book_preview_max_new_tokens,
        top_k=20,
        regex=BOOK_PREVIEW_RE,
    )
    num_generated_tokens += generated
    _debug_stage(
        "book_preview",
        finish_reason=finish_reason,
        generated=generated,
        cumulative_tokens=num_generated_tokens,
    )
    book_preview_hit_budget = finish_due_to_budget_limit(
        finish_reason,
        generated=generated,
        requested_max_new_tokens=book_preview_max_new_tokens,
    )
    all_eos_token_correct = all_eos_token_correct and (
        finish_due_to_stop_or_context_limit(finish_reason)
        or book_preview_hit_budget)
    if finish_reason == FinishReason.STOP:
        book_preview = book_preview + "<|eot_id|>"
    if finish_reason == FinishReason.CONTEXT_LIMIT or book_preview_hit_budget:
        carry = carry + book_preview
        generation = carry.replace(prompt_prefix, "", 1)
        return generation, all_eos_token_correct

    carry = carry + book_preview + "<|start_header_id|>book_plan<|stop_header_id|>\n\n"

    book_plan_max_new_tokens = BOOK_PLAN_MAX_NEW_TOKENS
    book_plan, finish_reason, generated = llm_call(
        carry,
        temperature=temperature,
        max_new_tokens=book_plan_max_new_tokens,
    )
    num_generated_tokens += generated
    _debug_stage(
        "book_plan",
        finish_reason=finish_reason,
        generated=generated,
        cumulative_tokens=num_generated_tokens,
    )
    book_plan_hit_budget = finish_due_to_budget_limit(
        finish_reason,
        generated=generated,
        requested_max_new_tokens=book_plan_max_new_tokens,
    )
    all_eos_token_correct = all_eos_token_correct and (
        finish_due_to_stop_or_context_limit(finish_reason)
        or book_plan_hit_budget)
    if finish_reason == FinishReason.STOP:
        book_plan = book_plan + "<|eot_id|>"
    if finish_reason == FinishReason.CONTEXT_LIMIT or book_plan_hit_budget:
        carry = carry + book_plan
        generation = carry.replace(prompt_prefix, "", 1)
        return generation, all_eos_token_correct

    carry = carry + book_plan

    try:
        chapters_info = extract_chapters_info(book_plan)
        assert len(chapters_info) > 2
    except Exception:
        generation = carry.replace(prompt_prefix, "", 1)
        return generation, all_eos_token_correct

    carry = carry + "<|start_header_id|>first_chapter_plan<|stop_header_id|>\n\n"

    first_chapter_info = chapters_info[0]
    first_chapter_plan_prefix = (
        f"# {first_chapter_info['chapter_name']}\n\n### Chapters Plan\n"
        f"**Word Count:** {first_chapter_info['chapter_word_count']}\n"
        f"**Embedding Space:** {first_chapter_info['chapter_embedding_space']}\n"
        "**Narrative Focuses:** "
    )

    first_chapter_plan_max_new_tokens = FIRST_CHAPTER_PLAN_MAX_NEW_TOKENS
    first_chapter_plan, finish_reason, generated = llm_call(
        carry + first_chapter_plan_prefix,
        temperature=temperature,
        max_new_tokens=first_chapter_plan_max_new_tokens,
        regex=FIRST_CHAPTER_PLAN_RE,
    )
    num_generated_tokens += generated
    _debug_stage(
        "first_chapter_plan",
        finish_reason=finish_reason,
        generated=generated,
        cumulative_tokens=num_generated_tokens,
    )
    first_chapter_plan = first_chapter_plan_prefix + first_chapter_plan
    first_chapter_plan_hit_budget = finish_due_to_budget_limit(
        finish_reason,
        generated=generated,
        requested_max_new_tokens=first_chapter_plan_max_new_tokens,
    )
    all_eos_token_correct = all_eos_token_correct and (
        finish_due_to_stop_or_context_limit(finish_reason)
        or first_chapter_plan_hit_budget)
    if finish_reason == FinishReason.STOP:
        first_chapter_plan = first_chapter_plan + "<|eot_id|>"
    if (finish_reason == FinishReason.CONTEXT_LIMIT
            or first_chapter_plan_hit_budget):
        carry = carry + first_chapter_plan
        generation = carry.replace(prompt_prefix, "", 1)
        return generation, all_eos_token_correct

    carry = carry + first_chapter_plan + "<|start_header_id|>first_chapter_text<|stop_header_id|>\n\n"

    first_chapter_prefix = f"### {first_chapter_info['chapter_name']}\n```\n"
    first_chapter_max_new_tokens = int(
        inverse_upper_bound(first_chapter_info["chapter_word_count"]) * 2)
    first_chapter, finish_reason, generated = llm_call(
        carry + first_chapter_prefix,
        temperature=temperature,
        max_new_tokens=first_chapter_max_new_tokens,
    )
    num_generated_tokens += generated
    _debug_stage(
        "first_chapter_text",
        finish_reason=finish_reason,
        generated=generated,
        cumulative_tokens=num_generated_tokens,
    )
    first_chapter = first_chapter_prefix + first_chapter
    first_chapter_hit_budget = finish_due_to_budget_limit(
        finish_reason,
        generated=generated,
        requested_max_new_tokens=first_chapter_max_new_tokens,
    )
    all_eos_token_correct = all_eos_token_correct and (
        finish_due_to_stop_or_context_limit(finish_reason)
        or first_chapter_hit_budget)
    if finish_reason == FinishReason.STOP:
        first_chapter = first_chapter + "<|eot_id|>"
    if finish_reason == FinishReason.CONTEXT_LIMIT or first_chapter_hit_budget:
        carry = carry + first_chapter
        generation = carry.replace(prompt_prefix, "", 1)
        return generation, all_eos_token_correct

    carry = carry + first_chapter + "<|start_header_id|>full_book_chapters_plan<|stop_header_id|>\n\n"

    chapter_plan_max_new_tokens = min(1500 * (len(chapters_info) - 1),
                                      (MAX_TOKENS - 3000 -
                                       num_generated_tokens))
    chapter_plan, finish_reason, generated = llm_call(
        carry,
        temperature=temperature,
        max_new_tokens=chapter_plan_max_new_tokens,
        regex=build_full_book_chapters_plan_regex(chapters_info),
    )
    num_generated_tokens += generated
    chapter_plan_hit_budget = finish_due_to_budget_limit(
        finish_reason,
        generated=generated,
        requested_max_new_tokens=chapter_plan_max_new_tokens,
    )
    _debug_stage(
        "full_book_chapters_plan",
        finish_reason=finish_reason,
        generated=generated,
        cumulative_tokens=num_generated_tokens,
    )
    all_eos_token_correct = all_eos_token_correct and (
        (finish_reason == FinishReason.STOP)
        or (finish_reason == FinishReason.CONTEXT_LIMIT)
        or chapter_plan_hit_budget
        or (num_generated_tokens >= (MAX_TOKENS - 3000)))
    if finish_reason == FinishReason.STOP:
        chapter_plan = chapter_plan.strip() + "<|eot_id|>"
    if finish_reason == FinishReason.CONTEXT_LIMIT or chapter_plan_hit_budget:
        carry = carry + chapter_plan.strip()
        generation = carry.replace(prompt_prefix, "", 1)
        return generation, all_eos_token_correct

    carry = carry + chapter_plan.strip() + "<|start_header_id|>book_characters_list<|stop_header_id|>\n\n"

    try:
        characters_list_max_new_tokens = CHARACTERS_LIST_MAX_NEW_TOKENS
        characters_list, finish_reason, _ = llm_call(
            carry,
            temperature=temperature,
            max_new_tokens=characters_list_max_new_tokens,
            regex=(
                r"\A### Main Characters\n"
                r"(?:#### (?:\S+(?:[^\S\n]+\S+){0,7})\n(?:(?:" + BULLET_RE +
                r")\n){1,20}\n?)+"
                r"### Side Characters\n"
                r"(?:#### (?:\S+(?:[^\S\n]+\S+){0,7})\n(?:(?:" + BULLET_RE +
                r")\n){1,20}\n?)*"
                r"#### (?:\S+(?:[^\S\n]+\S+){0,7})\n(?:(?:" + BULLET_RE +
                r")\n){0,19}" + BULLET_RE + r"\z"
            ),
        )
    except Exception:
        generation = carry.replace(prompt_prefix, "", 1)
        return generation, all_eos_token_correct

    _debug_stage(
        "book_characters_list",
        finish_reason=finish_reason,
        generated=0,
        cumulative_tokens=num_generated_tokens,
    )
    characters_list_hit_budget = finish_due_to_budget_limit(
        finish_reason,
        generated=0,
        requested_max_new_tokens=characters_list_max_new_tokens,
    )
    all_eos_token_correct = all_eos_token_correct and (
        finish_due_to_stop_or_context_limit(finish_reason)
        or characters_list_hit_budget)
    if finish_reason == FinishReason.STOP:
        characters_list = characters_list + "<|eot_id|>"
    if finish_reason == FinishReason.CONTEXT_LIMIT or characters_list_hit_budget:
        carry = carry + characters_list
        generation = carry.replace(prompt_prefix, "", 1)
        return generation, all_eos_token_correct

    carry = carry + characters_list

    try:
        for chapter_info in chapters_info[1:]:
            scene_breakdown_prefix = (
                "<|start_header_id|>scene_breakdown<|stop_header_id|>\n\n"
                f"### {chapter_info['chapter_name']}\n#### Scene 1: "
            )
            scene_breakdown_max_new_tokens = SCENE_BREAKDOWN_MAX_NEW_TOKENS
            scene_breakdown, finish_reason, generated = llm_call(
                carry + scene_breakdown_prefix,
                temperature=temperature,
                max_new_tokens=scene_breakdown_max_new_tokens,
                regex=(
                    r"\A"
                    r"(?:\S+(?:[^\S\n]+\S+){0,11})\n"
                    r"\*\*Word Count:\*\* [0-9]{2,5}\n"
                    r"\*\*Embedding Space:\*\* " + EMBEDDING_SPACE_RE +
                    r"\*\*Narrative Focus:\*\*[^\S\n]+" + NARRATIVE_FOCUSES_RE
                    + r"\n"
                    r"\*\*Narrative Perspective:\*\* [^\n]{1,120}\n"
                    r"\*\*Scene Summary:\*\*\n"
                    r"(?:" + BULLET_RE + r"\n){0,16}" + BULLET_RE + r"\n?"
                    r"(?:(?:\n\n|\n)"
                    r"#### Scene (?:[2-9]|1[0-2]): "
                    r"(?:\S+(?:[^\S\n]+\S+){0,11})\n"
                    r"\*\*Word Count:\*\* [0-9]{2,5}\n"
                    r"\*\*Embedding Space:\*\* " + EMBEDDING_SPACE_RE +
                    r"\*\*Narrative Focus:\*\*[^\S\n]+" +
                    NARRATIVE_FOCUSES_RE + r"\n"
                    r"\*\*Narrative Perspective:\*\* [^\n]{1,120}\n"
                    r"\*\*Scene Summary:\*\*\n"
                    r"(?:" + BULLET_RE + r"\n){0,16}" + BULLET_RE +
                    r"\n?){0,11}"
                    r"\z"
                ),
            )

            scene_breakdown = scene_breakdown_prefix + scene_breakdown
            _debug_stage(
                f"scene_breakdown:{chapter_info['chapter_name']}",
                finish_reason=finish_reason,
                generated=generated,
                cumulative_tokens=num_generated_tokens,
            )
            scene_breakdown_hit_budget = finish_due_to_budget_limit(
                finish_reason,
                generated=generated,
                requested_max_new_tokens=scene_breakdown_max_new_tokens,
            )
            all_eos_token_correct = all_eos_token_correct and (
                finish_due_to_stop_or_context_limit(finish_reason)
                or scene_breakdown_hit_budget)
            if finish_reason == FinishReason.STOP:
                scene_breakdown = scene_breakdown + "<|eot_id|>"
            if (finish_reason == FinishReason.CONTEXT_LIMIT
                    or scene_breakdown_hit_budget):
                carry = carry + scene_breakdown
                generation = carry.replace(prompt_prefix, "", 1)
                return generation, all_eos_token_correct

            carry = carry + scene_breakdown + "<|start_header_id|>chapter_text<|stop_header_id|>\n\n"

            chapter_text_prefix = f"### {chapter_info['chapter_name']}\n```\n"
            chapter_text_max_new_tokens = int(
                inverse_upper_bound(chapter_info["chapter_word_count"]) * 2)
            chapter_text, finish_reason, generated = llm_call(
                carry + chapter_text_prefix,
                temperature=temperature,
                max_new_tokens=chapter_text_max_new_tokens,
            )

            chapter_text = chapter_text_prefix + chapter_text
            _debug_stage(
                f"chapter_text:{chapter_info['chapter_name']}",
                finish_reason=finish_reason,
                generated=generated,
                cumulative_tokens=num_generated_tokens,
            )
            chapter_text_hit_budget = finish_due_to_budget_limit(
                finish_reason,
                generated=generated,
                requested_max_new_tokens=chapter_text_max_new_tokens,
            )
            all_eos_token_correct = all_eos_token_correct and (
                finish_due_to_stop_or_context_limit(finish_reason)
                or chapter_text_hit_budget)
            if finish_reason == FinishReason.STOP:
                chapter_text = chapter_text + "<|eot_id|>"
            if finish_reason == FinishReason.CONTEXT_LIMIT or chapter_text_hit_budget:
                carry = carry + chapter_text
                generation = carry.replace(prompt_prefix, "", 1)
                return generation, all_eos_token_correct

            carry = carry + chapter_text
    except Exception:
        pass

    generation = carry.replace(prompt_prefix, "", 1)
    return generation, all_eos_token_correct


@dataclass
class _ServerContext:
    base_url: str
    server_ip: str
    server_port: int
    model_name: str
    tokenizer_dir: str
    max_model_len: int
    checkpoint_path: str
    process: subprocess.Popen
    stdout_path: str
    stderr_path: str


def _build_llm_call(
    ctx: _ServerContext,
    *,
    request_timeout: float,
    block_length: int | tuple[int, int] | None = None,
) -> Callable[..., tuple[str, FinishReason, int]]:
    token_counter = _get_prompt_token_counter(ctx.tokenizer_dir)
    return cast(
        Callable[..., tuple[str, FinishReason, int]],
        lambda text, **kwargs: call_llm(
            text,
            server_ip=ctx.server_ip,
            server_port=ctx.server_port,
            model_name=ctx.model_name,
            request_timeout=request_timeout,
            block_length=block_length,
            max_model_len=ctx.max_model_len,
            count_prompt_tokens=token_counter,
            **kwargs,
        ),
    )


def _book_preview_carry(prompt: str) -> str:
    prompt_prefix = (
        f"<|start_header_id|>prompt<|stop_header_id|>\n\n{prompt}<|eot_id|>"
    )
    return prompt_prefix + "<|start_header_id|>book_preview<|stop_header_id|>\n\n"


def _request_book_preview_raw_completion(
    ctx: _ServerContext,
    *,
    prompt: str,
    temperature: float,
    request_timeout: float,
) -> dict[str, Any]:
    max_tokens = _env_int(
        "TPU_ONLINE_RL_PROD_EOS_BOOK_PREVIEW_MAX_NEW_TOKENS",
        max(BOOK_PREVIEW_MAX_NEW_TOKENS, 512),
    )
    payload: dict[str, Any] = {
        "model": ctx.model_name,
        "prompt": _book_preview_carry(prompt),
        "stream": False,
        "temperature": temperature,
        "top_k": 20,
        "max_tokens": max_tokens,
        "skip_special_tokens": False,
        "logprobs": 1,
        "return_token_ids": True,
        "return_native_token_logprobs": True,
        "structured_outputs": {
            "regex": BOOK_PREVIEW_RE
        },
    }
    response = requests.post(
        f"{ctx.base_url}/v1/completions",
        json=payload,
        timeout=request_timeout,
    )
    assert response.status_code == 200, (
        f"book_preview completion failed: {response.status_code} "
        f"{response.text[:4000]}")
    return cast(dict[str, Any], response.json())


def _assert_eos_returned_for_completion_components(
    payload: dict[str, Any],
    *,
    eos_token_id: int,
    eos_text: str,
) -> None:
    choices = payload.get("choices")
    assert isinstance(choices, list) and choices, "completion returned no choices"
    choice = choices[0]
    assert isinstance(choice, dict), "completion choice must be a dictionary"
    assert choice.get("finish_reason") == "stop"

    text = choice.get("text")
    assert isinstance(text, str) and text.endswith(eos_text), (
        "completion text did not include the final EOS token")

    token_ids = choice.get("token_ids")
    assert isinstance(token_ids, list) and token_ids, (
        "completion did not return generated token_ids")
    assert token_ids[-1] == eos_token_id

    logprobs = choice.get("logprobs")
    assert isinstance(logprobs, dict), "completion did not return logprobs"
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    text_offsets = logprobs.get("text_offset")
    top_logprobs = logprobs.get("top_logprobs")
    assert isinstance(tokens, list) and tokens, "logprobs.tokens missing"
    assert isinstance(token_logprobs,
                      list) and token_logprobs, "token_logprobs missing"
    assert isinstance(text_offsets, list) and text_offsets, "text_offset missing"
    assert isinstance(top_logprobs, list) and top_logprobs, "top_logprobs missing"
    assert len(tokens) == len(token_ids)
    assert len(token_logprobs) == len(token_ids)
    assert len(text_offsets) == len(token_ids)
    assert len(top_logprobs) == len(token_ids)

    assert tokens[-1] == eos_text
    assert text_offsets[-1] == len(text) - len(eos_text)
    eos_logprob = token_logprobs[-1]
    assert isinstance(eos_logprob, (int, float))
    assert math.isfinite(float(eos_logprob))

    eos_top_logprobs = top_logprobs[-1]
    assert isinstance(eos_top_logprobs, dict)
    assert eos_text in eos_top_logprobs
    assert math.isfinite(float(eos_top_logprobs[eos_text]))

    usage = payload.get("usage")
    assert isinstance(usage, dict), "completion did not return usage"
    assert usage.get("completion_tokens") == len(token_ids)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_status(base_url: str, timeout_s: int) -> int:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/status", timeout=3)
            if resp.status_code in (200, 503):
                return resp.status_code
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("status endpoint did not become reachable in time")


def _wait_ready(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.get(f"{base_url}/status", timeout=5)
        if resp.status_code == 200 and resp.json().get("ready") is True:
            return
        time.sleep(2)
    raise TimeoutError("server did not become ready in time")


def _reload(
    base_url: str,
    checkpoint_path: str,
    timeout_s: int,
    *,
    wait_for_inflight_requests: bool = False,
) -> requests.Response:
    return requests.post(
        f"{base_url}/v1/reload_weights",
        json={
            "checkpoint_path": checkpoint_path,
            "wait_for_inflight_requests": wait_for_inflight_requests,
            "clear_cache": True,
            "release_kv_cache": True,
            "timeout_seconds": float(timeout_s),
        },
        timeout=timeout_s + 120,
    )


def _resolve_model_info(
    base_url: str,
    fallback_model_name: str,
    fallback_max_model_len: int,
) -> tuple[str, int]:
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=30)
        if resp.status_code != 200:
            return fallback_model_name, fallback_max_model_len
        payload = resp.json()
        items = payload.get("data")
        if isinstance(items, list) and items and isinstance(items[0], dict):
            model_id = items[0].get("id")
            max_model_len = items[0].get("max_model_len")
            resolved_model_name = (
                model_id if isinstance(model_id, str) and model_id else
                fallback_model_name)
            resolved_max_model_len = (
                max_model_len
                if isinstance(max_model_len, int) and max_model_len > 0
                else fallback_max_model_len
            )
            return resolved_model_name, resolved_max_model_len
    except requests.RequestException:
        pass
    return fallback_model_name, fallback_max_model_len


def _reset_server_for_test(ctx: _ServerContext) -> None:
    timeout_s = _env_int("TPU_ONLINE_RL_E2E_RELOAD_TIMEOUT_SECONDS", 7200)
    resp = _reload(
        ctx.base_url,
        ctx.checkpoint_path,
        timeout_s,
        wait_for_inflight_requests=True,
    )
    assert resp.status_code == 200, (
        f"test reset reload failed: {resp.status_code} {resp.text[:2000]}")
    _wait_ready(ctx.base_url, timeout_s)


@pytest.fixture(scope="module")
def live_server_ctx() -> _ServerContext:
    model_dir = os.environ.get("TPU_ONLINE_RL_E2E_MODEL_DIR",
                               _default_model_dir())
    tokenizer_dir = os.environ.get("TPU_ONLINE_RL_E2E_TOKENIZER_DIR",
                                   model_dir)
    checkpoint_path = os.environ.get("TPU_ONLINE_RL_E2E_RELOAD_CHECKPOINT",
                                     _default_orbax_checkpoint())
    max_model_len = int(os.environ.get("TPU_ONLINE_RL_E2E_MAX_MODEL_LEN", "16384"))

    port = _pick_free_port()
    host = "127.0.0.1"
    base_url = f"http://{host}:{port}"
    repo_root = Path(__file__).resolve().parents[2]

    stdout_tmp = tempfile.NamedTemporaryFile(prefix="prod-harness-stdout-",
                                             suffix=".log",
                                             delete=False)
    stderr_tmp = tempfile.NamedTemporaryFile(prefix="prod-harness-stderr-",
                                             suffix=".log",
                                             delete=False)
    stdout_path = stdout_tmp.name
    stderr_path = stderr_tmp.name
    stdout_tmp.close()
    stderr_tmp.close()

    command = [
        sys.executable,
        "-m",
        "tpu_inference.entrypoints.online_rl_server",
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        os.environ.get("TPU_ONLINE_RL_E2E_TP", "4"),
        "--max-model-len",
        os.environ.get("TPU_ONLINE_RL_E2E_MAX_MODEL_LEN", "16384"),
        "--max-num-batched-tokens",
        os.environ.get("TPU_ONLINE_RL_E2E_MAX_BATCHED_TOKENS", "16384"),
        "--max-num-seqs",
        os.environ.get("TPU_ONLINE_RL_E2E_MAX_NUM_SEQS", "16"),
        "--model",
        model_dir,
        "--tokenizer",
        tokenizer_dir,
        "--model-weights",
        "/tmp/tpu-online-rl-prod-harness-dummy-startup",
    ]
    if os.environ.get("TPU_ONLINE_RL_E2E_DISABLE_PREFIX_CACHING", "0") == "1":
        command.append("--no-enable-prefix-caching")

    env = os.environ.copy()
    env.setdefault("MODEL_IMPL_TYPE", "flax_nnx")
    env.setdefault("SKIP_JAX_PRECOMPILE", "1")
    env.setdefault("VLLM_XLA_CHECK_RECOMPILATION", "0")

    with open(stdout_path, "wb") as so, open(stderr_path, "wb") as se:
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=so,
            stderr=se,
            preexec_fn=os.setsid,
        )

    ctx = _ServerContext(
        base_url=base_url,
        server_ip=host,
        server_port=port,
        model_name=model_dir,
        tokenizer_dir=tokenizer_dir,
        max_model_len=max_model_len,
        checkpoint_path=checkpoint_path,
        process=process,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    try:
        _wait_for_status(base_url,
                         _env_int("TPU_ONLINE_RL_E2E_SERVER_START_TIMEOUT_SECONDS",
                                  1800))
        reload_timeout = _env_int("TPU_ONLINE_RL_E2E_RELOAD_TIMEOUT_SECONDS", 7200)
        reload_resp = _reload(base_url, checkpoint_path, reload_timeout)
        assert reload_resp.status_code == 200, (
            f"initial reload failed: {reload_resp.status_code} "
            f"{reload_resp.text[:2000]}")
        _wait_ready(base_url, reload_timeout)
        ctx.model_name, ctx.max_model_len = _resolve_model_info(
            base_url,
            model_dir,
            max_model_len,
        )
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


class _ReloadLoop:

    def __init__(
        self,
        *,
        ctx: _ServerContext,
        interval_seconds: float,
        max_cycles: int,
        timeout_seconds: int,
        wait_for_inflight_requests: bool = False,
    ):
        self._ctx = ctx
        self._interval_seconds = interval_seconds
        self._max_cycles = max_cycles
        self._timeout_seconds = timeout_seconds
        self._wait_for_inflight_requests = wait_for_inflight_requests
        self._stop = threading.Event()
        self.success_count = 0
        self.failure_count = 0
        self.last_error: str | None = None
        self.cycle_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=60)

    def _run(self) -> None:
        while not self._stop.is_set() and self.cycle_count < self._max_cycles:
            try:
                resp = _reload(
                    self._ctx.base_url,
                    self._ctx.checkpoint_path,
                    self._timeout_seconds,
                    wait_for_inflight_requests=self._wait_for_inflight_requests,
                )
                if resp.status_code == 200:
                    self.success_count += 1
                else:
                    self.failure_count += 1
                    self.last_error = (
                        f"reload HTTP {resp.status_code}: {resp.text[:1000]}")
            except requests.RequestException as exc:
                self.failure_count += 1
                self.last_error = str(exc)
            self.cycle_count += 1
            self._stop.wait(self._interval_seconds)


def test_book_preview_native_logprobs_returns_eos_for_all_completion_components(
    live_server_ctx: _ServerContext,
) -> None:
    _reset_server_for_test(live_server_ctx)

    prompt = os.environ.get(
        "TPU_ONLINE_RL_PROD_TEST_PROMPT",
        (
            "Write a gritty fantasy setup where a city of scholars hides a "
            "violent underworld, and a disgraced archivist uncovers a conspiracy."
        ),
    )
    payload = _request_book_preview_raw_completion(
        live_server_ctx,
        prompt=prompt,
        temperature=_env_float("TPU_ONLINE_RL_PROD_TEMPERATURE", 0.0),
        request_timeout=_env_float("TPU_ONLINE_RL_PROD_REQUEST_TIMEOUT", 7200.0),
    )
    eos_token_id, eos_text = _get_eos_token_info(live_server_ctx.tokenizer_dir)

    _assert_eos_returned_for_completion_components(
        payload,
        eos_token_id=eos_token_id,
        eos_text=eos_text,
    )


def test_production_generation_harness_with_live_reloads(
    live_server_ctx: _ServerContext,
) -> None:
    _reset_server_for_test(live_server_ctx)

    prompt = os.environ.get(
        "TPU_ONLINE_RL_PROD_TEST_PROMPT",
        (
            "Write a gritty fantasy setup where a city of scholars hides a "
            "violent underworld, and a disgraced archivist uncovers a conspiracy."
        ),
    )
    temperature = _env_float("TPU_ONLINE_RL_PROD_TEMPERATURE", 0.0)
    request_timeout = _env_float("TPU_ONLINE_RL_PROD_REQUEST_TIMEOUT", 7200.0)
    block_min = _env_int("TPU_ONLINE_RL_PROD_BLOCK_MIN", 128)
    block_max = _env_int("TPU_ONLINE_RL_PROD_BLOCK_MAX", 384)

    llm_call = _build_llm_call(
        live_server_ctx,
        request_timeout=request_timeout,
        block_length=(block_min, block_max),
    )

    reload_loop = _ReloadLoop(
        ctx=live_server_ctx,
        interval_seconds=_env_float("TPU_ONLINE_RL_PROD_RELOAD_INTERVAL_SECONDS",
                                    0.2),
        max_cycles=_env_int("TPU_ONLINE_RL_PROD_RELOAD_MAX_CYCLES", 4),
        timeout_seconds=_env_int("TPU_ONLINE_RL_E2E_RELOAD_TIMEOUT_SECONDS", 7200),
    )

    reload_loop.start()
    try:
        generation, all_eos_token_correct = process_prompt(
            prompt, temperature, llm_call=llm_call)
    finally:
        reload_loop.stop()

    assert generation, "production harness returned empty generation."
    assert all_eos_token_correct, "production harness observed unexpected EOS handling."
    assert reload_loop.cycle_count > 0, "reload loop did not run any cycle."
    assert reload_loop.success_count > 0, (
        f"no successful live reload observed; last_error={reload_loop.last_error}")
    assert reload_loop.failure_count == 0, (
        f"reload loop observed failures; last_error={reload_loop.last_error}")


def test_production_harness_identical_output_with_continuous_reloads(
    live_server_ctx: _ServerContext,
) -> None:
    _reset_server_for_test(live_server_ctx)

    prompt = os.environ.get(
        "TPU_ONLINE_RL_PROD_TEST_PROMPT",
        (
            "Write a gritty fantasy setup where a city of scholars hides a "
            "violent underworld, and a disgraced archivist uncovers a conspiracy."
        ),
    )
    temperature = 0.0
    request_timeout = _env_float("TPU_ONLINE_RL_PROD_REQUEST_TIMEOUT", 7200.0)
    block_min = _env_int("TPU_ONLINE_RL_PROD_BLOCK_MIN", 128)
    block_max = _env_int("TPU_ONLINE_RL_PROD_BLOCK_MAX", 384)
    equality_block_length = _env_int("TPU_ONLINE_RL_PROD_EQUALITY_BLOCK_LENGTH",
                                     block_max)

    llm_call = _build_llm_call(
        live_server_ctx,
        request_timeout=request_timeout,
        block_length=equality_block_length,
    )

    baseline_generation, baseline_all_eos = process_prompt(
        prompt, temperature, llm_call=llm_call)
    assert baseline_generation, "baseline generation returned empty output."
    assert baseline_all_eos, "baseline generation observed unexpected EOS handling."

    ngram_n = _env_int("TPU_ONLINE_RL_PROD_NGRAM_N", 1)
    ngram_min_f1 = _env_float("TPU_ONLINE_RL_PROD_NGRAM_MIN_F1", 0.9)
    control_f1_tolerance = _env_float("TPU_ONLINE_RL_PROD_CONTROL_F1_TOLERANCE",
                                      0.02)

    control_generation, control_all_eos = process_prompt(
        prompt, temperature, llm_call=llm_call)
    assert control_generation, "control generation returned empty output."
    assert control_all_eos, "control generation observed unexpected EOS handling."
    control_ngram_f1 = _ngram_multiset_f1(
        baseline_generation,
        control_generation,
        n=ngram_n,
    )
    required_ngram_f1 = max(
        0.0,
        min(ngram_min_f1, control_ngram_f1) - control_f1_tolerance,
    )

    reload_timeout = _env_int("TPU_ONLINE_RL_E2E_RELOAD_TIMEOUT_SECONDS", 7200)
    reload_interval = _env_float("TPU_ONLINE_RL_PROD_RELOAD_INTERVAL_SECONDS", 0.2)
    reload_cycles = _env_int("TPU_ONLINE_RL_PROD_EQUALITY_RELOAD_MAX_CYCLES", 1000)

    equality_runs = _env_int("TPU_ONLINE_RL_PROD_EQUALITY_RUNS", 2)
    assert equality_runs > 0, "TPU_ONLINE_RL_PROD_EQUALITY_RUNS must be > 0."

    for run_index in range(equality_runs):
        reload_loop = _ReloadLoop(
            ctx=live_server_ctx,
            interval_seconds=reload_interval,
            max_cycles=reload_cycles,
            timeout_seconds=reload_timeout,
            wait_for_inflight_requests=True,
        )
        reload_loop.start()
        try:
            generation, all_eos_token_correct = process_prompt(
                prompt, temperature, llm_call=llm_call)
        finally:
            reload_loop.stop()

        assert generation, f"reload run {run_index + 1} returned empty output."
        assert all_eos_token_correct, (
            f"reload run {run_index + 1} observed unexpected EOS handling.")
        assert reload_loop.cycle_count > 0, (
            f"reload run {run_index + 1} did not execute any reload cycle.")
        assert reload_loop.success_count > 0, (
            f"reload run {run_index + 1} had no successful reload; "
            f"last_error={reload_loop.last_error}")
        assert reload_loop.failure_count == 0, (
            f"reload run {run_index + 1} observed reload failures; "
            f"last_error={reload_loop.last_error}")

        ngram_f1 = _ngram_multiset_f1(
            baseline_generation,
            generation,
            n=ngram_n,
        )
        assert ngram_f1 >= required_ngram_f1, (
            f"reload run {run_index + 1} ngram F1 below threshold.\n"
            f"n={ngram_n}\n"
            f"control_f1={control_ngram_f1:.6f}\n"
            f"observed_f1={ngram_f1:.6f}\n"
            f"required_f1>={required_ngram_f1:.6f}\n"
            f"exact_match={generation == baseline_generation}\n"
            f"baseline_len={len(baseline_generation)}\n"
            f"run_len={len(generation)}"
        )
