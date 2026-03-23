# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from http import HTTPStatus
from types import SimpleNamespace

import llguidance
import llguidance.hf as llguidance_hf
from pydantic import ValidationError
from transformers import AutoTokenizer

from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponse,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse, UsageInfo
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

from tpu_inference.entrypoints import online_rl_server
from tpu_inference.entrypoints.stacked_regex import (
    TPUCompletionRequest,
    StagedGuidanceGrammar,
    extract_stacked_regexes,
    install_staged_guidance_patch,
    normalize_completion_request,
    validate_stacked_regexes,
)


class _FakeRawRequest:

    def __init__(self, *, backend: str):
        self.headers = {}
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                args=SimpleNamespace(
                    structured_outputs_config=SimpleNamespace(backend=backend)
                )
            )
        )


class _SamplingParamsCapturingHandler:

    def __init__(self):
        self.requests = []
        self.sampling_params = []

    async def create_completion(self, request, _raw_request) -> CompletionResponse:
        self.requests.append(request)
        self.sampling_params.append(request.to_sampling_params(32, None, None))
        return CompletionResponse(
            model="model",
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text="",
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(),
        )

    def create_error_response(
        self,
        message,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorInfo(
                message=str(message),
                type=err_type,
                code=int(status_code),
                param=param,
            )
        )


class _FakeMatcher:

    def __init__(self, grammar):
        self._grammar = grammar

    def is_accepting(self) -> bool:
        return self._grammar.is_accepting


class _FakeStageGrammar:

    def __init__(self, expected_tokens: list[int]) -> None:
        self.expected_tokens = expected_tokens
        self.consumed: list[int] = []
        self.ll_matcher = _FakeMatcher(self)

    @property
    def is_accepting(self) -> bool:
        return len(self.consumed) == len(self.expected_tokens)

    def accept_tokens(self, _request_id: str, tokens: list[int]) -> bool:
        for token in tokens:
            token_index = len(self.consumed)
            if token_index >= len(self.expected_tokens):
                return False
            if self.expected_tokens[token_index] != token:
                return False
            self.consumed.append(token)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        accepted: list[int] = []
        token_index = len(self.consumed)
        for token in tokens:
            if token_index >= len(self.expected_tokens):
                break
            if self.expected_tokens[token_index] != token:
                break
            accepted.append(token)
            token_index += 1
        return accepted

    def rollback(self, num_tokens: int) -> None:
        del self.consumed[-num_tokens:]

    def fill_bitmask(self, bitmask, idx: int) -> None:
        if self.is_accepting:
            bitmask[idx] = "EOS"
        else:
            bitmask[idx] = self.expected_tokens[len(self.consumed)]

    def is_terminated(self) -> bool:
        return False

    def reset(self):
        self.consumed.clear()


class _FakeGuidanceBackend:

    def __init__(self, expected_by_regex: dict[str, list[int]]) -> None:
        self.expected_by_regex = expected_by_regex
        self.compile_calls: list[str] = []

    def compile_grammar(self, request_type, grammar_spec):
        self.compile_calls.append(grammar_spec)
        assert request_type == StructuredOutputOptions.REGEX
        return _FakeStageGrammar(self.expected_by_regex[grammar_spec])


class TestStackedRegexNormalization(unittest.TestCase):

    def test_tpu_completion_request_accepts_regex_list(self):
        request = TPUCompletionRequest.model_validate(
            {
                "model": "model",
                "prompt": "prompt",
                "structured_outputs": {"regex": ["foo+", "bar?"]},
            }
        )

        self.assertIsNotNone(request.structured_outputs)
        self.assertEqual(request.structured_outputs.regex, ["foo+", "bar?"])

    def test_request_rejects_mixed_structured_output_constraints(self):
        with self.assertRaises(ValidationError):
            TPUCompletionRequest.model_validate(
                {
                    "model": "model",
                    "prompt": "prompt",
                    "structured_outputs": {
                        "regex": ["foo+"],
                        "choice": ["bar"],
                    },
                }
            )

    def test_validate_stacked_regexes_rejects_empty_list(self):
        with self.assertRaises(VLLMValidationError):
            validate_stacked_regexes([])

    def test_validate_stacked_regexes_rejects_empty_stage(self):
        with self.assertRaises(VLLMValidationError):
            validate_stacked_regexes([r"foo+", ""])

    def test_to_sampling_params_keeps_single_regex_unchanged(self):
        request = TPUCompletionRequest.model_validate(
            {
                "model": "model",
                "prompt": "prompt",
                "structured_outputs": {"regex": r"foo+"},
            }
        )

        sampling_params = request.to_sampling_params(16, None, None)

        self.assertIsInstance(sampling_params, SamplingParams)
        self.assertIsNotNone(sampling_params.structured_outputs)
        self.assertEqual(sampling_params.structured_outputs.regex, r"foo+")
        self.assertIsNone(extract_stacked_regexes(sampling_params))

    def test_to_sampling_params_stashes_regex_list_in_extra_args(self):
        request = TPUCompletionRequest.model_validate(
            {
                "model": "model",
                "prompt": "prompt",
                "structured_outputs": {"regex": [r"foo+", r"bar?"]},
            }
        )

        sampling_params = request.to_sampling_params(16, None, None)

        self.assertIsNotNone(sampling_params.structured_outputs)
        self.assertEqual(sampling_params.structured_outputs.regex, r"foo+")
        self.assertEqual(extract_stacked_regexes(sampling_params), [r"foo+", r"bar?"])

    def test_normalize_regex_list_requires_guidance_backend(self):
        request = TPUCompletionRequest.model_validate(
            {
                "model": "model",
                "prompt": "prompt",
                "structured_outputs": {"regex": [r"foo+", r"bar?"]},
            }
        )

        with self.assertRaises(VLLMValidationError):
            normalize_completion_request(request, "outlines")


class TestStagedGuidanceGrammar(unittest.TestCase):

    def test_staged_guidance_switches_with_real_llguidance_matchers(self):
        class _RealStageGrammar:

            def __init__(self, ll_matcher):
                self.ll_matcher = ll_matcher

            def accept_tokens(self, _request_id: str, tokens: list[int]) -> bool:
                if self.ll_matcher.is_stopped():
                    return True
                return self.ll_matcher.consume_tokens(tokens)

            def validate_tokens(self, tokens: list[int]) -> list[int]:
                if not tokens or self.ll_matcher.is_stopped():
                    return []
                return tokens[:self.ll_matcher.validate_tokens(tokens)]

            def rollback(self, num_tokens: int) -> None:
                if num_tokens > 0:
                    self.ll_matcher.rollback(num_tokens)

            def fill_bitmask(self, bitmask, idx: int) -> None:
                bitmask[idx] = self.ll_matcher.compute_ff_bytes()

            def is_terminated(self) -> bool:
                return False

            def reset(self):
                self.ll_matcher.reset()

        class _RealGuidanceBackend:

            def __init__(self, ll_tokenizer) -> None:
                self.ll_tokenizer = ll_tokenizer

            def compile_grammar(self, request_type, grammar_spec):
                assert request_type == StructuredOutputOptions.REGEX
                grammar = llguidance.LLMatcher.grammar_from_regex(grammar_spec)
                ll_matcher = llguidance.LLMatcher(
                    self.ll_tokenizer,
                    grammar,
                    log_level=0,
                )
                return _RealStageGrammar(ll_matcher)

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        ll_tokenizer = llguidance_hf.from_tokenizer(tokenizer, len(tokenizer))
        backend = _RealGuidanceBackend(ll_tokenizer)
        grammar = StagedGuidanceGrammar(backend, ["foo ", "bar"])

        stage_one_tokens = tokenizer.encode("foo ", add_special_tokens=False)
        stage_two_tokens = tokenizer.encode("bar", add_special_tokens=False)

        self.assertTrue(grammar.accept_tokens("req", stage_one_tokens))
        self.assertEqual(grammar.current_stage_index, 0)

        bitmask = [None]
        grammar.fill_bitmask(bitmask, 0)
        self.assertEqual(grammar.current_stage_index, 1)
        self.assertEqual(bitmask[0], b"bar")

        self.assertTrue(grammar.accept_tokens("req", stage_two_tokens))

        bitmask = [None]
        grammar.fill_bitmask(bitmask, 0)
        self.assertEqual(bitmask[0], b"")

    def test_staged_guidance_switches_to_next_stage_without_recompiling_first(self):
        backend = _FakeGuidanceBackend({"r0": [1, 2], "r1": [3]})
        grammar = StagedGuidanceGrammar(backend, ["r0", "r1"])

        bitmask = [None]
        grammar.fill_bitmask(bitmask, 0)
        self.assertEqual(bitmask[0], 1)
        self.assertEqual(backend.compile_calls, ["r0"])

        self.assertTrue(grammar.accept_tokens("req", [1, 2]))
        self.assertEqual(grammar.current_stage_index, 0)

        bitmask = [None]
        grammar.fill_bitmask(bitmask, 0)
        self.assertEqual(bitmask[0], 3)
        self.assertEqual(grammar.current_stage_index, 1)
        self.assertEqual(backend.compile_calls, ["r0", "r1"])

    def test_staged_guidance_rolls_back_across_stage_boundary(self):
        backend = _FakeGuidanceBackend({"r0": [1], "r1": [2]})
        grammar = StagedGuidanceGrammar(backend, ["r0", "r1"])

        self.assertTrue(grammar.accept_tokens("req", [1, 2]))
        grammar.rollback(1)

        bitmask = [None]
        grammar.fill_bitmask(bitmask, 0)
        self.assertEqual(bitmask[0], 2)
        self.assertEqual(grammar.current_stage_index, 1)

        grammar.rollback(1)
        bitmask = [None]
        grammar.fill_bitmask(bitmask, 0)
        self.assertEqual(bitmask[0], 1)
        self.assertEqual(grammar.current_stage_index, 0)

    def test_staged_guidance_validate_tokens_crosses_stage_boundary(self):
        backend = _FakeGuidanceBackend({"r0": [1], "r1": [2, 3]})
        grammar = StagedGuidanceGrammar(backend, ["r0", "r1"])

        self.assertEqual(grammar.validate_tokens([1, 2, 3, 9]), [1, 2, 3])

        bitmask = [None]
        grammar.fill_bitmask(bitmask, 0)
        self.assertEqual(bitmask[0], 1)

    def test_install_patch_returns_staged_guidance_grammar(self):
        install_staged_guidance_patch()

        from vllm.v1.structured_output import StructuredOutputManager

        manager = StructuredOutputManager.__new__(StructuredOutputManager)
        manager.backend = _FakeGuidanceBackend({"r0": [1], "r1": [2]})

        request = SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args={"tpu_staged_regex": ["r0", "r1"]}),
            structured_output_request=SimpleNamespace(
                structured_output_key=(StructuredOutputOptions.REGEX, "r0")
            ),
        )

        grammar = StructuredOutputManager._create_grammar(manager, request)
        self.assertIsInstance(grammar, StagedGuidanceGrammar)


class TestStackedRegexRoute(unittest.IsolatedAsyncioTestCase):

    async def test_route_passes_tpu_request_with_staged_regex_metadata(self):
        request = TPUCompletionRequest.model_validate(
            {
                "model": "model",
                "prompt": "prompt",
                "structured_outputs": {"regex": [r"foo+", r"bar?"]},
            }
        )
        raw_request = _FakeRawRequest(backend="guidance")
        handler = _SamplingParamsCapturingHandler()

        original_handler = online_rl_server.create_completion.__wrapped__.__wrapped__
        original_completion = online_rl_server.completion
        try:
            online_rl_server.completion = lambda _request: handler
            response = await original_handler(request, raw_request)
        finally:
            online_rl_server.completion = original_completion

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(handler.requests), 1)
        self.assertEqual(
            extract_stacked_regexes(handler.sampling_params[0]),
            [r"foo+", r"bar?"],
        )

    async def test_route_returns_bad_request_for_invalid_backend(self):
        request = TPUCompletionRequest.model_validate(
            {
                "model": "model",
                "prompt": "prompt",
                "structured_outputs": {"regex": [r"foo+", r"bar?"]},
            }
        )
        raw_request = _FakeRawRequest(backend="outlines")
        handler = _SamplingParamsCapturingHandler()

        original_handler = online_rl_server.create_completion.__wrapped__.__wrapped__
        original_completion = online_rl_server.completion
        try:
            online_rl_server.completion = lambda _request: handler
            response = await original_handler(request, raw_request)
        finally:
            online_rl_server.completion = original_completion

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(handler.requests, [])
        self.assertIn("guidance backend", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
