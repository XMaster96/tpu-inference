"""TPU-local helpers for staged guidance regex requests."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Sequence, cast

from pydantic import Field
from pydantic.dataclasses import dataclass

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

_STRUCTURED_OUTPUT_PRIVATE_FIELDS = {"_backend", "_backend_was_auto"}
_STACKED_REGEX_EXTRA_ARG = "tpu_staged_regex"


def validate_stacked_regexes(regexes: Sequence[str]) -> list[str]:
    if not regexes:
        raise VLLMValidationError(
            "structured_outputs.regex list must contain at least one regex.",
            parameter="structured_outputs",
        )

    import llguidance

    normalized: list[str] = []
    for regex in regexes:
        if not isinstance(regex, str) or not regex:
            raise VLLMValidationError(
                "structured_outputs.regex list entries must be non-empty strings.",
                parameter="structured_outputs",
                value=regex,
            )

        guidance_grammar = llguidance.LLMatcher.grammar_from_regex(regex)
        grammar_error = llguidance.LLMatcher.validate_grammar(guidance_grammar)
        if grammar_error:
            raise VLLMValidationError(
                f"Invalid staged regex: {grammar_error}",
                parameter="structured_outputs",
                value=regex,
            )
        normalized.append(regex)

    return normalized


def extract_stacked_regexes(sampling_params: SamplingParams | None) -> list[str] | None:
    if sampling_params is None or sampling_params.extra_args is None:
        return None

    regexes = sampling_params.extra_args.get(_STACKED_REGEX_EXTRA_ARG)
    if regexes is None:
        return None
    if not isinstance(regexes, list) or not all(isinstance(item, str) for item in regexes):
        raise ValueError(
            f"{_STACKED_REGEX_EXTRA_ARG} must be a list[str], got {regexes!r}"
        )
    return cast(list[str], regexes)


@dataclass
class TPUStructuredOutputsParams:
    """TPU-local request type that widens ``regex`` to ``str | list[str]``."""

    json: str | dict[str, Any] | None = None
    regex: str | list[str] | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None

    _backend: str | None = field(default=None, init=False)
    _backend_was_auto: bool = field(default=False, init=False)

    def __post_init__(self):
        count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
                self.structural_tag is not None,
            ]
        )
        if count > 1:
            raise ValueError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )
        if count < 1:
            raise ValueError(
                "You must use one kind of structured outputs constraint "
                f"but none are specified: {self.__dict__}"
            )


class TPUCompletionRequest(CompletionRequest):
    structured_outputs: TPUStructuredOutputsParams | None = Field(
        default=None,
        description="Additional kwargs for structured outputs",
    )

    def to_sampling_params(
        self,
        max_tokens: int,
        logits_processor_pattern: str | None,
        default_sampling_params: dict[str, Any] | None = None,
    ) -> SamplingParams:
        regexes = self.stacked_regexes()
        if regexes is None:
            return super().to_sampling_params(
                max_tokens,
                logits_processor_pattern,
                default_sampling_params,
            )

        payload = _completion_request_payload(self)
        structured_outputs = cast(dict[str, Any], payload["structured_outputs"])
        structured_outputs["regex"] = regexes[0]

        standard_request = CompletionRequest.model_validate(payload)
        sampling_params = standard_request.to_sampling_params(
            max_tokens,
            logits_processor_pattern,
            default_sampling_params,
        )
        extra_args = dict(sampling_params.extra_args or {})
        extra_args[_STACKED_REGEX_EXTRA_ARG] = regexes
        sampling_params.extra_args = extra_args
        return sampling_params

    def stacked_regexes(self) -> list[str] | None:
        if self.structured_outputs is None or not isinstance(self.structured_outputs.regex, list):
            return None
        return validate_stacked_regexes(self.structured_outputs.regex)


class StagedGuidanceGrammar(StructuredOutputGrammar):
    """Runs one guidance regex matcher at a time while preserving request state."""

    def __init__(self, backend: Any, regexes: Sequence[str]) -> None:
        self._backend = backend
        self._regexes = list(validate_stacked_regexes(regexes))
        self._grammars: dict[int, StructuredOutputGrammar] = {}
        self._current_stage_idx = 0
        self._stage_token_counts = [0] * len(self._regexes)
        self._token_stage_history: list[int] = []

    @property
    def current_stage_index(self) -> int:
        return self._current_stage_idx

    def _get_stage_grammar(self, idx: int) -> StructuredOutputGrammar:
        grammar = self._grammars.get(idx)
        if grammar is None:
            grammar = self._backend.compile_grammar(
                StructuredOutputOptions.REGEX,
                self._regexes[idx],
            )
            self._grammars[idx] = grammar
        return grammar

    def _current_grammar(self) -> StructuredOutputGrammar:
        return self._get_stage_grammar(self._current_stage_idx)

    def _is_stage_accepting(self, idx: int) -> bool:
        grammar = self._get_stage_grammar(idx)
        ll_matcher = getattr(grammar, "ll_matcher", None)
        return bool(ll_matcher is not None and ll_matcher.is_accepting())

    def _advance_if_complete(self) -> None:
        while (
            self._current_stage_idx < len(self._regexes) - 1
            and self._is_stage_accepting(self._current_stage_idx)
        ):
            self._current_stage_idx += 1

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        for token in tokens:
            self._advance_if_complete()
            grammar = self._current_grammar()
            accepted = grammar.accept_tokens(request_id, [token])
            if not accepted:
                return False
            self._stage_token_counts[self._current_stage_idx] += 1
            self._token_stage_history.append(self._current_stage_idx)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        accepted = 0
        try:
            for token in tokens:
                if not self.accept_tokens("__validate__", [token]):
                    break
                accepted += 1
        finally:
            if accepted:
                self.rollback(accepted)
        return tokens[:accepted]

    def rollback(self, num_tokens: int) -> None:
        for _ in range(min(num_tokens, len(self._token_stage_history))):
            stage_idx = self._token_stage_history.pop()
            self._get_stage_grammar(stage_idx).rollback(1)
            self._stage_token_counts[stage_idx] -= 1

        self._current_stage_idx = 0
        for idx in range(len(self._stage_token_counts) - 1, -1, -1):
            if self._stage_token_counts[idx] > 0:
                self._current_stage_idx = idx
                break

    def fill_bitmask(self, bitmask, batch_index: int) -> None:
        self._advance_if_complete()
        self._current_grammar().fill_bitmask(bitmask, batch_index)

    def is_terminated(self) -> bool:
        self._advance_if_complete()
        return self._current_stage_idx == len(self._regexes) - 1 and self._current_grammar().is_terminated()

    def reset(self):
        for grammar in self._grammars.values():
            grammar.reset()
        self._current_stage_idx = 0
        self._stage_token_counts = [0] * len(self._regexes)
        self._token_stage_history.clear()


def normalize_completion_request(
    request: TPUCompletionRequest,
    backend: str | None,
) -> CompletionRequest:
    payload = _completion_request_payload(request)
    structured_outputs = payload.get("structured_outputs")
    if not isinstance(structured_outputs, dict):
        return request

    regex_value = structured_outputs.get("regex")
    if isinstance(regex_value, str) or regex_value is None:
        return request

    if not isinstance(regex_value, list):
        raise VLLMValidationError(
            "structured_outputs.regex must be either a string or a list of strings.",
            parameter="structured_outputs",
            value=regex_value,
        )

    if backend != "guidance":
        raise VLLMValidationError(
            "List-valued structured_outputs.regex requires the guidance backend.",
            parameter="structured_outputs",
            value=backend,
        )

    validate_stacked_regexes(regex_value)
    return request


def install_staged_guidance_patch() -> None:
    from vllm.v1.structured_output import StructuredOutputManager

    if getattr(StructuredOutputManager, "_tpu_staged_guidance_patch_installed", False):
        return

    original_create_grammar = StructuredOutputManager._create_grammar

    def _patched_create_grammar(self, request):
        regexes = extract_stacked_regexes(getattr(request, "sampling_params", None))
        if not regexes:
            return original_create_grammar(self, request)

        if self.backend is None or not hasattr(self.backend, "compile_grammar"):
            raise ValueError("TPU staged regex requires the guidance backend.")

        request_type, _grammar_spec = request.structured_output_request.structured_output_key
        if request_type != StructuredOutputOptions.REGEX:
            raise ValueError("TPU staged regex requests must use regex structured outputs.")

        return StagedGuidanceGrammar(self.backend, regexes)

    StructuredOutputManager._create_grammar = _patched_create_grammar
    StructuredOutputManager._tpu_staged_guidance_patch_installed = True


def _completion_request_payload(request: TPUCompletionRequest) -> dict[str, Any]:
    exclude: dict[str, Any] = {}
    if request.structured_outputs is not None:
        exclude["structured_outputs"] = _STRUCTURED_OUTPUT_PRIVATE_FIELDS
    return request.model_dump(exclude=exclude)
