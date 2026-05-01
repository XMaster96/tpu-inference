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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


_THOUGHT_PREFIX = "thought\n"


class TpuGemma4ReasoningParser(ReasoningParser):
    """Gemma 4 thinking parser for TPU vLLM structured-output gating.

    Gemma 4 thinking mode is enabled by the chat template with ``<|think|>``.
    The generated reasoning span is delimited as::

        <|channel>thought
        ...reasoning...<channel|>

    vLLM's structured-output manager uses ``is_reasoning_end*`` to keep regex
    or JSON guidance disabled during that reasoning span, then enables guidance
    for the final answer tokens after ``<channel|>``.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        required = [
            "<|channel>",
            "<channel|>",
            "<|turn>",
            "<|tool_call>",
            "<|tool_response>",
        ]
        missing = [token for token in required if token not in self.vocab]
        if missing:
            raise RuntimeError(
                "Gemma 4 reasoning parser could not locate tokenizer tokens: "
                f"{missing}")

        self.start_token_id = self.vocab["<|channel>"]
        self.end_token_id = self.vocab["<channel|>"]
        self.new_turn_token_id = self.vocab["<|turn>"]
        self.tool_call_token_id = self.vocab["<|tool_call>"]
        self.tool_response_token_id = self.vocab["<|tool_response>"]

        self._reasoning_text = ""
        self._prefix_stripped = False

    @property
    def reasoning_start_str(self) -> str:
        return "<|channel>"

    @property
    def reasoning_end_str(self) -> str:
        return "<channel|>"

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        # Keep delimiter tokens available for non-streaming extraction.
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Look at the current generated turn only. A previous turn delimiter
        # means no reasoning span has started for this answer yet.
        for token_id in reversed(input_ids):
            if token_id == self.start_token_id:
                return False
            if token_id == self.end_token_id:
                return True
            if token_id == self.tool_call_token_id:
                return True
            if token_id in (self.new_turn_token_id, self.tool_response_token_id):
                return False
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        # The structured-output manager calls this after a decode step. If the
        # end marker appeared in the new tokens, enable guidance on the next
        # step so the delimiter itself is not forced to match the final regex.
        return self.end_token_id in tuple(delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Use the last delimiter in case earlier history contains a completed
        # reasoning span.
        for index in range(len(input_ids) - 2, -1, -1):
            if input_ids[index] == self.end_token_id:
                return input_ids[index + 1:]
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        start = "<|channel>"
        end = "<channel|>"
        if start not in model_output and end not in model_output:
            return None, model_output

        _, has_start, after_start = model_output.partition(start)
        if not has_start:
            return None, model_output

        reasoning, has_end, content = after_start.partition(end)
        reasoning = _strip_thought_label(reasoning).strip() or None
        content = (content or None) if has_end else None
        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if len(delta_token_ids) == 1:
            token_id = delta_token_ids[0]
            if token_id in (self.start_token_id, self.end_token_id):
                return None

        if self.start_token_id in previous_token_ids:
            if self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text)
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find("<channel|>")
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len("<channel|>"):]
                return self._reasoning_delta(reasoning, content or None)
            return self._reasoning_delta(delta_text, None)

        if self.start_token_id in delta_token_ids:
            start_index = delta_text.find("<|channel>")
            after_start = delta_text[start_index + len("<|channel>"):]
            if self.end_token_id in delta_token_ids:
                reasoning, _, content = after_start.partition("<channel|>")
                return self._reasoning_delta(reasoning, content or None)
            return self._reasoning_delta(after_start, None)

        return DeltaMessage(content=delta_text)

    def _reasoning_delta(self, reasoning: str,
                         content: str | None) -> DeltaMessage | None:
        self._reasoning_text += reasoning

        if self._prefix_stripped:
            return DeltaMessage(reasoning=reasoning or None, content=content)

        if self._reasoning_text.startswith(_THOUGHT_PREFIX):
            previous_len = len(self._reasoning_text) - len(reasoning)
            prefix_len = len(_THOUGHT_PREFIX)
            strip_from_delta = max(prefix_len - previous_len, 0)
            stripped = reasoning[strip_from_delta:]
            if previous_len + len(reasoning) >= prefix_len:
                self._prefix_stripped = True
            return DeltaMessage(reasoning=stripped or None, content=content)

        if _THOUGHT_PREFIX.startswith(self._reasoning_text):
            return DeltaMessage(content=content) if content else None

        self._prefix_stripped = True
        return DeltaMessage(reasoning=self._reasoning_text or None,
                            content=content)


def _strip_thought_label(text: str) -> str:
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX):]
    return text


ReasoningParserManager.register_module(
    name="tpu_gemma4",
    module=TpuGemma4ReasoningParser,
)
