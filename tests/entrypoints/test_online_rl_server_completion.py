# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from http import HTTPStatus
from types import SimpleNamespace

from vllm.entrypoints.openai.completion.protocol import (
    CompletionResponse,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse, UsageInfo
from vllm.exceptions import VLLMValidationError

from tpu_inference.entrypoints import online_rl_server
from tpu_inference.entrypoints.stacked_regex import TPUCompletionRequest


class _FakeRawRequest:

    def __init__(self):
        self.headers = {}
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                args=SimpleNamespace(
                    structured_outputs_config=SimpleNamespace(backend="guidance")
                )
            )
        )


class _RetryingCompletionHandler:

    def __init__(self, max_model_len: int):
        self.max_model_len = max_model_len
        self.requests = []

    async def create_completion(self, request, _raw_request) -> CompletionResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise VLLMValidationError(
                "This model's maximum context length is 32 tokens. "
                "However, your request has 40 input tokens. "
                "Please reduce the length of the input messages.",
                parameter="input_tokens",
                value=40,
            )

        return CompletionResponse(
            model="model",
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text="ok",
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


class TestOnlineRLServerCompletion(unittest.IsolatedAsyncioTestCase):

    async def test_route_retries_context_limit_with_truncation(self):
        request = TPUCompletionRequest.model_validate(
            {
                "model": "model",
                "prompt": "prompt",
                "max_tokens": 8,
            }
        )
        raw_request = _FakeRawRequest()
        handler = _RetryingCompletionHandler(max_model_len=32)

        original_handler = online_rl_server.create_completion.__wrapped__.__wrapped__
        original_completion = online_rl_server.completion
        try:
            online_rl_server.completion = lambda _request: handler
            response = await original_handler(request, raw_request)
        finally:
            online_rl_server.completion = original_completion

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["choices"][0]["text"], "ok")
        self.assertEqual([req.truncate_prompt_tokens for req in handler.requests], [None, 24])


if __name__ == "__main__":
    unittest.main()
