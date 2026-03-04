# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

from fastapi import HTTPException

from tpu_inference.entrypoints import online_rl_server


class _FakeEngineCore:

    def __init__(self):
        self.abort_calls: list[list[str]] = []
        self.add_calls: list[object] = []


class _FakeEngine:

    def __init__(
        self,
        *,
        initial_paused: bool = False,
        reset_prefix_cache_result: bool = True,
        collective_exc: Exception | None = None,
        pause_exc: Exception | None = None,
    ):
        self._pause_cond = asyncio.Condition()
        self._paused = initial_paused
        self.engine_core = _FakeEngineCore()

        self.pause_calls: list[dict] = []
        self.resume_calls = 0
        self.reset_prefix_cache_calls: list[dict] = []
        self.reset_mm_cache_calls = 0
        self.collective_calls: list[dict] = []

        self._reset_prefix_cache_result = reset_prefix_cache_result
        self._collective_exc = collective_exc
        self._pause_exc = pause_exc

    async def pause_generation(self, **kwargs):
        if self._pause_exc is not None:
            raise self._pause_exc
        self.pause_calls.append(kwargs)

    async def resume_generation(self):
        self.resume_calls += 1
        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()

    async def reset_prefix_cache(self, **kwargs):
        self.reset_prefix_cache_calls.append(kwargs)
        return self._reset_prefix_cache_result

    async def reset_mm_cache(self):
        self.reset_mm_cache_calls += 1

    async def collective_rpc(self, **kwargs):
        self.collective_calls.append(kwargs)
        if self._collective_exc is not None:
            raise self._collective_exc
        return [{"status": "ok"}]


class _EngineRejectingReplayInternals(_FakeEngine):

    @property
    def output_processor(self):
        raise AssertionError("Replay internals should not be touched in strict mode")


class _EngineMissingPauseInternals:

    def __init__(self):
        self.resume_calls = 0

    async def resume_generation(self):
        self.resume_calls += 1

    async def collective_rpc(self, **kwargs):
        return [{"status": "ok"}]


class _FakeRawRequest:

    def __init__(self, app_state):
        self.app = SimpleNamespace(state=app_state)


@dataclass
class _RegexTracker:
    regex: str
    accepted_token_ids: list[int] = field(default_factory=list)

    def accept_tokens(self, token_ids: list[int]) -> None:
        self.accepted_token_ids.extend(token_ids)


@dataclass
class _StressRequestState:
    request_id: str
    prompt_text: str
    prompt_token_ids: list[int]
    regex_tracker: _RegexTracker
    generated_token_ids: list[int] = field(default_factory=list)
    preemptions: int = 0
    num_computed_tokens: int = 0
    status: str = "running"


class _StressEngine(_FakeEngine):

    def __init__(self, rpc_delay_seconds: float = 0.0):
        super().__init__()
        self.running_requests: dict[str, _StressRequestState] = {}
        self.waiting_requests: dict[str, _StressRequestState] = {}
        self.rpc_delay_seconds = rpc_delay_seconds
        self.rpc_inflight = 0
        self.max_rpc_inflight = 0

    def add_request_state(self, req: _StressRequestState) -> None:
        self.running_requests[req.request_id] = req

    def generate_chunk(self, request_id: str, token_ids: list[int]) -> None:
        if self._paused:
            return
        req = self.running_requests.get(request_id)
        if req is None:
            req = self.waiting_requests.pop(request_id)
            req.status = "running"
            self.running_requests[request_id] = req
        req.generated_token_ids.extend(token_ids)
        req.regex_tracker.accept_tokens(token_ids)
        req.num_computed_tokens += len(token_ids)

    async def reset_prefix_cache(self, **kwargs):
        self.reset_prefix_cache_calls.append(kwargs)
        if not kwargs.get("reset_running_requests", False):
            return False

        for request_id in list(self.running_requests):
            req = self.running_requests.pop(request_id)
            req.preemptions += 1
            req.num_computed_tokens = 0
            req.status = "preempted"
            self.waiting_requests[request_id] = req
        return self._reset_prefix_cache_result

    async def collective_rpc(self, **kwargs):
        self.collective_calls.append(kwargs)
        self.rpc_inflight += 1
        self.max_rpc_inflight = max(self.max_rpc_inflight, self.rpc_inflight)
        try:
            if self.rpc_delay_seconds > 0:
                await asyncio.sleep(self.rpc_delay_seconds)
            return [{"status": "ok"}]
        finally:
            self.rpc_inflight -= 1


class TestReloadWeightsEndpoint(unittest.IsolatedAsyncioTestCase):

    @staticmethod
    def _build_request(engine):
        app_state = SimpleNamespace(
            engine_client=engine,
            reload_lock=asyncio.Lock(),
            current_checkpoint_path="old-ckpt",
            first_weights_loaded=asyncio.Event(),
            require_first_reload=False,
        )
        return app_state, _FakeRawRequest(app_state)

    async def test_preempt_mode_uses_reset_running_requests(self):
        engine = _FakeEngine()
        app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path="new-ckpt",
            wait_for_inflight_requests=False,
            clear_cache=True,
            release_kv_cache=True,
            timeout_seconds=30.0,
        )

        response = await online_rl_server.reload_weights(payload, raw_request)
        body = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["checkpoint_path"], "new-ckpt")
        self.assertEqual(engine.pause_calls, [])
        self.assertEqual(engine.resume_calls, 0)
        self.assertEqual(
            engine.reset_prefix_cache_calls,
            [
                {"reset_running_requests": True, "reset_connector": False},
                {"reset_running_requests": True, "reset_connector": False},
            ],
        )
        self.assertEqual(engine.reset_mm_cache_calls, 1)
        self.assertFalse(engine._paused)
        self.assertEqual(engine.engine_core.abort_calls, [])
        self.assertEqual(engine.engine_core.add_calls, [])
        self.assertEqual(len(engine.collective_calls), 1)
        self.assertEqual(
            engine.collective_calls[0]["kwargs"],
            {"release_kv_cache": True},
        )
        self.assertEqual(engine.collective_calls[0]["args"], ("new-ckpt",))
        self.assertTrue(app_state.first_weights_loaded.is_set())
        self.assertEqual(app_state.current_checkpoint_path, "new-ckpt")

    async def test_preempt_mode_does_not_touch_replay_internals(self):
        engine = _EngineRejectingReplayInternals()
        _app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path="new-ckpt",
            wait_for_inflight_requests=False,
            clear_cache=True,
            release_kv_cache=True,
            timeout_seconds=30.0,
        )

        response = await online_rl_server.reload_weights(payload, raw_request)
        self.assertEqual(response.status_code, 200)

    async def test_preempt_mode_without_checkpoint_uses_current_path(self):
        engine = _FakeEngine()
        app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path=None,
            wait_for_inflight_requests=False,
            clear_cache=True,
            release_kv_cache=False,
            timeout_seconds=30.0,
        )

        response = await online_rl_server.reload_weights(payload, raw_request)
        body = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["checkpoint_path"], "old-ckpt")
        self.assertEqual(engine.collective_calls[0]["args"], ())
        self.assertEqual(
            engine.collective_calls[0]["kwargs"],
            {"release_kv_cache": False},
        )
        self.assertEqual(app_state.current_checkpoint_path, "old-ckpt")

    async def test_preempt_mode_clear_cache_false_still_preempts(self):
        engine = _FakeEngine()
        app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path="new-ckpt",
            wait_for_inflight_requests=False,
            clear_cache=False,
            release_kv_cache=True,
            timeout_seconds=30.0,
        )

        response = await online_rl_server.reload_weights(payload, raw_request)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            engine.reset_prefix_cache_calls,
            [
                {"reset_running_requests": True, "reset_connector": False},
                {"reset_running_requests": True, "reset_connector": False},
            ],
        )
        self.assertEqual(engine.reset_mm_cache_calls, 0)
        self.assertEqual(app_state.current_checkpoint_path, "new-ckpt")

    async def test_preempt_mode_respects_existing_pause_state(self):
        engine = _FakeEngine(initial_paused=True)
        _app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path="new-ckpt",
            wait_for_inflight_requests=False,
            clear_cache=True,
            release_kv_cache=True,
            timeout_seconds=30.0,
        )

        response = await online_rl_server.reload_weights(payload, raw_request)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(engine._paused)
        self.assertEqual(engine.resume_calls, 0)

    async def test_preempt_mode_collective_failure_unpauses_and_errors(self):
        engine = _FakeEngine(collective_exc=RuntimeError("rpc failed"))
        app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path="new-ckpt",
            wait_for_inflight_requests=False,
            clear_cache=True,
            release_kv_cache=True,
            timeout_seconds=30.0,
        )

        with self.assertRaises(HTTPException) as cm:
            await online_rl_server.reload_weights(payload, raw_request)

        self.assertEqual(cm.exception.status_code, 500)
        self.assertIn("rpc failed", cm.exception.detail)
        self.assertFalse(engine._paused)
        self.assertEqual(app_state.current_checkpoint_path, "old-ckpt")
        self.assertFalse(app_state.first_weights_loaded.is_set())

    async def test_preempt_mode_prefix_reset_failure_unpauses_and_errors(self):
        engine = _FakeEngine(reset_prefix_cache_result=False)
        app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path="new-ckpt",
            wait_for_inflight_requests=False,
            clear_cache=True,
            release_kv_cache=True,
            timeout_seconds=30.0,
        )

        with self.assertRaises(HTTPException) as cm:
            await online_rl_server.reload_weights(payload, raw_request)

        self.assertEqual(cm.exception.status_code, 500)
        self.assertIn("reset_prefix_cache returned failure", cm.exception.detail)
        self.assertFalse(engine._paused)
        self.assertEqual(engine.collective_calls, [])
        self.assertEqual(engine.reset_mm_cache_calls, 0)
        self.assertEqual(app_state.current_checkpoint_path, "old-ckpt")
        self.assertFalse(app_state.first_weights_loaded.is_set())

    async def test_wait_mode_uses_pause_and_resume(self):
        engine = _FakeEngine()
        app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path="new-ckpt",
            wait_for_inflight_requests=True,
            clear_cache=False,
            release_kv_cache=True,
            timeout_seconds=30.0,
        )

        response = await online_rl_server.reload_weights(payload, raw_request)
        body = json.loads(response.body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(
            engine.pause_calls,
            [{"wait_for_inflight_requests": True, "clear_cache": False}],
        )
        self.assertEqual(engine.resume_calls, 1)
        self.assertEqual(engine.reset_prefix_cache_calls, [])
        self.assertEqual(engine.reset_mm_cache_calls, 0)
        self.assertFalse(engine._paused)
        self.assertTrue(app_state.first_weights_loaded.is_set())

    async def test_wait_mode_pause_failure_does_not_force_resume(self):
        engine = _FakeEngine(pause_exc=RuntimeError("pause failed"))
        _app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path="new-ckpt",
            wait_for_inflight_requests=True,
            clear_cache=False,
            release_kv_cache=True,
            timeout_seconds=30.0,
        )

        with self.assertRaises(HTTPException) as cm:
            await online_rl_server.reload_weights(payload, raw_request)

        self.assertEqual(cm.exception.status_code, 500)
        self.assertIn("pause failed", cm.exception.detail)
        self.assertEqual(engine.resume_calls, 0)
        self.assertEqual(engine.collective_calls, [])

    async def test_missing_pause_internals_returns_500(self):
        engine = _EngineMissingPauseInternals()
        _app_state, raw_request = self._build_request(engine)
        payload = online_rl_server.ReloadWeightsRequest(
            checkpoint_path="new-ckpt",
            wait_for_inflight_requests=False,
            clear_cache=True,
            release_kv_cache=True,
            timeout_seconds=30.0,
        )

        with self.assertRaises(HTTPException) as cm:
            await online_rl_server.reload_weights(payload, raw_request)

        self.assertEqual(cm.exception.status_code, 500)
        self.assertIn("missing", cm.exception.detail)
        self.assertEqual(engine.resume_calls, 0)


class TestPauseHelpers(unittest.IsolatedAsyncioTestCase):

    async def test_set_generation_paused_roundtrip(self):
        engine = _FakeEngine(initial_paused=False)
        was_paused = await online_rl_server._set_generation_paused(engine, True)
        self.assertFalse(was_paused)
        self.assertTrue(engine._paused)

        was_paused = await online_rl_server._set_generation_paused(engine, False)
        self.assertTrue(was_paused)
        self.assertFalse(engine._paused)

    def test_ensure_pause_controls_supported(self):
        online_rl_server._ensure_pause_controls_supported(_FakeEngine())

        with self.assertRaises(RuntimeError):
            online_rl_server._ensure_pause_controls_supported(
                _EngineMissingPauseInternals())

    def test_patch_async_scheduler_preempt_discard_idempotent(self):
        online_rl_server._patch_async_scheduler_preempt_discard()
        online_rl_server._patch_async_scheduler_preempt_discard()


class TestReloadWeightsStress(unittest.IsolatedAsyncioTestCase):

    @staticmethod
    def _build_request(engine):
        app_state = SimpleNamespace(
            engine_client=engine,
            reload_lock=asyncio.Lock(),
            current_checkpoint_path="old-ckpt",
            first_weights_loaded=asyncio.Event(),
            require_first_reload=False,
        )
        return app_state, _FakeRawRequest(app_state)

    @staticmethod
    def _make_complex_regex() -> str:
        head = r"^(?:(?:[A-Z]{2}\d{3}|[a-z]{4,9}|[_-]{1,3})\s){64}"
        middle = r"(?:(?:foo|bar|baz|qux)\d{1,2}|[A-F0-9]{6}|(?:xy){2,5}){32}"
        tail = r"(?:\.(?:json|yaml|txt)){1,4}(?:(?!forbidden).)*(?:END|STOP)$"
        return head + middle + tail

    @staticmethod
    def _make_long_prompt() -> tuple[str, list[int]]:
        prompt_chunks = [f"segment_{i:05d}=alpha123" for i in range(12000)]
        prompt_text = " | ".join(prompt_chunks)
        prompt_token_ids = list(range(32768))
        return prompt_text, prompt_token_ids

    async def test_long_prompt_complex_regex_and_many_reloads(self):
        engine = _StressEngine()
        app_state, raw_request = self._build_request(engine)
        regex = self._make_complex_regex()
        prompt_text, prompt_token_ids = self._make_long_prompt()

        req = _StressRequestState(
            request_id="req-long",
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            regex_tracker=_RegexTracker(regex=regex),
        )
        engine.add_request_state(req)

        engine.generate_chunk("req-long", list(range(0, 128)))
        tracker_obj_id = id(req.regex_tracker)

        for i in range(8):
            payload = online_rl_server.ReloadWeightsRequest(
                checkpoint_path=f"ckpt-{i}",
                wait_for_inflight_requests=False,
                clear_cache=(i % 2 == 0),
                release_kv_cache=True,
                timeout_seconds=30.0,
            )
            response = await online_rl_server.reload_weights(payload, raw_request)
            self.assertEqual(response.status_code, 200)
            self.assertIn("req-long", engine.waiting_requests)

            next_chunk_start = 128 + i * 256
            engine.generate_chunk(
                "req-long", list(range(next_chunk_start, next_chunk_start + 256)))
            self.assertIn("req-long", engine.running_requests)
            self.assertEqual(id(req.regex_tracker), tracker_obj_id)

        self.assertEqual(req.preemptions, 8)
        self.assertEqual(len(req.prompt_token_ids), 32768)
        self.assertGreater(len(req.prompt_text), 200000)
        self.assertEqual(req.regex_tracker.regex, regex)
        self.assertEqual(len(req.regex_tracker.accepted_token_ids), 128 + 8 * 256)
        self.assertEqual(len(engine.reset_prefix_cache_calls), 16)
        self.assertFalse(engine._paused)
        self.assertEqual(app_state.current_checkpoint_path, "ckpt-7")
        self.assertTrue(app_state.first_weights_loaded.is_set())

    async def test_concurrent_reload_storm_serializes_and_does_not_break_state(self):
        engine = _StressEngine(rpc_delay_seconds=0.03)
        app_state, raw_request = self._build_request(engine)
        regex = self._make_complex_regex()
        prompt_text, prompt_token_ids = self._make_long_prompt()
        req = _StressRequestState(
            request_id="req-storm",
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            regex_tracker=_RegexTracker(regex=regex),
        )
        engine.add_request_state(req)
        tracker_obj_id = id(req.regex_tracker)

        stop_generation = asyncio.Event()

        async def generation_loop():
            token = 100000
            while not stop_generation.is_set():
                engine.generate_chunk("req-storm", [token])
                token += 1
                await asyncio.sleep(0.001)

        async def trigger_reload(idx: int):
            payload = online_rl_server.ReloadWeightsRequest(
                checkpoint_path=f"storm-ckpt-{idx}",
                wait_for_inflight_requests=False,
                clear_cache=(idx % 2 == 0),
                release_kv_cache=True,
                timeout_seconds=30.0,
            )
            response = await online_rl_server.reload_weights(payload, raw_request)
            self.assertEqual(response.status_code, 200)

        generation_task = asyncio.create_task(generation_loop())
        try:
            await asyncio.gather(*[trigger_reload(i) for i in range(12)])
        finally:
            stop_generation.set()
            await generation_task

        self.assertEqual(len(engine.reset_prefix_cache_calls), 24)
        self.assertEqual(len(engine.collective_calls), 12)
        self.assertEqual(engine.max_rpc_inflight, 1)
        self.assertEqual(id(req.regex_tracker), tracker_obj_id)
        self.assertGreater(len(req.regex_tracker.accepted_token_ids), 0)
        self.assertFalse(engine._paused)
        self.assertTrue(app_state.first_weights_loaded.is_set())


if __name__ == "__main__":
    unittest.main()
