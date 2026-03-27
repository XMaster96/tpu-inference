# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from vllm.v1.request import RequestStatus

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from tpu_inference.vllm_runtime_patches import (
    apply_vllm_runtime_patches,
    patch_async_scheduler_preempt_discard,
    patch_async_llm_request_admission_gate,
    run_with_request_admission_gate,
)


def test_tpu_worker_import_installs_vllm_runtime_patches():
    import tpu_inference.worker.tpu_worker  # noqa: F401

    assert getattr(Scheduler, "_tpu_reload_stale_output_patch_installed", False)


def test_scheduler_patch_drops_reload_stale_req_ids_without_keyerror():
    apply_vllm_runtime_patches()

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {
        "cmpl-b4b24bb5ab28f67c-0-a649d07b": 8,
    }
    scheduler_output.total_num_scheduled_tokens = 8
    scheduler_output.scheduled_spec_decode_tokens = {
        "cmpl-b4b24bb5ab28f67c-0-a649d07b": [11, 12, 13],
    }

    model_runner_output = SimpleNamespace(
        sampled_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={},
    )
    fake_scheduler = SimpleNamespace(
        perf_metrics=None,
        connector=None,
        kv_cache_manager=SimpleNamespace(take_events=lambda: None),
        kv_event_publisher=SimpleNamespace(publish=lambda batch: None),
        finished_req_ids_dict={},
        make_stats=lambda *args, **kwargs: None,
    )

    outputs = Scheduler.update_from_output(
        fake_scheduler,
        scheduler_output,
        model_runner_output,
    )
    assert outputs == {}


def test_async_scheduler_patch_clears_preempt_discard_for_waiting_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    original_update = AsyncScheduler._update_request_with_output
    original_installed = getattr(
        AsyncScheduler, "_tpu_preempt_discard_patch_installed", False)

    seen = {}

    def fake_original(self, request, new_token_ids):
        seen["discard_latest_async_tokens"] = request.discard_latest_async_tokens
        seen["status"] = request.status
        return list(new_token_ids), False

    try:
        monkeypatch.setattr(AsyncScheduler, "_update_request_with_output",
                            fake_original)
        AsyncScheduler._tpu_preempt_discard_patch_installed = False
        patch_async_scheduler_preempt_discard()

        request = SimpleNamespace(
            discard_latest_async_tokens=True,
            status=RequestStatus.WAITING,
        )
        new_token_ids, stopped = AsyncScheduler._update_request_with_output(
            SimpleNamespace(),
            request,
            [11, 12],
        )

        assert (new_token_ids, stopped) == ([11, 12], False)
        assert request.discard_latest_async_tokens is False
        assert seen == {
            "discard_latest_async_tokens": False,
            "status": RequestStatus.WAITING,
        }
    finally:
        AsyncScheduler._update_request_with_output = original_update
        AsyncScheduler._tpu_preempt_discard_patch_installed = original_installed


def test_run_with_request_admission_gate_waits_for_reload_gate():
    async def _run():
        gate = asyncio.Lock()
        engine = SimpleNamespace(_tpu_reload_request_gate=gate)
        seen: list[str] = []

        async def _op() -> str:
            seen.append("ran")
            return "ok"

        async with gate:
            task = asyncio.create_task(run_with_request_admission_gate(engine, _op))
            await asyncio.sleep(0)
            assert seen == []

        assert await task == "ok"
        assert seen == ["ran"]

    asyncio.run(_run())


def test_async_llm_request_admission_patch_uses_reload_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.engine.async_llm import AsyncLLM

    original_add_request = AsyncLLM.add_request
    original_installed = getattr(
        AsyncLLM, "_tpu_request_admission_gate_patch_installed", False)
    call_order: list[str] = []

    async def _run():
        async def fake_original(self, *args, **kwargs):
            call_order.append("original")
            return "ok"

        monkeypatch.setattr(AsyncLLM, "add_request", fake_original)
        AsyncLLM._tpu_request_admission_gate_patch_installed = False
        patch_async_llm_request_admission_gate()

        gate = asyncio.Lock()
        engine = SimpleNamespace(_tpu_reload_request_gate=gate)

        async with gate:
            task = asyncio.create_task(
                AsyncLLM.add_request(engine, "req-1", "prompt", None))
            await asyncio.sleep(0)
            assert call_order == []

        assert await task == "ok"
        assert call_order == ["original"]

    try:
        asyncio.run(_run())
    finally:
        AsyncLLM.add_request = original_add_request
        AsyncLLM._tpu_request_admission_gate_patch_installed = original_installed
