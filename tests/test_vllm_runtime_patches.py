# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.request import RequestStatus
from vllm.v1.request import Request

from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from tpu_inference.vllm_runtime_patches import (
    apply_vllm_runtime_patches,
    filter_scheduler_output_missing_req_indices,
    patch_async_scheduler_preempt_discard,
    patch_async_llm_request_admission_gate,
    requeue_missing_live_reload_requests,
    run_with_request_admission_gate,
)

pytestmark = pytest.mark.online_rl_server_related


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


def test_filter_scheduler_output_missing_req_indices_filters_cached_state():
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {
        "req-0": 1,
        "req-1": 2,
    }
    scheduler_output.total_num_scheduled_tokens = 3
    scheduler_output.scheduled_new_reqs = [
        SimpleNamespace(req_id="req-0"),
        SimpleNamespace(req_id="req-1"),
    ]
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=["req-0", "req-1"],
        resumed_req_ids={"req-1"},
        new_token_ids=[[11], [22]],
        all_token_ids={"req-0": [1, 2], "req-1": [3, 4]},
        new_block_ids=[None, ([7],)],
        num_computed_tokens=[2, 4],
        num_output_tokens=[0, 1],
    )
    scheduler_output.scheduled_spec_decode_tokens = {
        "req-1": [99],
    }
    scheduler_output.scheduled_encoder_inputs = {
        "req-0": [0],
        "req-1": [1],
    }
    scheduler_output.assigned_dp_rank = {
        "req-0": 0,
        "req-1": 1,
    }
    scheduler_output.num_invalid_spec_tokens = {
        "req-0": 0,
        "req-1": 1,
    }

    filtered_output, missing_req_ids = filter_scheduler_output_missing_req_indices(
        scheduler_output,
        SimpleNamespace(req_id_to_index={"req-0": 0}),
    )

    assert missing_req_ids == ("req-1",)
    assert filtered_output.total_num_scheduled_tokens == 1
    assert filtered_output.num_scheduled_tokens == {"req-0": 1}
    assert [req.req_id for req in filtered_output.scheduled_new_reqs] == ["req-0"]
    assert filtered_output.scheduled_cached_reqs.req_ids == ["req-0"]
    assert filtered_output.scheduled_cached_reqs.resumed_req_ids == set()
    assert filtered_output.scheduled_cached_reqs.new_token_ids == [[11]]
    assert filtered_output.scheduled_cached_reqs.all_token_ids == {
        "req-0": [1, 2],
    }
    assert filtered_output.scheduled_cached_reqs.num_output_tokens == [0]
    assert filtered_output.scheduled_spec_decode_tokens == {}
    assert filtered_output.scheduled_encoder_inputs == {"req-0": [0]}
    assert filtered_output.assigned_dp_rank == {"req-0": 0}
    assert filtered_output.num_invalid_spec_tokens == {"req-0": 0}


def test_requeue_missing_live_reload_requests_preempts_running_request():
    class _WaitingQueue:

        def __init__(self):
            self.prepended: list[object] = []

        def prepend_request(self, request):
            self.prepended.append(request)

        def __iter__(self):
            return iter(self.prepended)

    waiting = _WaitingQueue()
    request0 = SimpleNamespace(
        request_id="req-0",
        status=RequestStatus.RUNNING,
        num_computed_tokens=4,
        num_output_placeholders=1,
        spec_token_ids=[1],
        num_preemptions=0,
        discard_latest_async_tokens=True,
    )
    request1 = SimpleNamespace(
        request_id="req-1",
        status=RequestStatus.RUNNING,
        num_computed_tokens=7,
        num_output_placeholders=3,
        spec_token_ids=[2, 3],
        num_preemptions=0,
        discard_latest_async_tokens=True,
    )

    def _preempt_request(request, _timestamp):
        assert request.status == RequestStatus.RUNNING
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.spec_token_ids.clear()
        request.num_preemptions += 1
        waiting.prepend_request(request)

    scheduler = SimpleNamespace(
        requests={
            "req-0": request0,
            "req-1": request1,
        },
        running=[request0, request1],
        waiting=waiting,
        prev_step_scheduled_req_ids={"req-0", "req-1"},
        _preempt_request=_preempt_request,
    )

    requeue_missing_live_reload_requests(scheduler, ("req-1",))

    assert scheduler.running == [request0]
    assert waiting.prepended == [request1]
    assert scheduler.prev_step_scheduled_req_ids == {"req-0"}
    assert request1.status == RequestStatus.PREEMPTED
    assert request1.num_computed_tokens == 0
    assert request1.num_output_placeholders == 0
    assert request1.spec_token_ids == []
    assert request1.num_preemptions == 1
    assert request1.discard_latest_async_tokens is True


def test_requeued_live_reload_request_discards_late_async_token():
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    patch_async_scheduler_preempt_discard()

    class _WaitingQueue:

        def __init__(self):
            self.prepended: list[Request] = []

        def prepend_request(self, request):
            self.prepended.append(request)

    waiting = _WaitingQueue()
    request = Request(
        request_id="req-0",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        eos_token_id=0,
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 4
    request.num_output_placeholders = 1
    request.spec_token_ids = [7]
    request.discard_latest_async_tokens = True

    def _preempt_request(request, _timestamp):
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.spec_token_ids.clear()
        request.num_preemptions += 1
        waiting.prepend_request(request)

    scheduler = SimpleNamespace(
        requests={"req-0": request},
        running=[request],
        waiting=waiting,
        prev_step_scheduled_req_ids={"req-0"},
        _preempt_request=_preempt_request,
    )
    fake_async_scheduler = object.__new__(AsyncScheduler)
    fake_async_scheduler.max_model_len = 128
    fake_async_scheduler.kv_cache_manager = SimpleNamespace(
        cache_blocks=lambda *args: None)

    requeue_missing_live_reload_requests(scheduler, ("req-0",))
    new_token_ids, stopped = AsyncScheduler._update_request_with_output(
        fake_async_scheduler,
        request,
        [42],
    )

    assert (new_token_ids, stopped) == ([], False)
    assert list(request.output_token_ids) == []
    assert request.num_output_placeholders == 0
    assert request.discard_latest_async_tokens is False


def test_requeue_missing_live_reload_requests_is_idempotent_for_waiting_request():
    class _WaitingQueue:

        def __init__(self):
            self.prepended: list[object] = []

        def prepend_request(self, request):
            self.prepended.append(request)

        def __iter__(self):
            return iter(self.prepended)

    waiting = _WaitingQueue()
    request = SimpleNamespace(
        request_id="req-0",
        status=RequestStatus.PREEMPTED,
        num_computed_tokens=0,
        num_output_placeholders=0,
        spec_token_ids=[],
        num_preemptions=1,
        discard_latest_async_tokens=True,
    )
    waiting.prepend_request(request)

    scheduler = SimpleNamespace(
        requests={"req-0": request},
        running=[],
        waiting=waiting,
        prev_step_scheduled_req_ids=set(),
        _preempt_request=lambda *_args: None,
    )

    requeue_missing_live_reload_requests(scheduler, ("req-0",))

    assert waiting.prepended == [request]
    assert request.num_preemptions == 1
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_output_placeholders == 0


def test_requeue_missing_live_reload_requests_restores_orphaned_preempted_request():
    class _WaitingQueue:

        def __init__(self):
            self.prepended: list[object] = []

        def prepend_request(self, request):
            self.prepended.append(request)

    waiting = _WaitingQueue()
    request = SimpleNamespace(
        request_id="req-0",
        status=RequestStatus.PREEMPTED,
        num_computed_tokens=0,
        num_output_placeholders=3,
        spec_token_ids=[],
        num_preemptions=1,
        discard_latest_async_tokens=False,
    )

    scheduler = SimpleNamespace(
        requests={"req-0": request},
        running=[],
        waiting=waiting,
        prev_step_scheduled_req_ids={"req-0"},
        _preempt_request=lambda *_args: None,
    )

    requeue_missing_live_reload_requests(scheduler, ("req-0",))

    assert waiting.prepended == [request]
    assert scheduler.prev_step_scheduled_req_ids == set()
    assert request.num_preemptions == 1
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_output_placeholders == 0
    assert request.discard_latest_async_tokens is True


def test_async_scheduler_patch_discards_stale_token_without_placeholders():
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    patch_async_scheduler_preempt_discard()

    request = Request(
        request_id="req-0",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        eos_token_id=0,
    )
    request.status = RequestStatus.PREEMPTED
    request.num_output_placeholders = 0
    request.discard_latest_async_tokens = True

    fake_async_scheduler = object.__new__(AsyncScheduler)
    fake_async_scheduler.max_model_len = 128
    fake_async_scheduler.kv_cache_manager = SimpleNamespace(
        cache_blocks=lambda *args: None)

    new_token_ids, stopped = AsyncScheduler._update_request_with_output(
        fake_async_scheduler,
        request,
        [42],
    )

    assert (new_token_ids, stopped) == ([], False)
    assert list(request.output_token_ids) == []
    assert request.num_output_placeholders == 0
    assert request.discard_latest_async_tokens is False


def test_async_scheduler_patch_clears_preempt_discard_for_waiting_requests(
):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    patch_async_scheduler_preempt_discard()

    request = Request(
        request_id="req-0",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        eos_token_id=0,
    )
    request.status = RequestStatus.WAITING
    request.num_output_placeholders = 2
    request.discard_latest_async_tokens = True

    fake_async_scheduler = object.__new__(AsyncScheduler)
    fake_async_scheduler.max_model_len = 128
    fake_async_scheduler.kv_cache_manager = SimpleNamespace(
        cache_blocks=lambda *args: None)

    new_token_ids, stopped = AsyncScheduler._update_request_with_output(
        fake_async_scheduler,
        request,
        [11, 12],
    )

    assert (new_token_ids, stopped) == ([11, 12], False)
    assert list(request.output_token_ids) == [11, 12]
    assert request.num_output_placeholders == 0
    assert request.discard_latest_async_tokens is False


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
