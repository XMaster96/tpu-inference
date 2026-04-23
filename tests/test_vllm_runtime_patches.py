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

import tpu_inference.vllm_runtime_patches as runtime_patches
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


def test_scheduler_missing_request_indices_warning_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
):
    apply_vllm_runtime_patches()

    class _WaitingQueue:

        def __init__(self):
            self.prepended: list[object] = []

        def prepend_request(self, request):
            if request not in self.prepended:
                self.prepended.append(request)

        def __iter__(self):
            return iter(self.prepended)

    def _scheduler_output() -> SchedulerOutput:
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.num_scheduled_tokens = {"req-spam": 1}
        scheduler_output.total_num_scheduled_tokens = 1
        return scheduler_output

    waiting = _WaitingQueue()
    request = SimpleNamespace(
        request_id="req-spam",
        status=RequestStatus.RUNNING,
        num_computed_tokens=12,
        num_output_placeholders=1,
        spec_token_ids=[],
        num_preemptions=0,
        discard_latest_async_tokens=False,
    )

    def _preempt_request(request, _timestamp):
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        request.discard_latest_async_tokens = True
        waiting.prepend_request(request)

    fake_scheduler = SimpleNamespace(
        perf_metrics=None,
        connector=None,
        kv_cache_manager=SimpleNamespace(take_events=lambda: None),
        kv_event_publisher=SimpleNamespace(publish=lambda batch: None),
        finished_req_ids_dict={},
        make_stats=lambda *args, **kwargs: None,
        requests={"req-spam": request},
        running=[request],
        waiting=waiting,
        prev_step_scheduled_req_ids={"req-spam"},
        _preempt_request=_preempt_request,
        _tpu_reload_generation=3,
    )
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
    warning_messages: list[str] = []

    def _capture_warning(message, *args, **kwargs):
        del kwargs
        warning_messages.append(message % args if args else message)

    monkeypatch.setattr(runtime_patches.logger, "warning", _capture_warning)
    for _ in range(3):
        Scheduler.update_from_output(
            fake_scheduler,
            _scheduler_output(),
            model_runner_output,
        )

    stale_drop_warnings = [
        message for message in warning_messages
        if "Dropping stale model output entries missing request indices"
        in message
    ]
    assert len(stale_drop_warnings) == 1
    assert request.status == RequestStatus.PREEMPTED
    assert waiting.prepended == [request]


def test_scheduler_logs_fully_filtered_live_reload_recovery(
    monkeypatch: pytest.MonkeyPatch,
):
    apply_vllm_runtime_patches()

    class _WaitingQueue:

        def __init__(self):
            self.prepended: list[object] = []

        def prepend_request(self, request):
            if request not in self.prepended:
                self.prepended.append(request)

        def __iter__(self):
            return iter(self.prepended)

    def _scheduler_output() -> SchedulerOutput:
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.num_scheduled_tokens = {
            "req-0": 1,
            "req-1": 1,
        }
        scheduler_output.total_num_scheduled_tokens = 2
        return scheduler_output

    waiting = _WaitingQueue()
    requests = {
        req_id: SimpleNamespace(
            request_id=req_id,
            status=RequestStatus.RUNNING,
            num_computed_tokens=16,
            num_output_placeholders=1,
            spec_token_ids=[],
            num_preemptions=0,
            discard_latest_async_tokens=False,
        )
        for req_id in ("req-0", "req-1")
    }

    def _preempt_request(request, _timestamp):
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        request.discard_latest_async_tokens = True
        waiting.prepend_request(request)

    fake_scheduler = SimpleNamespace(
        perf_metrics=None,
        connector=None,
        kv_cache_manager=SimpleNamespace(take_events=lambda: None),
        kv_event_publisher=SimpleNamespace(publish=lambda batch: None),
        finished_req_ids_dict={},
        make_stats=lambda *args, **kwargs: None,
        requests=requests,
        running=list(requests.values()),
        waiting=waiting,
        prev_step_scheduled_req_ids=set(requests),
        _preempt_request=_preempt_request,
        _tpu_reload_generation=34,
    )
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
    error_messages: list[str] = []

    def _capture_error(message, *args, **kwargs):
        del kwargs
        error_messages.append(message % args if args else message)

    monkeypatch.setattr(runtime_patches.logger, "error", _capture_error)

    Scheduler.update_from_output(
        fake_scheduler,
        _scheduler_output(),
        model_runner_output,
    )

    assert any(
        "filtered every scheduled model output" in message
        for message in error_messages
    )
    recovery = fake_scheduler._tpu_last_live_reload_recovery
    assert recovery["reload_generation"] == 34
    assert recovery["model_runner_output"]["req_id_to_index_count"] == 0
    assert recovery["scheduler_output_before_filter"][
        "total_num_scheduled_tokens"] == 2
    assert recovery["scheduler_output_after_filter"][
        "total_num_scheduled_tokens"] == 0
    assert recovery["scheduler"]["request_count"] == 2
    assert all(request.discard_latest_async_tokens for request in requests.values())


def test_stale_model_output_wave_requeues_once_and_request_resumes(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    apply_vllm_runtime_patches()

    class _WaitingQueue:

        def __init__(self):
            self.prepended: list[Request] = []

        def prepend_request(self, request):
            if request not in self.prepended:
                self.prepended.append(request)

        def remove_requests(self, requests):
            request_set = set(requests)
            self.prepended = [
                request for request in self.prepended
                if request not in request_set
            ]

        def __iter__(self):
            return iter(self.prepended)

    def _scheduler_output(req_id: str) -> SchedulerOutput:
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.num_scheduled_tokens = {req_id: 1}
        scheduler_output.total_num_scheduled_tokens = 1
        return scheduler_output

    req_id = "req-live-reload-wave"
    waiting = _WaitingQueue()
    request = Request(
        request_id=req_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        eos_token_id=0,
    )
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 3
    request.num_output_placeholders = 1
    preempt_calls: list[int] = []

    def _preempt_request(request, _timestamp):
        preempt_calls.append(request.num_preemptions)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.spec_token_ids.clear()
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        request.discard_latest_async_tokens = True
        waiting.prepend_request(request)

    fake_scheduler = object.__new__(AsyncScheduler)
    fake_scheduler.perf_metrics = None
    fake_scheduler.connector = None
    fake_scheduler.kv_cache_manager = SimpleNamespace(
        take_events=lambda: None,
        cache_blocks=lambda *args: None,
    )
    fake_scheduler.kv_event_publisher = SimpleNamespace(
        publish=lambda batch: None)
    fake_scheduler.finished_req_ids_dict = {}
    fake_scheduler.make_stats = lambda *args, **kwargs: None
    fake_scheduler.make_spec_decoding_stats = lambda *args, **kwargs: None
    fake_scheduler.structured_output_manager = SimpleNamespace(
        should_advance=lambda request: False)
    fake_scheduler.requests = {req_id: request}
    fake_scheduler.running = [request]
    fake_scheduler.waiting = waiting
    fake_scheduler.prev_step_scheduled_req_ids = {req_id}
    fake_scheduler._preempt_request = _preempt_request
    fake_scheduler.max_model_len = 128
    fake_scheduler._tpu_reload_generation = 11
    missing_model_output = SimpleNamespace(
        sampled_token_ids=[],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={},
    )
    warning_messages: list[str] = []

    def _capture_warning(message, *args, **kwargs):
        del kwargs
        warning_messages.append(message % args if args else message)

    monkeypatch.setattr(runtime_patches.logger, "warning", _capture_warning)

    for _ in range(100):
        Scheduler.update_from_output(
            fake_scheduler,
            _scheduler_output(req_id),
            missing_model_output,
        )

    stale_drop_warnings = [
        message for message in warning_messages
        if "Dropping stale model output entries missing request indices"
        in message
    ]
    assert len(stale_drop_warnings) == 1
    assert preempt_calls == [0]
    assert request.status == RequestStatus.PREEMPTED
    assert request.discard_latest_async_tokens is True
    assert list(request.output_token_ids) == []
    assert waiting.prepended == [request]

    waiting.prepended.clear()
    fake_scheduler.running = [request]
    fake_scheduler.prev_step_scheduled_req_ids = {req_id}
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 3
    request.num_output_placeholders = 1

    valid_model_output = SimpleNamespace(
        sampled_token_ids=[[42]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        num_nans_in_logits=None,
        kv_connector_output=None,
        cudagraph_stats=None,
        req_id_to_index={req_id: 0},
    )

    Scheduler.update_from_output(
        fake_scheduler,
        _scheduler_output(req_id),
        valid_model_output,
    )

    assert list(request.output_token_ids) == [42]
    assert request.num_output_placeholders == 0
    assert request.discard_latest_async_tokens is False
    assert request.status == RequestStatus.RUNNING
    assert waiting.prepended == []


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


def test_requeue_missing_live_reload_requests_resets_stale_waiting_request_state():
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
        num_computed_tokens=128,
        num_output_placeholders=3,
        spec_token_ids=[7, 8],
        num_preemptions=1,
        discard_latest_async_tokens=False,
    )
    waiting.prepend_request(request)

    scheduler = SimpleNamespace(
        requests={"req-0": request},
        running=[],
        waiting=waiting,
        prev_step_scheduled_req_ids={"req-0"},
        _preempt_request=lambda *_args: None,
        _tpu_reload_generation=2,
    )

    requeue_missing_live_reload_requests(scheduler, ("req-0",))

    assert waiting.prepended == [request]
    assert scheduler.prev_step_scheduled_req_ids == set()
    assert request.num_computed_tokens == 0
    assert request.spec_token_ids == []
    assert request.num_output_placeholders == 0
    assert request.num_preemptions == 1
    assert request.discard_latest_async_tokens is True


def test_requeue_missing_live_reload_requests_is_idempotent_within_reload_generation():
    class _WaitingQueue:

        def __init__(self):
            self.prepended: list[object] = []

        def prepend_request(self, request):
            if request not in self.prepended:
                self.prepended.append(request)

        def remove(self, request):
            self.prepended.remove(request)

        def __iter__(self):
            return iter(self.prepended)

    waiting = _WaitingQueue()
    request = SimpleNamespace(
        request_id="req-0",
        status=RequestStatus.RUNNING,
        num_computed_tokens=64,
        num_output_placeholders=1,
        spec_token_ids=[3],
        num_preemptions=0,
        discard_latest_async_tokens=False,
    )
    preempt_calls: list[int] = []

    def _preempt_request(request, _timestamp):
        preempt_calls.append(request.num_preemptions)
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
        _tpu_reload_generation=4,
    )

    requeue_missing_live_reload_requests(scheduler, ("req-0",))

    waiting.remove(request)
    scheduler.running.append(request)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 17
    request.num_output_placeholders = 2
    request.spec_token_ids = [11]
    request.discard_latest_async_tokens = False

    requeue_missing_live_reload_requests(scheduler, ("req-0",))

    assert preempt_calls == [0]
    assert request.num_preemptions == 1
    assert scheduler.running == []
    assert waiting.prepended == [request]
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert request.spec_token_ids == []
    assert request.num_output_placeholders == 0
    assert request.discard_latest_async_tokens is True


def test_requeue_missing_live_reload_requests_allows_resume_after_stale_output_wave():
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    patch_async_scheduler_preempt_discard()

    class _WaitingQueue:

        def __init__(self):
            self.prepended: list[Request] = []

        def prepend_request(self, request):
            if request not in self.prepended:
                self.prepended.append(request)

        def remove(self, request):
            self.prepended.remove(request)

        def __iter__(self):
            return iter(self.prepended)

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
        _tpu_reload_generation=7,
    )
    fake_async_scheduler = object.__new__(AsyncScheduler)
    fake_async_scheduler.max_model_len = 128
    fake_async_scheduler.kv_cache_manager = SimpleNamespace(
        cache_blocks=lambda *args: None)

    requeue_missing_live_reload_requests(scheduler, ("req-0",))

    waiting.remove(request)
    scheduler.running.append(request)
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = 19
    request.num_output_placeholders = 2
    request.spec_token_ids = [11]
    request.discard_latest_async_tokens = False

    requeue_missing_live_reload_requests(scheduler, ("req-0",))

    waiting.remove(request)
    request.status = RequestStatus.RUNNING
    request.num_output_placeholders = 2

    new_token_ids, stopped = AsyncScheduler._update_request_with_output(
        fake_async_scheduler,
        request,
        [41, 42],
    )

    assert (new_token_ids, stopped) == ([41, 42], False)
    assert list(request.output_token_ids) == [41, 42]
    assert request.num_preemptions == 1
    assert request.discard_latest_async_tokens is False


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
