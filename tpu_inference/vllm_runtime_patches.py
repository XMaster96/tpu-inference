# SPDX-License-Identifier: Apache-2.0

import copy
import faulthandler
import os
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from tpu_inference.logger import init_logger
from tpu_inference.runner.logprob_options import (
    RETURN_NATIVE_TOKEN_LOGPROBS_EXTRA_ARG,
)

logger = init_logger(__name__)
_T = TypeVar("_T")
_DEBUG_RELOAD_RACE = os.environ.get("TPU_DEBUG_RELOAD_RACE") == "1"
_LIVE_RELOAD_NO_PROGRESS_LOG_INTERVAL_SECONDS = float(
    os.environ.get("TPU_LIVE_RELOAD_NO_PROGRESS_LOG_INTERVAL_SECONDS", "30.0")
)


def _safe_len(value) -> int | None:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return None


def _truncate_values(values, limit: int = 8) -> list[object]:
    values = list(values)
    if len(values) <= limit:
        return values
    return values[:limit] + [f"...(+{len(values) - limit} more)"]


def _request_id_sample(requests, limit: int = 8) -> list[str]:
    if requests is None:
        return []

    try:
        if isinstance(requests, dict):
            request_ids = [str(req_id) for req_id in requests.keys()]
        else:
            request_ids = [
                str(getattr(request, "request_id", request))
                for request in requests
            ]
    except Exception as exc:
        return [f"<unavailable:{type(exc).__name__}>"]

    return [str(value) for value in _truncate_values(request_ids, limit=limit)]


def _parse_proc_kb_fields(path: str, fields: set[str]) -> dict[str, object]:
    values: dict[str, object] = {}
    try:
        with open(path, encoding="utf-8") as proc_file:
            for line in proc_file:
                key, _, rest = line.partition(":")
                if key not in fields:
                    continue
                raw_value = rest.strip().split()[0]
                values[f"{key}_gib"] = round(int(raw_value) / (1024**2), 3)
    except Exception as exc:
        return {"error": type(exc).__name__}
    return values


def _host_memory_snapshot() -> dict[str, object]:
    return _parse_proc_kb_fields(
        "/proc/meminfo",
        {
            "MemTotal",
            "MemFree",
            "MemAvailable",
            "AnonPages",
            "PageTables",
            "CommitLimit",
            "Committed_AS",
        },
    )


def _process_memory_snapshot() -> dict[str, object]:
    return _parse_proc_kb_fields(
        "/proc/self/status",
        {
            "VmPeak",
            "VmSize",
            "VmRSS",
            "VmHWM",
            "VmData",
        },
    )


def _request_state_summary(request) -> dict[str, object]:
    return {
        "request_id": str(getattr(request, "request_id", None)),
        "status": str(getattr(request, "status", None)),
        "num_computed_tokens": getattr(request, "num_computed_tokens", None),
        "num_output_placeholders": getattr(
            request, "num_output_placeholders", None),
        "num_preemptions": getattr(request, "num_preemptions", None),
        "discard_latest_async_tokens": getattr(
            request, "discard_latest_async_tokens", None),
    }


def _scheduler_state_snapshot(scheduler) -> dict[str, object]:
    return {
        "reload_generation": get_reload_generation(scheduler),
        "request_count": _safe_len(getattr(scheduler, "requests", None)),
        "waiting_count": _safe_len(getattr(scheduler, "waiting", None)),
        "running_count": _safe_len(getattr(scheduler, "running", None)),
        "waiting_ids": _request_id_sample(getattr(scheduler, "waiting", None)),
        "running_ids": _request_id_sample(getattr(scheduler, "running", None)),
        "prev_step_scheduled_req_count": _safe_len(
            getattr(scheduler, "prev_step_scheduled_req_ids", None)
        ),
    }


def _scheduler_output_snapshot(scheduler_output) -> dict[str, object]:
    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    return {
        "total_num_scheduled_tokens": getattr(
            scheduler_output, "total_num_scheduled_tokens", None),
        "num_scheduled_tokens": dict(
            getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        ),
        "scheduled_new_req_ids": [
            str(getattr(req_data, "req_id", "<unknown>"))
            for req_data in getattr(
                scheduler_output, "scheduled_new_reqs", ()
            ) or ()
        ],
        "scheduled_cached_req_ids": _request_id_sample(
            getattr(cached_reqs, "req_ids", ())),
        "finished_req_ids": _request_id_sample(
            getattr(scheduler_output, "finished_req_ids", ())),
        "preempted_req_ids": _request_id_sample(
            getattr(scheduler_output, "preempted_req_ids", ())),
    }


def _model_runner_output_snapshot(model_runner_output) -> dict[str, object]:
    req_id_to_index = getattr(model_runner_output, "req_id_to_index", {}) or {}
    sampled_token_ids = getattr(model_runner_output, "sampled_token_ids", None)
    return {
        "req_id_to_index_count": _safe_len(req_id_to_index),
        "req_id_to_index_ids": _request_id_sample(req_id_to_index),
        "sampled_token_rows": _safe_len(sampled_token_ids),
        "has_sampled_token_ids": sampled_token_ids is not None,
    }


def _public_recovery_snapshot(recovery: dict[str, object] | None):
    if not recovery:
        return None
    public = dict(recovery)
    monotonic_time = public.pop("monotonic_time", None)
    if isinstance(monotonic_time, (int, float)):
        public["age_seconds"] = round(time.monotonic() - monotonic_time, 3)
    return public


def _last_live_reload_recovery_snapshot(scheduler):
    return _public_recovery_snapshot(
        getattr(scheduler, "_tpu_last_live_reload_recovery", None))


def _record_live_reload_recovery(
    scheduler,
    *,
    missing_req_ids,
    scheduler_output,
    filtered_output,
    model_runner_output,
    requeue_actions: list[dict[str, object]],
) -> dict[str, object]:
    recovery = {
        "monotonic_time": time.monotonic(),
        "reload_generation": get_reload_generation(scheduler),
        "missing_req_ids": _request_id_sample(missing_req_ids),
        "missing_req_count": len(missing_req_ids),
        "scheduler": _scheduler_state_snapshot(scheduler),
        "scheduler_output_before_filter": _scheduler_output_snapshot(
            scheduler_output),
        "scheduler_output_after_filter": _scheduler_output_snapshot(
            filtered_output),
        "model_runner_output": _model_runner_output_snapshot(
            model_runner_output),
        "requeue_actions": _summarize_requeue_actions(requeue_actions),
        "host_memory": _host_memory_snapshot(),
        "process_memory": _process_memory_snapshot(),
    }
    setattr(scheduler, "_tpu_last_live_reload_recovery", recovery)
    return recovery


def _annotate_scheduler_output_debug_context(scheduler, scheduler_output) -> None:
    setattr(
        scheduler_output,
        "_tpu_scheduler_state_snapshot",
        _scheduler_state_snapshot(scheduler),
    )
    recovery = _last_live_reload_recovery_snapshot(scheduler)
    if recovery is not None:
        setattr(scheduler_output, "_tpu_last_live_reload_recovery", recovery)


def _request_context_snapshot(
    scheduler,
    request_ids,
    *,
    limit: int = 4,
) -> dict[str, dict[str, object]]:
    requests = getattr(scheduler, "requests", {}) or {}
    context: dict[str, dict[str, object]] = {}
    for request_id in list(request_ids)[:limit]:
        request = requests.get(request_id)
        if request is None:
            context[str(request_id)] = {"status": "<missing-request-object>"}
            continue
        context[str(request_id)] = _request_state_summary(request)
    omitted = len(tuple(request_ids)) - len(context)
    if omitted > 0:
        context["..."] = {"omitted_requests": omitted}
    return context


def _should_log_reload_event_context(
    scheduler,
    event_name: str,
    request_ids,
) -> bool:
    logged_events = getattr(scheduler, "_tpu_reload_event_context_logged", None)
    if logged_events is None:
        logged_events = set()
        setattr(scheduler, "_tpu_reload_event_context_logged", logged_events)

    key = (
        event_name,
        get_reload_generation(scheduler),
        tuple(str(request_id) for request_id in request_ids),
    )
    if key in logged_events:
        return False

    if len(logged_events) >= 256:
        logged_events.clear()
    logged_events.add(key)
    return True


def _log_reload_event_context_once(
    scheduler,
    *,
    event_name: str,
    request_ids,
    message: str,
    extra: dict[str, object] | None = None,
) -> bool:
    if not _should_log_reload_event_context(scheduler, event_name, request_ids):
        return False

    logger.warning(
        "%s | request_ids=%s | scheduler=%s | request_context=%s | extra=%s",
        message,
        tuple(str(request_id) for request_id in request_ids),
        _scheduler_state_snapshot(scheduler),
        _request_context_snapshot(scheduler, request_ids),
        extra or {},
    )
    return True


def _should_log_request_event_context(
    request,
    event_name: str,
) -> bool:
    logged_events = getattr(
        request,
        "_tpu_reload_request_event_context_logged",
        None,
    )
    if logged_events is None:
        logged_events = set()
        setattr(
            request,
            "_tpu_reload_request_event_context_logged",
            logged_events,
        )

    key = (
        event_name,
        int(getattr(request, "_tpu_last_reload_requeue_generation", -1) or -1),
    )
    if key in logged_events:
        return False

    if len(logged_events) >= 32:
        logged_events.clear()
    logged_events.add(key)
    return True


def _log_request_event_context_once(
    request,
    *,
    event_name: str,
    message: str,
    extra: dict[str, object] | None = None,
) -> None:
    if not _should_log_request_event_context(request, event_name):
        return

    logger.warning(
        "%s | request=%s | extra=%s",
        message,
        _request_state_summary(request),
        extra or {},
    )


def _summarize_requeue_actions(
    requeue_actions: list[dict[str, object]],
) -> dict[str, object]:
    action_counts: dict[str, int] = {}
    for action in requeue_actions:
        action_name = str(action.get("action", "<unknown>"))
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

    return {
        "count": len(requeue_actions),
        "action_counts": action_counts,
        "sample": _truncate_values(requeue_actions, limit=4),
    }


def get_reload_generation(scheduler) -> int:
    return int(getattr(scheduler, "_tpu_reload_generation", 0))


def mark_scheduler_output_reload_generation(scheduler, scheduler_output) -> None:
    setattr(
        scheduler_output,
        "_tpu_reload_generation",
        get_reload_generation(scheduler),
    )
    _annotate_scheduler_output_debug_context(scheduler, scheduler_output)


def is_stale_scheduler_output(scheduler, scheduler_output) -> bool:
    output_generation = getattr(scheduler_output, "_tpu_reload_generation", None)
    if output_generation is None:
        return False
    return int(output_generation) != get_reload_generation(scheduler)


def filter_scheduler_output_missing_req_indices(
    scheduler_output,
    model_runner_output,
):
    req_id_to_index = getattr(model_runner_output, "req_id_to_index", {}) or {}
    num_scheduled_tokens = dict(
        getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
    )
    missing_req_ids = tuple(
        req_id for req_id in num_scheduled_tokens if req_id not in req_id_to_index
    )
    if not missing_req_ids:
        return scheduler_output, ()

    kept_req_ids = set(num_scheduled_tokens) - set(missing_req_ids)
    filtered_output = copy.copy(scheduler_output)
    filtered_output.num_scheduled_tokens = {
        req_id: num_tokens
        for req_id, num_tokens in num_scheduled_tokens.items()
        if req_id in kept_req_ids
    }
    filtered_output.total_num_scheduled_tokens = sum(
        filtered_output.num_scheduled_tokens.values()
    )
    filtered_output.scheduled_spec_decode_tokens = {
        req_id: token_ids
        for req_id, token_ids in getattr(
            scheduler_output, "scheduled_spec_decode_tokens", {}
        ).items()
        if req_id in kept_req_ids
    }
    filtered_output.scheduled_encoder_inputs = {
        req_id: encoder_inputs
        for req_id, encoder_inputs in getattr(
            scheduler_output, "scheduled_encoder_inputs", {}
        ).items()
        if req_id in kept_req_ids
    }
    filtered_output.scheduled_new_reqs = [
        req_data
        for req_data in getattr(scheduler_output, "scheduled_new_reqs", ())
        if getattr(req_data, "req_id", None) in kept_req_ids
    ]

    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached_reqs is not None:
        keep_indices = [
            index
            for index, req_id in enumerate(getattr(cached_reqs, "req_ids", ()))
            if req_id in kept_req_ids
        ]
        filtered_cached_reqs = copy.copy(cached_reqs)
        filtered_cached_reqs.req_ids = [
            cached_reqs.req_ids[index] for index in keep_indices
        ]
        filtered_cached_reqs.resumed_req_ids = {
            req_id
            for req_id in getattr(cached_reqs, "resumed_req_ids", set())
            if req_id in kept_req_ids
        }
        filtered_cached_reqs.new_token_ids = [
            cached_reqs.new_token_ids[index] for index in keep_indices
        ]
        filtered_cached_reqs.all_token_ids = {
            req_id: token_ids
            for req_id, token_ids in getattr(cached_reqs, "all_token_ids", {}).items()
            if req_id in kept_req_ids
        }
        filtered_cached_reqs.new_block_ids = [
            cached_reqs.new_block_ids[index] for index in keep_indices
        ]
        filtered_cached_reqs.num_computed_tokens = [
            cached_reqs.num_computed_tokens[index] for index in keep_indices
        ]
        filtered_cached_reqs.num_output_tokens = [
            cached_reqs.num_output_tokens[index] for index in keep_indices
        ]
        filtered_cached_reqs.__dict__.pop("_req_id_to_num_output_tokens", None)
        filtered_output.scheduled_cached_reqs = filtered_cached_reqs

    assigned_dp_rank = getattr(scheduler_output, "assigned_dp_rank", None)
    if assigned_dp_rank is not None:
        filtered_output.assigned_dp_rank = {
            req_id: dp_rank
            for req_id, dp_rank in assigned_dp_rank.items()
            if req_id in kept_req_ids
        }

    num_invalid_spec_tokens = getattr(
        scheduler_output, "num_invalid_spec_tokens", None)
    if num_invalid_spec_tokens is not None:
        filtered_output.num_invalid_spec_tokens = {
            req_id: count
            for req_id, count in num_invalid_spec_tokens.items()
            if req_id in kept_req_ids
        }

    return filtered_output, missing_req_ids


def requeue_missing_live_reload_requests(
    scheduler,
    missing_req_ids,
) -> list[dict[str, object]]:
    if not missing_req_ids:
        return []

    prev_step_scheduled_req_ids = getattr(
        scheduler, "prev_step_scheduled_req_ids", None)
    if prev_step_scheduled_req_ids is not None:
        prev_step_scheduled_req_ids.difference_update(missing_req_ids)

    running = getattr(scheduler, "running", None)
    waiting = getattr(scheduler, "waiting", None)
    preempt_request = getattr(scheduler, "_preempt_request", None)
    timestamp = time.monotonic()
    reload_generation = get_reload_generation(scheduler)

    try:
        from vllm.v1.request import RequestStatus
    except Exception:
        RequestStatus = None

    def _is_in_waiting_queue(request) -> bool:
        if waiting is None:
            return False

        try:
            return any(queued_request is request for queued_request in waiting)
        except TypeError:
            return False

    def _free_request_caches(request) -> None:
        for manager_name in ("kv_cache_manager", "encoder_cache_manager"):
            manager = getattr(scheduler, manager_name, None)
            free = getattr(manager, "free", None)
            if callable(free):
                free(request)

    def _reset_request_for_live_reload(
        request,
        *,
        increment_preemptions: bool,
        free_caches: bool,
    ) -> None:
        if free_caches:
            _free_request_caches(request)
        if RequestStatus is not None:
            request.status = RequestStatus.PREEMPTED
        if hasattr(request, "num_computed_tokens"):
            request.num_computed_tokens = 0
        if hasattr(request, "spec_token_ids"):
            request.spec_token_ids.clear()
        if increment_preemptions and hasattr(request, "num_preemptions"):
            request.num_preemptions += 1
        if hasattr(request, "num_output_placeholders"):
            request.num_output_placeholders = 0
        if hasattr(request, "discard_latest_async_tokens"):
            request.discard_latest_async_tokens = True
        setattr(
            request,
            "_tpu_last_reload_requeue_generation",
            reload_generation,
        )

    def _already_requeued_for_generation(request) -> bool:
        return int(
            getattr(
                request,
                "_tpu_last_reload_requeue_generation",
                -1,
            )
            or -1
        ) == reload_generation

    requeue_actions: list[dict[str, object]] = []

    for req_id in missing_req_ids:
        request = getattr(scheduler, "requests", {}).get(req_id)
        if request is None:
            continue

        removed_from_running = False
        was_in_waiting = _is_in_waiting_queue(request)
        if isinstance(running, list):
            try:
                running.remove(request)
                removed_from_running = True
            except ValueError:
                pass

        request_status = getattr(request, "status", None)
        already_preempted = (
            RequestStatus is not None
            and request_status == RequestStatus.PREEMPTED
        )
        already_waiting = (
            RequestStatus is not None
            and request_status == RequestStatus.WAITING
        )
        already_requeued_this_generation = _already_requeued_for_generation(request)

        if (
            callable(preempt_request)
            and removed_from_running
            and not already_requeued_this_generation
        ):
            preempt_request(request, timestamp)
            if hasattr(request, "num_output_placeholders"):
                request.num_output_placeholders = 0
            if hasattr(request, "discard_latest_async_tokens"):
                request.discard_latest_async_tokens = True
            setattr(
                request,
                "_tpu_last_reload_requeue_generation",
                reload_generation,
            )
            action = "preempt_running"
        elif already_preempted or already_waiting:
            _reset_request_for_live_reload(
                request,
                increment_preemptions=False,
                free_caches=False,
            )
            if not _is_in_waiting_queue(request) and hasattr(
                waiting, "prepend_request"
            ):
                waiting.prepend_request(request)
                action = "restore_waiting_queue"
            else:
                action = "refresh_waiting_request"
        else:
            _reset_request_for_live_reload(
                request,
                increment_preemptions=not already_requeued_this_generation,
                free_caches=removed_from_running,
            )
            if hasattr(waiting, "prepend_request"):
                waiting.prepend_request(request)
            action = (
                "requeue_running_request"
                if removed_from_running and already_requeued_this_generation
                else "mark_preempted_and_prepend"
            )

        requeue_actions.append(
            {
                "request_id": str(req_id),
                "action": action,
                "removed_from_running": removed_from_running,
                "was_in_waiting": was_in_waiting,
                "is_in_waiting": _is_in_waiting_queue(request),
                "reload_generation": reload_generation,
                **_request_state_summary(request),
            }
        )

    return requeue_actions


def patch_async_scheduler_preempt_discard() -> None:
    """Patch AsyncScheduler to avoid discarding valid post-reload tokens."""
    try:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    except Exception:
        return

    if getattr(AsyncScheduler, "_tpu_preempt_discard_patch_installed", False):
        return

    original_update_with_output = AsyncScheduler._update_request_with_output

    def _patched_update_request_with_output(self, request, new_token_ids):
        if getattr(request, "discard_latest_async_tokens", False):
            num_output_placeholders = int(
                getattr(request, "num_output_placeholders", 0) or 0)
            if num_output_placeholders < len(new_token_ids):
                # Forced reload preemption resets placeholders to zero. If an
                # async output arrives before the request has been rescheduled
                # for at least this many fresh output tokens, it still belongs
                # to the pre-reload execution and must be dropped.
                _log_request_event_context_once(
                    request,
                    event_name="drop_stale_async_output",
                    message=(
                        "Dropping stale async output after live reload "
                        "preemption before fresh placeholders were restored"
                    ),
                    extra={
                        "reload_generation": getattr(
                            request,
                            "_tpu_last_reload_requeue_generation",
                            None,
                        ),
                        "incoming_token_count": len(new_token_ids),
                        "num_output_placeholders": num_output_placeholders,
                    },
                )
                request.discard_latest_async_tokens = False
                return [], False

            # The request has already been rescheduled for enough new output
            # tokens, so this async result is from the post-reload execution.
            request.discard_latest_async_tokens = False
        return original_update_with_output(self, request, new_token_ids)

    AsyncScheduler._update_request_with_output = _patched_update_request_with_output
    AsyncScheduler._tpu_preempt_discard_patch_installed = True


async def run_with_request_admission_gate(
    engine,
    op: Callable[[], Awaitable[_T]],
) -> _T:
    gate = getattr(engine, "_tpu_reload_request_gate", None)
    if gate is None:
        return await op()

    if _DEBUG_RELOAD_RACE and gate.locked():
        logger.warning("Request admission is waiting for the live-reload gate.")
    async with gate:
        if _DEBUG_RELOAD_RACE:
            logger.warning("Request admission acquired the live-reload gate.")
        return await op()


def patch_async_llm_request_admission_gate() -> None:
    """Patch AsyncLLM request admission to serialize with live reload."""
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception:
        return

    if getattr(AsyncLLM, "_tpu_request_admission_gate_patch_installed", False):
        return

    original_add_request = AsyncLLM.add_request

    async def _patched_add_request(
        self,
        request_id,
        prompt,
        params,
        *args,
        **kwargs,
    ):
        if _DEBUG_RELOAD_RACE:
            logger.warning(
                "AsyncLLM.add_request start | request_id=%s | paused=%s",
                request_id,
                getattr(self, "_paused", None),
            )

        result = await run_with_request_admission_gate(
            self,
            lambda: original_add_request(
                self,
                request_id,
                prompt,
                params,
                *args,
                **kwargs,
            ),
        )
        if _DEBUG_RELOAD_RACE:
            logger.warning(
                "AsyncLLM.add_request done | request_id=%s | paused=%s",
                request_id,
                getattr(self, "_paused", None),
            )
        return result

    AsyncLLM.add_request = _patched_add_request
    AsyncLLM._tpu_request_admission_gate_patch_installed = True


def patch_native_logprob_eos_detokenization() -> None:
    """Keep EOS text for native-logprob requests without changing stop strings."""
    try:
        from vllm.v1.engine.detokenizer import BaseIncrementalDetokenizer
    except Exception:
        return

    if getattr(
        BaseIncrementalDetokenizer,
        "_tpu_native_logprob_eos_detokenization_patch_installed",
        False,
    ):
        return

    original_init = BaseIncrementalDetokenizer.__init__
    original_update = BaseIncrementalDetokenizer.update

    def _patched_init(self, request, *args, **kwargs):
        original_init(self, request, *args, **kwargs)
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None) or {}
        self._tpu_return_native_token_logprobs = bool(
            extra_args.get(RETURN_NATIVE_TOKEN_LOGPROBS_EXTRA_ARG))
        self._tpu_eos_token_id = getattr(request, "eos_token_id", None)

    def _patched_update(self, new_token_ids, stop_terminated):
        if (stop_terminated and new_token_ids
                and getattr(self, "_tpu_return_native_token_logprobs", False)
                and new_token_ids[-1] == getattr(self, "_tpu_eos_token_id",
                                                 None)):
            return original_update(self, new_token_ids, False)
        return original_update(self, new_token_ids, stop_terminated)

    BaseIncrementalDetokenizer.__init__ = _patched_init
    BaseIncrementalDetokenizer.update = _patched_update
    BaseIncrementalDetokenizer._tpu_native_logprob_eos_detokenization_patch_installed = (
        True)


def patch_scheduler_reload_stale_output() -> None:
    """Patch Scheduler to drop outputs invalidated by reload preemption."""
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        return

    if getattr(Scheduler, "_tpu_reload_stale_output_patch_installed", False):
        return

    original_schedule = Scheduler.schedule
    original_reset_prefix_cache = Scheduler.reset_prefix_cache
    original_update_from_output = Scheduler.update_from_output
    original_add_request = Scheduler.add_request
    original_finish_requests = Scheduler.finish_requests

    def _patched_schedule(self, *args, **kwargs):
        scheduler_output = original_schedule(self, *args, **kwargs)
        mark_scheduler_output_reload_generation(self, scheduler_output)
        live_requests = getattr(self, "requests", None)
        if not live_requests:
            setattr(self, "_tpu_last_live_reload_recovery", None)
        if (
            scheduler_output.total_num_scheduled_tokens == 0
            and live_requests
            and not getattr(scheduler_output, "finished_req_ids", None)
        ):
            now = time.monotonic()
            last_log = float(getattr(self, "_tpu_debug_last_empty_schedule_log", 0.0))
            recovery = _last_live_reload_recovery_snapshot(self)
            should_log = _DEBUG_RELOAD_RACE or recovery is not None
            if (
                should_log
                and now - last_log >= _LIVE_RELOAD_NO_PROGRESS_LOG_INTERVAL_SECONDS
            ):
                request_states = {
                    req_id: str(getattr(req, "status", "<unknown>"))
                    for req_id, req in self.requests.items()
                }
                waiting_ids = [
                    str(getattr(req, "request_id", "<unknown>"))
                    for req in list(getattr(self, "waiting", []))
                ]
                running_ids = [
                    str(getattr(req, "request_id", "<unknown>"))
                    for req in list(getattr(self, "running", []))
                ]
                log = logger.error if recovery is not None else logger.warning
                log(
                    "Scheduler produced empty output with live requests after "
                    "live reload recovery; EngineCore may be wedged if this "
                    "repeats | waiting=%s | running=%s | request_states=%s | "
                    "scheduler=%s | last_live_reload_recovery=%s | "
                    "host_memory=%s | process_memory=%s",
                    waiting_ids,
                    running_ids,
                    request_states,
                    _scheduler_state_snapshot(self),
                    recovery,
                    _host_memory_snapshot(),
                    _process_memory_snapshot(),
                )
                self._tpu_debug_last_empty_schedule_log = now
        return scheduler_output

    def _patched_reset_prefix_cache(self, *args, **kwargs):
        reset_running_requests = bool(kwargs.get("reset_running_requests", False))
        if args:
            reset_running_requests = bool(args[0])
        previous_generation = get_reload_generation(self)
        if reset_running_requests:
            logger.info(
                "Scheduler.reset_prefix_cache start for live reload | "
                "reset_running=%s | reload_generation=%s | scheduler=%s",
                reset_running_requests,
                previous_generation,
                _scheduler_state_snapshot(self),
            )
        elif _DEBUG_RELOAD_RACE:
            logger.warning(
                "Scheduler.reset_prefix_cache start | reset_running=%s | "
                "waiting=%s | running=%s | requests=%s",
                reset_running_requests,
                len(getattr(self, "waiting", [])),
                len(getattr(self, "running", [])),
                sorted(getattr(self, "requests", {}).keys()),
            )
        reset_ok = original_reset_prefix_cache(self, *args, **kwargs)
        if reset_running_requests and reset_ok:
            self._tpu_reload_generation = previous_generation + 1
        if reset_running_requests:
            logger.info(
                "Scheduler.reset_prefix_cache done for live reload | "
                "reset_running=%s | reset_ok=%s | reload_generation=%s->%s | "
                "scheduler=%s",
                reset_running_requests,
                reset_ok,
                previous_generation,
                get_reload_generation(self),
                _scheduler_state_snapshot(self),
            )
        elif _DEBUG_RELOAD_RACE:
            logger.warning(
                "Scheduler.reset_prefix_cache done | reset_running=%s | "
                "reset_ok=%s | waiting=%s | running=%s | requests=%s",
                reset_running_requests,
                reset_ok,
                len(getattr(self, "waiting", [])),
                len(getattr(self, "running", [])),
                sorted(getattr(self, "requests", {}).keys()),
            )
        return reset_ok

    def _patched_update_from_output(self, scheduler_output, model_runner_output):
        if is_stale_scheduler_output(self, scheduler_output):
            stale_req_ids = tuple(
                getattr(scheduler_output, "num_scheduled_tokens", {}).keys()
            )
            if stale_req_ids:
                logged_context = _log_reload_event_context_once(
                    self,
                    event_name="stale_scheduler_output",
                    request_ids=stale_req_ids,
                    message=(
                        "Live reload stale scheduler output context"
                    ),
                )
                if logged_context:
                    logger.warning(
                        "Dropping stale scheduler output after live reload "
                        "preemption for request ids: %s",
                        stale_req_ids,
                    )
            return {}

        filtered_output, missing_req_ids = (
            filter_scheduler_output_missing_req_indices(
                scheduler_output,
                model_runner_output,
            )
        )
        if missing_req_ids:
            requeue_actions = requeue_missing_live_reload_requests(
                self,
                missing_req_ids,
            )
            recovery = _record_live_reload_recovery(
                self,
                missing_req_ids=missing_req_ids,
                scheduler_output=scheduler_output,
                filtered_output=filtered_output,
                model_runner_output=model_runner_output,
                requeue_actions=requeue_actions,
            )
            logged_context = _log_reload_event_context_once(
                self,
                event_name="missing_request_indices",
                request_ids=missing_req_ids,
                message=(
                    "Live reload stale model output context"
                ),
                extra={
                    "requeue_actions": _summarize_requeue_actions(requeue_actions),
                    "missing_req_count": len(missing_req_ids),
                    "kept_req_ids": _request_id_sample(
                        getattr(filtered_output, "num_scheduled_tokens", {}),
                    ),
                    "model_runner_req_ids": _request_id_sample(
                        getattr(model_runner_output, "req_id_to_index", {}),
                    ),
                },
            )
            if logged_context:
                logger.warning(
                    "Dropping stale model output entries missing request "
                    "indices after live reload preemption for request ids: %s",
                    missing_req_ids,
                )
                if (
                    not getattr(filtered_output, "num_scheduled_tokens", None)
                    and getattr(self, "requests", None)
                ):
                    logger.error(
                        "Live reload recovery filtered every scheduled model "
                        "output while requests remain live; next logs should "
                        "show whether the scheduler rescheduled them or wedged "
                        "| recovery=%s",
                        _public_recovery_snapshot(recovery),
                    )
        return original_update_from_output(
            self,
            filtered_output,
            model_runner_output,
        )

    def _patched_add_request(self, request):
        if _DEBUG_RELOAD_RACE:
            logger.warning(
                "Scheduler.add_request start | req=%s | status=%s | waiting=%s | "
                "running=%s | requests=%s",
                getattr(request, "request_id", "<unknown>"),
                getattr(request, "status", "<unknown>"),
                len(getattr(self, "waiting", [])),
                len(getattr(self, "running", [])),
                sorted(getattr(self, "requests", {}).keys()),
            )
        result = original_add_request(self, request)
        if _DEBUG_RELOAD_RACE:
            logger.warning(
                "Scheduler.add_request done | req=%s | waiting=%s | running=%s | "
                "requests=%s",
                getattr(request, "request_id", "<unknown>"),
                len(getattr(self, "waiting", [])),
                len(getattr(self, "running", [])),
                sorted(getattr(self, "requests", {}).keys()),
            )
        return result

    def _patched_finish_requests(self, request_ids, finished_status):
        if _DEBUG_RELOAD_RACE:
            request_ids_list = (
                [request_ids] if isinstance(request_ids, str) else list(request_ids)
            )
            logger.warning(
                "Scheduler.finish_requests start | req_ids=%s | status=%s | "
                "waiting=%s | running=%s | requests=%s",
                request_ids_list,
                finished_status,
                len(getattr(self, "waiting", [])),
                len(getattr(self, "running", [])),
                sorted(getattr(self, "requests", {}).keys()),
            )
        result = original_finish_requests(self, request_ids, finished_status)
        if not getattr(self, "requests", None):
            setattr(self, "_tpu_last_live_reload_recovery", None)
        if _DEBUG_RELOAD_RACE:
            logger.warning(
                "Scheduler.finish_requests done | waiting=%s | running=%s | "
                "requests=%s",
                len(getattr(self, "waiting", [])),
                len(getattr(self, "running", [])),
                sorted(getattr(self, "requests", {}).keys()),
            )
        return result

    Scheduler.add_request = _patched_add_request
    Scheduler.finish_requests = _patched_finish_requests
    Scheduler.schedule = _patched_schedule
    Scheduler.reset_prefix_cache = _patched_reset_prefix_cache
    Scheduler.update_from_output = _patched_update_from_output
    Scheduler._tpu_reload_stale_output_patch_installed = True


def apply_vllm_runtime_patches() -> None:
    if not getattr(apply_vllm_runtime_patches, "_faulthandler_enabled", False):
        try:
            faulthandler.enable(all_threads=True)
            apply_vllm_runtime_patches._faulthandler_enabled = True
        except Exception as exc:
            logger.warning(
                "Could not enable faulthandler for EngineCore diagnostics: %s",
                exc,
            )
    patch_async_llm_request_admission_gate()
    patch_async_scheduler_preempt_discard()
    patch_native_logprob_eos_detokenization()
    patch_scheduler_reload_stale_output()
