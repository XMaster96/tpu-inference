# SPDX-License-Identifier: Apache-2.0

import copy
import os
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from tpu_inference.logger import init_logger

logger = init_logger(__name__)
_T = TypeVar("_T")
_DEBUG_RELOAD_RACE = os.environ.get("TPU_DEBUG_RELOAD_RACE") == "1"


def get_reload_generation(scheduler) -> int:
    return int(getattr(scheduler, "_tpu_reload_generation", 0))


def mark_scheduler_output_reload_generation(scheduler, scheduler_output) -> None:
    setattr(
        scheduler_output,
        "_tpu_reload_generation",
        get_reload_generation(scheduler),
    )


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

    filtered_output = copy.copy(scheduler_output)
    filtered_output.num_scheduled_tokens = {
        req_id: num_tokens
        for req_id, num_tokens in num_scheduled_tokens.items()
        if req_id in req_id_to_index
    }
    filtered_output.scheduled_spec_decode_tokens = {
        req_id: token_ids
        for req_id, token_ids in getattr(
            scheduler_output, "scheduled_spec_decode_tokens", {}
        ).items()
        if req_id in req_id_to_index
    }
    return filtered_output, missing_req_ids


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
            # Live reload preemption already invalidates stale async outputs at
            # the scheduler boundary. Clearing this flag unconditionally avoids
            # discarding the first valid post-reload token for resumed requests
            # that are still transitioning through WAITING/PREEMPTED states.
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
        if (
            _DEBUG_RELOAD_RACE
            and scheduler_output.total_num_scheduled_tokens == 0
            and getattr(self, "requests", None)
        ):
            now = time.monotonic()
            last_log = float(getattr(self, "_tpu_debug_last_empty_schedule_log", 0.0))
            if now - last_log >= 1.0:
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
                logger.warning(
                    "Empty scheduler output with live requests | waiting=%s | "
                    "running=%s | request_states=%s",
                    waiting_ids,
                    running_ids,
                    request_states,
                )
                self._tpu_debug_last_empty_schedule_log = now
        mark_scheduler_output_reload_generation(self, scheduler_output)
        return scheduler_output

    def _patched_reset_prefix_cache(self, *args, **kwargs):
        reset_running_requests = bool(kwargs.get("reset_running_requests", False))
        if args:
            reset_running_requests = bool(args[0])
        if _DEBUG_RELOAD_RACE:
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
            self._tpu_reload_generation = get_reload_generation(self) + 1
        if _DEBUG_RELOAD_RACE:
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
                logger.warning(
                    "Dropping stale scheduler output after live reload preemption "
                    "for request ids: %s",
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
            logger.warning(
                "Dropping stale model output entries missing request indices "
                "after live reload preemption for request ids: %s",
                missing_req_ids,
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
    patch_async_llm_request_admission_gate()
    patch_async_scheduler_preempt_discard()
    patch_scheduler_reload_stale_output()
