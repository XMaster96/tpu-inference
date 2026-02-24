# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import inspect
from argparse import Namespace
from http import HTTPStatus

import uvloop
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client,
    init_app_state,
    setup_server,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.server_utils import (
    AuthenticationMiddleware,
    XRequestIdMiddleware,
    get_uvicorn_log_config,
    http_exception_handler,
    lifespan,
    log_response,
    validation_exception_handler,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.sagemaker.api_router import sagemaker_standards_bootstrap
from vllm.entrypoints.serve.elastic_ep.middleware import ScalingMiddleware
from vllm.entrypoints.serve.instrumentator.health import (
    attach_router as attach_health_router,
)
from vllm.entrypoints.serve.instrumentator.metrics import (
    attach_router as attach_metrics_router,
)
from vllm.entrypoints.utils import (
    cli_env_setup,
    load_aware_call,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import decorate_logs

from tpu_inference.models.jax.utils.weight_utils import _is_modlax_orbax_checkpoint

logger = init_logger("tpu_inference.entrypoints.online_rl_server")

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


class ReloadWeightsRequest(BaseModel):
    checkpoint_path: str | None = None
    wait_for_inflight_requests: bool = False
    clear_cache: bool = True
    release_kv_cache: bool = True
    timeout_seconds: float = Field(default=600.0, gt=0)


def completion(request: Request) -> OpenAIServingCompletion | None:
    return request.app.state.openai_serving_completion


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _weights_ready(app_state) -> bool:
    return (not app_state.require_first_reload) or app_state.first_weights_loaded.is_set()


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"application/json": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    # Explicitly disable SSE streaming while preserving all non-stream
    # completion behavior from vLLM (including structured outputs handling).
    if request.stream:
        handler = completion(raw_request)
        if handler is None:
            base_server = raw_request.app.state.openai_serving_tokenization
            error = base_server.create_error_response(
                message="The model does not support Completions API"
            )
        else:
            error = handler.create_error_response(
                message="Streaming is disabled on this server."
            )
        return JSONResponse(content=error.model_dump(), status_code=error.error.code)

    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = completion(raw_request)
    if handler is None:
        base_server = raw_request.app.state.openai_serving_tokenization
        error = base_server.create_error_response(
            message="The model does not support Completions API"
        )
        return JSONResponse(content=error.model_dump(), status_code=error.error.code)

    try:
        output = await handler.create_completion(request, raw_request)
    except Exception as exc:
        error = handler.create_error_response(exc)
        return JSONResponse(content=error.model_dump(), status_code=error.error.code)

    if isinstance(output, ErrorResponse):
        return JSONResponse(content=output.model_dump(), status_code=output.error.code)

    if isinstance(output, CompletionResponse):
        return JSONResponse(
            content=output.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    error = handler.create_error_response(
        message="Unexpected streaming output while streaming is disabled."
    )
    return JSONResponse(content=error.model_dump(), status_code=error.error.code)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    available_models = await models(raw_request).show_available_models()
    return JSONResponse(content=available_models.model_dump())


@router.get("/models")
async def show_available_models_alias(raw_request: Request):
    return await show_available_models(raw_request)


@router.get("/status")
async def status(raw_request: Request):
    app_state = raw_request.app.state
    ready = _weights_ready(app_state)
    return JSONResponse(
        status_code=HTTPStatus.OK.value if ready else HTTPStatus.SERVICE_UNAVAILABLE.value,
        content={
            "status": "ready" if ready else "waiting_for_initial_weights",
            "ready": ready,
            "checkpoint_path": app_state.current_checkpoint_path,
        },
    )


@router.post("/v1/reload_weights")
async def reload_weights(payload: ReloadWeightsRequest, raw_request: Request):
    """Reload model weights in-place without restarting the server process.

    This endpoint performs an online weight swap on all workers via
    ``reload_model_weights`` and keeps the compiled JAX executables alive.

    Request body:
    - ``checkpoint_path`` (str | null): Orbax checkpoint URI/path to load. If
      omitted, workers reload the current checkpoint already tracked by server
      state.
    - ``wait_for_inflight_requests`` (bool, default ``False``): If ``True``,
      pause waits for running requests to complete before reloading.
    - ``clear_cache`` (bool, default ``True``): Clears prefix/cache state during
      pause to reduce stale cache effects across checkpoints.
    - ``release_kv_cache`` (bool, default ``True``): Requests worker-side KV
      cache release before load. Recommended on TPU to reduce OOM risk.
    - ``timeout_seconds`` (float, default ``600``): RPC timeout budget for the
      distributed reload call.

    Behavior:
    - Reload operations are serialized with ``app_state.reload_lock``.
    - Generation is paused before reload and always resumed in ``finally``.
    - On first successful reload after dummy bootstrap, readiness flips to
      ready (`/health` and `/status` return 200; inference endpoints unblocked).
    - Architecture mismatches are rejected by worker-side compatibility checks.

    Success response (HTTP 200):
    - ``status``: ``"ok"``
    - ``checkpoint_path``: server-tracked active checkpoint path
    - ``worker_results``: per-worker reload details (timings, HBM stats, mode)

    Error response (HTTP 500):
    - Raised when pause/reload/resume orchestration fails, with
      ``detail = "Live model reload failed: <reason>"``.
    """
    app_state = raw_request.app.state
    engine = engine_client(raw_request)

    async with app_state.reload_lock:
        try:
            await engine.pause_generation(
                wait_for_inflight_requests=payload.wait_for_inflight_requests,
                clear_cache=payload.clear_cache,
            )
            rpc_args = (
                (payload.checkpoint_path,)
                if payload.checkpoint_path is not None
                else ()
            )
            worker_results = await engine.collective_rpc(
                method="reload_model_weights",
                timeout=payload.timeout_seconds,
                args=rpc_args,
                kwargs={"release_kv_cache": payload.release_kv_cache},
            )
        except Exception as exc:
            logger.exception("Live model reload failed")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail=f"Live model reload failed: {exc}",
            ) from exc
        finally:
            await engine.resume_generation()

        if payload.checkpoint_path is not None:
            app_state.current_checkpoint_path = payload.checkpoint_path
        app_state.first_weights_loaded.set()

    return JSONResponse(
        content={
            "status": "ok",
            "checkpoint_path": app_state.current_checkpoint_path,
            "worker_results": worker_results,
        }
    )


def _configure_initial_reload_mode(args: Namespace) -> tuple[bool, str | None]:
    checkpoint_path = getattr(args, "model_weights", None) or None
    if checkpoint_path is None:
        return False, None

    if _is_modlax_orbax_checkpoint(checkpoint_path):
        return False, checkpoint_path

    logger.warning(
        "Checkpoint path %r is not available yet. Booting with dummy weights "
        "to compile JIT first; requests will return 503 until /v1/reload_weights "
        "succeeds.",
        checkpoint_path,
    )
    # Avoid Run:AI object-store validation for startup in dummy mode.
    # We keep the intended checkpoint URI in app state for the first reload.
    args.model_weights = ""
    args.load_format = "dummy"
    return True, checkpoint_path


def _install_weight_gate(app: FastAPI) -> None:
    health_paths = {"/health", "/ping", "/status"}
    public_paths = {"/metrics", "/v1/reload_weights", "/openapi.json", "/redoc"}

    @app.middleware("http")
    async def reject_until_first_real_weights(request: Request, call_next):
        path = request.url.path.rstrip("/") or "/"
        app_state = request.app.state
        if _weights_ready(app_state):
            return await call_next(request)

        if path in health_paths:
            return JSONResponse(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
                content={
                    "status": "waiting_for_initial_weights",
                    "ready": False,
                    "checkpoint_path": app_state.current_checkpoint_path,
                },
            )

        if path in public_paths or path.startswith("/docs"):
            return await call_next(request)

        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            content={
                "error": {
                    "message": (
                        "Model weights are not loaded yet. "
                        "Call /v1/reload_weights and retry."
                    ),
                    "type": "service_unavailable",
                    "code": HTTPStatus.SERVICE_UNAVAILABLE.value,
                }
            },
        )


def build_minimal_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None,
            docs_url=None,
            redoc_url=None,
            lifespan=lifespan,
        )
    elif args.enable_offline_docs:
        app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)

    app.state.args = args
    app.root_path = args.root_path
    app.include_router(router)
    attach_metrics_router(app)
    attach_health_router(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)

    if tokens := [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]:
        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    if args.enable_request_id_headers:
        app.add_middleware(XRequestIdMiddleware)

    app.add_middleware(ScalingMiddleware)

    if envs.VLLM_DEBUG_LOG_API_SERVER_RESPONSE:
        logger.warning(
            "CAUTION: Enabling API response logging. "
            "This may include sensitive data and is not recommended in prod."
        )
        app.middleware("http")(log_response)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. Must be a function or a class."
            )

    _install_weight_gate(app)
    app = sagemaker_standards_bootstrap(app)
    return app


async def run_server(args: Namespace, **uvicorn_kwargs) -> None:
    decorate_logs("OnlineRLServer")
    require_first_reload, checkpoint_path = _configure_initial_reload_mode(args)

    listen_address, sock = setup_server(args)

    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_engine_client(args) as client:
        supported_tasks = await client.get_supported_tasks()
        if "generate" not in supported_tasks:
            raise RuntimeError(
                f"This server needs a generate-capable model; got tasks={supported_tasks}."
            )

        app = build_minimal_app(args)
        await init_app_state(client, app.state, args, supported_tasks)

        app.state.require_first_reload = require_first_reload
        app.state.current_checkpoint_path = checkpoint_path
        app.state.first_weights_loaded = asyncio.Event()
        if not require_first_reload:
            app.state.first_weights_loaded.set()
        app.state.reload_lock = asyncio.Lock()

        logger.info(
            "Starting online RL TPU server on %s (require_first_reload=%s)",
            listen_address,
            require_first_reload,
        )

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=args.ssl_ciphers,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )

    try:
        await shutdown_task
    finally:
        sock.close()


if __name__ == "__main__":
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description=(
            "Minimal TPU online-RL server: /health, /metrics, /v1/models, "
            "/v1/completions (non-stream), /v1/reload_weights"
        )
    )
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--model-weights",
        dest="model_weights",
        type=str,
        default="",
        help=(
            "Optional model weights URI/path for TPU Orbax loading. "
            "Use with --model pointing to a local HF config/tokenizer directory."
        ),
    )
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
