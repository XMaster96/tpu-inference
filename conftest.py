from __future__ import annotations

from pathlib import Path


ONLINE_RL_SERVER_MARKER = "online_rl_server_related"
ONLINE_RL_SERVER_PATHS = frozenset({
    Path("scripts/vllm/integration/test_accuracy.py"),
    Path("tests/e2e/test_online_rl_production_harness.py"),
    Path("tests/e2e/test_online_rl_reload_torture.py"),
    Path("tests/entrypoints/test_online_rl_production_harness_helpers.py"),
    Path("tests/entrypoints/test_online_rl_server_completion.py"),
    Path("tests/entrypoints/test_online_rl_server_reload.py"),
    Path("tests/entrypoints/test_stacked_regex.py"),
    Path("tests/models/jax/utils/test_weight_utils.py"),
    Path("tests/runner/test_persistent_batch_manager.py"),
    Path("tests/runner/test_tpu_runner_dp.py"),
    Path("tests/test_vllm_runtime_patches.py"),
})

_REPO_ROOT = Path(__file__).resolve().parent


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--online-rl-server",
        action="store_true",
        default=False,
        help="Collect only tests related to the online RL server.",
    )


def _repo_relative_path(collection_path) -> Path | None:
    try:
        return Path(collection_path).resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return None


def _is_selected_directory(relative_path: Path) -> bool:
    return any(relative_path == path or relative_path in path.parents
               for path in ONLINE_RL_SERVER_PATHS)


def pytest_ignore_collect(collection_path, config) -> bool | None:
    if not config.getoption("--online-rl-server"):
        return None

    relative_path = _repo_relative_path(collection_path)
    if relative_path is None:
        return None

    collection_path = Path(collection_path)
    if collection_path.is_dir():
        return False if _is_selected_directory(relative_path) else True

    if relative_path.name == "conftest.py":
        return False if _is_selected_directory(relative_path.parent) else True

    if collection_path.suffix == ".py":
        return False if relative_path in ONLINE_RL_SERVER_PATHS else True

    return None


def pytest_collection_modifyitems(config, items) -> None:
    if not config.getoption("--online-rl-server"):
        return

    selected_items = []
    deselected_items = []
    for item in items:
        if item.get_closest_marker(ONLINE_RL_SERVER_MARKER):
            selected_items.append(item)
        else:
            deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items
