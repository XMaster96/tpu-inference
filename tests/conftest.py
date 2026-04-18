from __future__ import annotations

from pathlib import Path


_DEFAULT_COLLECT_IGNORES = {
    Path("core/test_dp_scheduler.py"),
    Path("runner/test_tpu_runner_dp.py"),
}


def pytest_ignore_collect(collection_path, config) -> bool | None:
    if config.getoption("--online-rl-server", default=False):
        return None

    try:
        relative_path = Path(collection_path).resolve().relative_to(
            Path(__file__).resolve().parent)
    except ValueError:
        return None

    if relative_path in _DEFAULT_COLLECT_IGNORES:
        return True

    return None
