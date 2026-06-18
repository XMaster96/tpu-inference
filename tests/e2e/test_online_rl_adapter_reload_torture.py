# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Generator

import pytest

from tests.e2e import test_online_rl_reload_torture as reload_torture

pytestmark = pytest.mark.online_rl_server_related


@pytest.fixture(scope="module")
def online_rl_server_adapter_context(
) -> Generator[reload_torture._ServerContext, None, None]:
    yield from reload_torture._online_rl_server_context_for_config(
        reload_torture._build_adapter_config())


def test_torture_adapter_only_single_generation_with_multiple_reloads(
    online_rl_server_adapter_context: reload_torture._ServerContext, ) -> None:
    reload_torture._run_torture_single_generation_with_multiple_reloads(
        online_rl_server_adapter_context)


def test_torture_adapter_only_parallel_generations_with_reload_storm(
    online_rl_server_adapter_context: reload_torture._ServerContext, ) -> None:
    reload_torture._run_torture_parallel_generations_with_reload_storm(
        online_rl_server_adapter_context)


def test_torture_adapter_only_continuous_reload_soak(
    online_rl_server_adapter_context: reload_torture._ServerContext, ) -> None:
    reload_torture._run_torture_continuous_reload_soak(
        online_rl_server_adapter_context)
