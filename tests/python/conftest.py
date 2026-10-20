# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pytest configuration."""

from __future__ import annotations

import importlib.util

import jax
import pytest

collect_ignore = []
if importlib.util.find_spec("qiskit") is None:
    collect_ignore.append("qiskit")


@pytest.fixture(autouse=True, scope="module")
def clear_jax_caches():
    """Free JAX compiled executables after each test module.

    Each compiled executable holds memory mappings that are not released until its
    cache entry is cleared. Without this, the mappings accumulate across the test
    session and can exceed the Linux vm.max_map_count limit, which aborts the process
    during compilation.
    """
    yield
    jax.clear_caches()
