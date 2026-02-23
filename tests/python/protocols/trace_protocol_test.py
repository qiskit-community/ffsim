# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for trace protocol."""

import numpy as np
import pytest

import ffsim


class OpWithTrace:
    """An operator that implements _trace_ only."""

    def __init__(self, trace_val: float):
        self.trace_val = trace_val

    def _trace_(self, norb: int, nelec: int | tuple[int, int]) -> float:
        return self.trace_val


class OpWithDiag:
    """An operator that implements _diag_ only (no _trace_)."""

    def __init__(self, diag_vals: np.ndarray):
        self.diag_vals = diag_vals

    def _diag_(self, norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
        return self.diag_vals


class OpWithBoth:
    """An operator that implements both _trace_ and _diag_."""

    def __init__(self, trace_val: float, diag_vals: np.ndarray):
        self.trace_val = trace_val
        self.diag_vals = diag_vals

    def _trace_(self, norb: int, nelec: int | tuple[int, int]) -> float:
        return self.trace_val

    def _diag_(self, norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
        return self.diag_vals


class OpWithNeither:
    """An operator that implements neither _trace_ nor _diag_."""


def test_trace_from_trace():
    """Test that trace uses _trace_ when available."""
    op = OpWithTrace(trace_val=5.0)
    result = ffsim.trace(op, norb=4, nelec=(2, 2))
    assert result == 5.0


def test_trace_from_diag():
    """Test that trace falls back to summing _diag_ when _trace_ is absent."""
    diag_vals = np.array([1.0, 2.0, 3.0, 4.0])
    op = OpWithDiag(diag_vals=diag_vals)
    result = ffsim.trace(op, norb=4, nelec=(2, 2))
    np.testing.assert_allclose(result, 10.0)


def test_trace_prefers_trace_over_diag():
    """Test that _trace_ is preferred over _diag_ when both are available."""
    diag_vals = np.array([1.0, 2.0, 3.0])
    op = OpWithBoth(trace_val=99.0, diag_vals=diag_vals)
    result = ffsim.trace(op, norb=4, nelec=(2, 2))
    assert result == 99.0


def test_trace_raises_for_unsupported():
    """Test that trace raises TypeError when neither method is available."""
    op = OpWithNeither()
    with pytest.raises(TypeError, match="Could not compute trace"):
        ffsim.trace(op, norb=4, nelec=(2, 2))
