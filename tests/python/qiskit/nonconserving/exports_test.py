# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the exports of ffsim.qiskit.nonconserving."""

from __future__ import annotations

import ffsim


def test_not_exported_from_ffsim_qiskit():
    """Test that nonconserving gates are only accessible from their own module."""
    assert ffsim.qiskit.nonconserving.__all__
    for name in ffsim.qiskit.nonconserving.__all__:
        assert getattr(ffsim.qiskit.nonconserving, name) is not None
        assert not hasattr(ffsim.qiskit, name)
        assert name not in ffsim.qiskit.__all__
