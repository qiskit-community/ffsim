# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for lazy import of ffsim.qiskit."""

from __future__ import annotations

import ffsim


def test_lazy_import():
    """Test that ffsim.qiskit is accessible as an attribute of ffsim."""
    assert ffsim.qiskit.final_state_vector is not None
