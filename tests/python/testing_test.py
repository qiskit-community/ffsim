# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for testing utilities."""

from __future__ import annotations

import numpy as np
import pytest

import ffsim


def test_assert_allclose_up_to_global_phase():
    rng = np.random.default_rng()
    shape = (2, 3, 4)
    a = rng.standard_normal(shape).astype(complex)
    a += 1j + rng.standard_normal(shape)
    b = a * (1j) ** 1.5

    ffsim.testing.assert_allclose_up_to_global_phase(a, b)

    b[0, 0, 0] *= 1.1
    with pytest.raises(AssertionError):
        ffsim.testing.assert_allclose_up_to_global_phase(a, b)
