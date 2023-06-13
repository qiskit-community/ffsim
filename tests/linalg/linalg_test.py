# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for linear algebra utilities."""

from __future__ import annotations

import numpy as np

from ffsim.linalg import (
    lup,
)


def test_lup():
    dim = 5
    rng = np.random.default_rng()
    mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    ell, u, p = lup(mat)
    np.testing.assert_allclose(ell @ u @ p, mat)
    np.testing.assert_allclose(np.diagonal(ell), np.ones(dim))
