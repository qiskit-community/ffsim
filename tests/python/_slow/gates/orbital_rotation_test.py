# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import math

import numpy as np

import ffsim
from ffsim._lib import (
    apply_givens_rotation_in_place,
)
from ffsim._slow.gates.orbital_rotation import apply_givens_rotation_in_place_slow
from ffsim.gates.orbital_rotation import _zero_one_subspace_indices


def test_apply_givens_rotation_in_place_slow():
    """Test applying Givens rotation."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = math.comb(norb, n_alpha)
        dim_b = math.comb(norb, n_beta)
        vec_slow = ffsim.random.random_state_vector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        c = rng.uniform(0, 1)
        s = (1j) ** rng.uniform(0, 4) * np.sqrt(1 - c**2)
        indices = _zero_one_subspace_indices(norb, n_alpha, (1, 3))
        slice1 = indices[: len(indices) // 2]
        slice2 = indices[len(indices) // 2 :]
        apply_givens_rotation_in_place_slow(vec_slow, c, s, slice1, slice2)
        apply_givens_rotation_in_place(vec_fast, c, s, slice1, slice2)
        np.testing.assert_allclose(vec_slow, vec_fast)
