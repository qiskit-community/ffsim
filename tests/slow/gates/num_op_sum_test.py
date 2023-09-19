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

import numpy as np
from pyscf.fci import cistring
from scipy.special import comb

import ffsim
from ffsim._ffsim import apply_num_op_sum_evolution_in_place
from ffsim.slow.gates.num_op_sum import apply_num_op_sum_evolution_in_place_slow


def test_apply_num_op_sum_evolution_in_place_slow():
    """Test applying num op sum evolution."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations = cistring.gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
        exponents = np.random.uniform(0, 2 * np.pi, size=norb)
        phases = np.exp(1j * exponents)
        vec_slow = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        apply_num_op_sum_evolution_in_place_slow(
            vec_slow, phases, occupations=occupations
        )
        apply_num_op_sum_evolution_in_place(
            vec_fast,
            phases,
            occupations=occupations,
        )
        np.testing.assert_allclose(vec_slow, vec_fast, atol=1e-8)
