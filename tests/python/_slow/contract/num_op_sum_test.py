# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for gates."""

from __future__ import annotations

import math

import numpy as np

import ffsim
from ffsim import cistring
from ffsim._lib import contract_num_op_sum_spin_into_buffer
from ffsim._slow.contract.num_op_sum import contract_num_op_sum_spin_into_buffer_slow


def test_contract_num_op_sum_spin_into_buffer_slow():
    """Test applying num op sum evolution."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = math.comb(norb, n_alpha)
        dim_b = math.comb(norb, n_beta)
        occupations = cistring.gen_occslst(range(norb), n_alpha)
        coeffs = rng.uniform(size=norb)
        vec = ffsim.random.random_state_vector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        out_slow = np.zeros_like(vec)
        out_fast = np.zeros_like(vec)
        contract_num_op_sum_spin_into_buffer_slow(
            vec, coeffs, occupations=occupations, out=out_slow
        )
        contract_num_op_sum_spin_into_buffer(
            vec, coeffs, occupations=occupations, out=out_fast
        )
        np.testing.assert_allclose(out_slow, out_fast)
