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

import numpy as np
from scipy.special import comb

import ffsim
from ffsim import cistring
from ffsim._lib import (
    contract_diag_coulomb_into_buffer_num_rep,
    contract_diag_coulomb_into_buffer_z_rep,
)
from ffsim._slow.contract.diag_coulomb import (
    contract_diag_coulomb_into_buffer_num_rep_slow,
    contract_diag_coulomb_into_buffer_z_rep_slow,
)


def test_contract_diag_coulomb_into_buffer_num_rep_slow():
    """Test contracting diag Coulomb operator."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations_a = cistring.gen_occslst(range(norb), n_alpha)
        occupations_b = cistring.gen_occslst(range(norb), n_beta)
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_alpha_beta = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        vec = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        out_slow = np.zeros_like(vec)
        out_fast = np.zeros_like(vec)
        contract_diag_coulomb_into_buffer_num_rep_slow(
            vec,
            mat,
            norb=norb,
            mat_alpha_beta=mat_alpha_beta,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
            out=out_slow,
        )
        contract_diag_coulomb_into_buffer_num_rep(
            vec,
            mat,
            norb=norb,
            mat_alpha_beta=mat_alpha_beta,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
            out=out_fast,
        )
        np.testing.assert_allclose(out_slow, out_fast)


def test_contract_diag_coulomb_into_buffer_z_rep_slow():
    """Test contracting diag Coulomb operator."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        strings_a = cistring.make_strings(range(norb), n_alpha)
        strings_b = cistring.make_strings(range(norb), n_beta)
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_alpha_beta = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        vec = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        out_slow = np.zeros_like(vec)
        out_fast = np.zeros_like(vec)
        contract_diag_coulomb_into_buffer_z_rep_slow(
            vec,
            mat,
            norb=norb,
            mat_alpha_beta=mat_alpha_beta,
            strings_a=strings_a,
            strings_b=strings_b,
            out=out_slow,
        )
        contract_diag_coulomb_into_buffer_z_rep(
            vec,
            mat,
            norb=norb,
            mat_alpha_beta=mat_alpha_beta,
            strings_a=strings_a,
            strings_b=strings_b,
            out=out_fast,
        )
        np.testing.assert_allclose(out_slow, out_fast)
