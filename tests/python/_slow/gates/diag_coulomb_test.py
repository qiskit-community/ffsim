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
from scipy.special import comb

import ffsim
from ffsim import cistring
from ffsim._lib import (
    apply_diag_coulomb_evolution_in_place_num_rep,
    apply_diag_coulomb_evolution_in_place_z_rep,
)
from ffsim._slow.gates.diag_coulomb import (
    apply_diag_coulomb_evolution_in_place_num_rep_numpy,
    apply_diag_coulomb_evolution_in_place_num_rep_slow,
    apply_diag_coulomb_evolution_in_place_z_rep_slow,
)


def test_apply_diag_coulomb_evolution_num_rep_slow():
    """Test applying time evolution of diagonal Coulomb operator."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations_a = cistring.gen_occslst(range(norb), n_alpha)
        occupations_b = cistring.gen_occslst(range(norb), n_beta)
        time = 0.6
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_exp = np.exp(-1j * time * mat)
        mat_alpha_beta = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_alpha_beta_exp = np.exp(-1j * time * mat_alpha_beta)
        vec_slow = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        apply_diag_coulomb_evolution_in_place_num_rep_slow(
            vec_slow,
            mat_exp,
            norb=norb,
            mat_alpha_beta_exp=mat_alpha_beta_exp,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
        apply_diag_coulomb_evolution_in_place_num_rep(
            vec_fast,
            mat_exp,
            norb=norb,
            mat_alpha_beta_exp=mat_alpha_beta_exp,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
        np.testing.assert_allclose(vec_slow, vec_fast)


def test_apply_diag_coulomb_evolution_z_rep_slow():
    """Test applying time evolution of diagonal Coulomb operator."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        strings_a = cistring.make_strings(range(norb), n_alpha)
        strings_b = cistring.make_strings(range(norb), n_beta)
        time = 0.6
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_exp = np.exp(-1j * time * mat)
        mat_alpha_beta = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_alpha_beta_exp = np.exp(-1j * time * mat_alpha_beta)
        vec_slow = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        apply_diag_coulomb_evolution_in_place_z_rep_slow(
            vec_slow,
            mat_exp,
            mat_exp.conj(),
            norb=norb,
            mat_alpha_beta_exp=mat_alpha_beta_exp,
            mat_alpha_beta_exp_conj=mat_alpha_beta_exp.conj(),
            strings_a=strings_a,
            strings_b=strings_b,
        )
        apply_diag_coulomb_evolution_in_place_z_rep(
            vec_fast,
            mat_exp,
            mat_exp.conj(),
            norb=norb,
            mat_alpha_beta_exp=mat_alpha_beta_exp,
            mat_alpha_beta_exp_conj=mat_alpha_beta_exp.conj(),
            strings_a=strings_a,
            strings_b=strings_b,
        )
        np.testing.assert_allclose(vec_slow, vec_fast)


def test_apply_diag_coulomb_evolution_num_rep_numpy():
    """Test applying time evolution of diag Coulomb operator, numpy implementation."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations_a = cistring.gen_occslst(range(norb), n_alpha)
        occupations_b = cistring.gen_occslst(range(norb), n_beta)
        time = 0.6
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_exp = np.exp(-1j * time * mat)
        mat_alpha_beta = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        mat_alpha_beta_exp = np.exp(-1j * time * mat_alpha_beta)
        vec_slow = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        apply_diag_coulomb_evolution_in_place_num_rep_numpy(
            vec_slow,
            mat_exp,
            norb=norb,
            nelec=(n_alpha, n_beta),
            mat_alpha_beta_exp=mat_alpha_beta_exp,
        )
        apply_diag_coulomb_evolution_in_place_num_rep(
            vec_fast,
            mat_exp,
            norb=norb,
            mat_alpha_beta_exp=mat_alpha_beta_exp,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
        np.testing.assert_allclose(vec_slow, vec_fast)
