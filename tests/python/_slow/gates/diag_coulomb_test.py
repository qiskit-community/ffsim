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
import pytest

import ffsim
from ffsim import _cistring
from ffsim._lib import (
    apply_diag_coulomb_evolution_in_place_num_rep,
    apply_diag_coulomb_evolution_in_place_z_rep,
)
from ffsim._slow.gates.diag_coulomb import (
    apply_diag_coulomb_evolution_in_place_num_rep_numpy,
    apply_diag_coulomb_evolution_in_place_num_rep_slow,
    apply_diag_coulomb_evolution_in_place_z_rep_slow,
)

RNG = np.random.default_rng(114578808861875541540322785902551571980)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_apply_diag_coulomb_evolution_num_rep_slow(norb: int, nelec: tuple[int, int]):
    """Test applying time evolution of diagonal Coulomb operator."""
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    occupations_a = _cistring.gen_occslst(range(norb), n_alpha)
    occupations_b = _cistring.gen_occslst(range(norb), n_beta)
    for _ in range(5):
        time = RNG.uniform(-10, 10)
        mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
        mat_ab = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
        mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
        mat_exp_aa = np.exp(-1j * time * mat_aa)
        mat_exp_ab = np.exp(-1j * time * mat_ab)
        mat_exp_bb = np.exp(-1j * time * mat_bb)
        vec_slow = ffsim.random.random_state_vector(dim_a * dim_b, seed=RNG).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        apply_diag_coulomb_evolution_in_place_num_rep_slow(
            vec_slow,
            mat_exp_aa,
            mat_exp_ab,
            mat_exp_bb,
            norb=norb,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
        apply_diag_coulomb_evolution_in_place_num_rep(
            vec_fast,
            mat_exp_aa,
            mat_exp_ab,
            mat_exp_bb,
            norb=norb,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
        np.testing.assert_allclose(vec_slow, vec_fast)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_apply_diag_coulomb_evolution_z_rep_slow(norb: int, nelec: tuple[int, int]):
    """Test applying time evolution of diagonal Coulomb operator."""
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    strings_a = _cistring.make_strings(range(norb), n_alpha)
    strings_b = _cistring.make_strings(range(norb), n_beta)
    for _ in range(5):
        time = RNG.uniform(-10, 10)
        mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
        mat_ab = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
        mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
        mat_exp_aa = np.exp(-1j * time * mat_aa)
        mat_exp_ab = np.exp(-1j * time * mat_ab)
        mat_exp_bb = np.exp(-1j * time * mat_bb)
        vec_slow = ffsim.random.random_state_vector(dim_a * dim_b, seed=RNG).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        apply_diag_coulomb_evolution_in_place_z_rep_slow(
            vec_slow,
            mat_exp_aa,
            mat_exp_ab,
            mat_exp_bb,
            mat_exp_aa.conj(),
            mat_exp_ab.conj(),
            mat_exp_bb.conj(),
            norb=norb,
            strings_a=strings_a,
            strings_b=strings_b,
        )
        apply_diag_coulomb_evolution_in_place_z_rep(
            vec_fast,
            mat_exp_aa,
            mat_exp_ab,
            mat_exp_bb,
            mat_exp_aa.conj(),
            mat_exp_ab.conj(),
            mat_exp_bb.conj(),
            norb=norb,
            strings_a=strings_a,
            strings_b=strings_b,
        )
        np.testing.assert_allclose(vec_slow, vec_fast)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_apply_diag_coulomb_evolution_num_rep_numpy(norb: int, nelec: tuple[int, int]):
    """Test applying time evolution of diag Coulomb operator, numpy implementation."""
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    occupations_a = _cistring.gen_occslst(range(norb), n_alpha)
    occupations_b = _cistring.gen_occslst(range(norb), n_beta)
    for _ in range(5):
        time = RNG.uniform(-10, 10)
        mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
        mat_ab = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
        mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
        mat_exp_aa = np.exp(-1j * time * mat_aa)
        mat_exp_ab = np.exp(-1j * time * mat_ab)
        mat_exp_bb = np.exp(-1j * time * mat_bb)
        vec_slow = ffsim.random.random_state_vector(dim_a * dim_b, seed=RNG).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        apply_diag_coulomb_evolution_in_place_num_rep_numpy(
            vec_slow,
            mat_exp_aa,
            mat_exp_ab,
            mat_exp_bb,
            norb=norb,
            nelec=(n_alpha, n_beta),
        )
        apply_diag_coulomb_evolution_in_place_num_rep(
            vec_fast,
            mat_exp_aa,
            mat_exp_ab,
            mat_exp_bb,
            norb=norb,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
        np.testing.assert_allclose(vec_slow, vec_fast)
