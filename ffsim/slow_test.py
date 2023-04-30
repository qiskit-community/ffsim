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
from pyscf.fci import cistring
from scipy.special import comb

from ffsim._ffsim import (
    apply_diag_coulomb_evolution_in_place,
    apply_givens_rotation_in_place,
    apply_num_op_sum_evolution_in_place,
    apply_single_column_transformation_in_place,
    contract_diag_coulomb_into_buffer,
    gen_orbital_rotation_index_in_place,
)
from ffsim.fci import gen_orbital_rotation_index
from ffsim.gates import _zero_one_subspace_indices
from ffsim.random_utils import random_hermitian, random_statevector, random_unitary
from ffsim.slow import (
    apply_diag_coulomb_evolution_in_place_numpy,
    apply_diag_coulomb_evolution_in_place_slow,
    apply_givens_rotation_in_place_slow,
    apply_num_op_sum_evolution_in_place_slow,
    apply_single_column_transformation_in_place_slow,
    contract_diag_coulomb_into_buffer_slow,
    gen_orbital_rotation_index_in_place_slow,
)


def test_apply_givens_rotation_in_place_slow():
    """Test applying Givens rotation."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        mat = random_unitary(norb, seed=rng)
        vec_slow = random_statevector(dim_a * dim_b, seed=rng).reshape((dim_a, dim_b))
        vec_fast = vec_slow.copy()
        c = rng.uniform()
        s = 1 - c**2
        phase = (1j) ** rng.uniform(0, 4)
        indices = _zero_one_subspace_indices(norb, n_alpha, (1, 3))
        slice1 = indices[: len(indices) // 2]
        slice2 = indices[len(indices) // 2 :]
        apply_givens_rotation_in_place_slow(vec_slow, c, s, phase, slice1, slice2)
        apply_givens_rotation_in_place(vec_fast, c, s, phase, slice1, slice2)
        np.testing.assert_allclose(vec_slow, vec_fast, atol=1e-8)


def test_gen_orbital_rotation_index_in_place_slow():
    """Test generating orbital rotation index."""
    norb = 5
    rng = np.random.default_rng()
    nocc = rng.integers(1, norb + 1)
    linkstr_index = cistring.gen_linkstr_index(range(norb), nocc)

    dim_diag = comb(norb - 1, nocc - 1, exact=True)
    dim_off_diag = comb(norb - 1, nocc, exact=True)
    dim = dim_diag + dim_off_diag

    diag_strings_slow = np.empty((norb, dim_diag), dtype=np.uint)
    off_diag_strings_slow = np.empty((norb, dim_off_diag), dtype=np.uint)
    off_diag_index_slow = np.empty((norb, dim_off_diag, nocc, 3), dtype=np.int32)
    off_diag_strings_index_slow = np.empty((norb, dim), dtype=np.uint)

    diag_strings_fast = np.empty((norb, dim_diag), dtype=np.uint)
    off_diag_strings_fast = np.empty((norb, dim_off_diag), dtype=np.uint)
    off_diag_index_fast = np.empty((norb, dim_off_diag, nocc, 3), dtype=np.int32)
    off_diag_strings_index_fast = np.empty((norb, dim), dtype=np.uint)

    gen_orbital_rotation_index_in_place_slow(
        norb=norb,
        nocc=nocc,
        linkstr_index=linkstr_index,
        diag_strings=diag_strings_slow,
        off_diag_strings=off_diag_strings_slow,
        off_diag_strings_index=off_diag_strings_index_slow,
        off_diag_index=off_diag_index_slow,
    )
    gen_orbital_rotation_index_in_place(
        norb=norb,
        nocc=nocc,
        linkstr_index=linkstr_index,
        diag_strings=diag_strings_fast,
        off_diag_strings=off_diag_strings_fast,
        off_diag_strings_index=off_diag_strings_index_fast,
        off_diag_index=off_diag_index_fast,
    )
    np.testing.assert_array_equal(diag_strings_slow, diag_strings_fast)
    np.testing.assert_array_equal(off_diag_strings_slow, off_diag_strings_fast)
    np.testing.assert_array_equal(off_diag_index_slow, off_diag_index_fast)


def test_apply_single_column_transformation_in_place_slow():
    """Test applying single column transformation."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        orbital_rotation_index = gen_orbital_rotation_index(norb, n_alpha)
        column = rng.uniform(size=norb) + 1j * rng.uniform(size=norb)
        diag_val = rng.uniform() + 1j * rng.uniform()
        vec_slow = random_statevector(dim_a * dim_b, seed=rng).reshape((dim_a, dim_b))
        vec_fast = vec_slow.copy()
        index = [a[0] for a in orbital_rotation_index]
        apply_single_column_transformation_in_place_slow(
            vec_slow, column, diag_val, *index
        )
        apply_single_column_transformation_in_place(vec_fast, column, diag_val, *index)
        np.testing.assert_allclose(vec_slow, vec_fast, atol=1e-8)


def test_apply_num_op_sum_evolution_in_place_slow():
    """Test applying num op sum evolution."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations = cistring._gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
        exponents = np.random.uniform(0, 2 * np.pi, size=norb)
        phases = np.exp(1j * exponents)
        vec_slow = random_statevector(dim_a * dim_b, seed=rng).reshape((dim_a, dim_b))
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


def test_apply_diag_coulomb_evolution_slow():
    """Test applying time evolution of diagonal Coulomb operator."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
        occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )
        time = 0.6
        mat = np.real(random_hermitian(norb, seed=rng))
        mat_exp = np.exp(-1j * time * mat)
        mat_alpha_beta = np.real(random_hermitian(norb, seed=rng))
        mat_alpha_beta_exp = np.exp(-1j * time * mat_alpha_beta)
        vec_slow = random_statevector(dim_a * dim_b, seed=rng).reshape((dim_a, dim_b))
        vec_fast = vec_slow.copy()
        apply_diag_coulomb_evolution_in_place_slow(
            vec_slow,
            mat_exp,
            norb=norb,
            mat_alpha_beta_exp=mat_alpha_beta_exp,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
        apply_diag_coulomb_evolution_in_place(
            vec_fast,
            mat_exp,
            norb=norb,
            mat_alpha_beta_exp=mat_alpha_beta_exp,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
        np.testing.assert_allclose(vec_slow, vec_fast, atol=1e-8)


def test_apply_diag_coulomb_evolution_numpy():
    """Test applying time evolution of diag Coulomb operator, numpy implementation."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
        occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )
        time = 0.6
        mat = np.real(random_hermitian(norb, seed=rng))
        mat_exp = np.exp(-1j * time * mat)
        mat_alpha_beta = np.real(random_hermitian(norb, seed=rng))
        mat_alpha_beta_exp = np.exp(-1j * time * mat_alpha_beta)
        vec_slow = random_statevector(dim_a * dim_b, seed=rng).reshape((dim_a, dim_b))
        vec_fast = vec_slow.copy()
        apply_diag_coulomb_evolution_in_place_numpy(
            vec_slow,
            mat_exp,
            norb=norb,
            nelec=(n_alpha, n_beta),
            mat_alpha_beta_exp=mat_alpha_beta_exp,
        )
        apply_diag_coulomb_evolution_in_place(
            vec_fast,
            mat_exp,
            norb=norb,
            mat_alpha_beta_exp=mat_alpha_beta_exp,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
        np.testing.assert_allclose(vec_slow, vec_fast, atol=1e-8)


def test_contract_diag_coulomb_into_buffer_slow():
    """Test contracting diag Coulomb operator."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
        occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )
        time = 0.6
        mat = np.real(random_hermitian(norb, seed=rng))
        mat_alpha_beta = np.real(random_hermitian(norb, seed=rng))
        vec = random_statevector(dim_a * dim_b, seed=rng).reshape((dim_a, dim_b))
        out_slow = np.zeros_like(vec)
        out_fast = np.zeros_like(vec)
        contract_diag_coulomb_into_buffer_slow(
            vec,
            mat,
            norb=norb,
            mat_alpha_beta=mat_alpha_beta,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
            out=out_slow,
        )
        contract_diag_coulomb_into_buffer(
            vec,
            mat,
            norb=norb,
            mat_alpha_beta=mat_alpha_beta,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
            out=out_fast,
        )
        np.testing.assert_allclose(out_slow, out_fast, atol=1e-8)
