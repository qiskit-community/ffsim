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

import ffsim
from ffsim._ffsim import (
    apply_diag_coulomb_evolution_in_place_num_rep,
    apply_diag_coulomb_evolution_in_place_z_rep,
    apply_givens_rotation_in_place,
    apply_num_op_sum_evolution_in_place,
    apply_single_column_transformation_in_place,
    contract_diag_coulomb_into_buffer_num_rep,
    contract_diag_coulomb_into_buffer_z_rep,
    contract_num_op_sum_spin_into_buffer,
    gen_orbital_rotation_index_in_place,
)
from ffsim.gates.orbital_rotation import (
    _zero_one_subspace_indices,
    gen_orbital_rotation_index,
)
from ffsim.slow import (
    apply_diag_coulomb_evolution_in_place_num_rep_numpy,
    apply_diag_coulomb_evolution_in_place_num_rep_slow,
    apply_diag_coulomb_evolution_in_place_z_rep_slow,
    apply_givens_rotation_in_place_slow,
    apply_num_op_sum_evolution_in_place_slow,
    apply_single_column_transformation_in_place_slow,
    contract_diag_coulomb_into_buffer_num_rep_slow,
    contract_diag_coulomb_into_buffer_z_rep_slow,
    contract_num_op_sum_spin_into_buffer_slow,
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
        vec_slow = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
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
        vec_slow = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
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


def test_apply_diag_coulomb_evolution_num_rep_slow():
    """Test applying time evolution of diagonal Coulomb operator."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations_a = cistring.gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
        occupations_b = cistring.gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )
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
        np.testing.assert_allclose(vec_slow, vec_fast, atol=1e-8)


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
        np.testing.assert_allclose(vec_slow, vec_fast, atol=1e-8)


def test_apply_diag_coulomb_evolution_num_rep_numpy():
    """Test applying time evolution of diag Coulomb operator, numpy implementation."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations_a = cistring.gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
        occupations_b = cistring.gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )
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
        np.testing.assert_allclose(vec_slow, vec_fast, atol=1e-8)


def test_contract_diag_coulomb_into_buffer_num_rep_slow():
    """Test contracting diag Coulomb operator."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        occupations_a = cistring.gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
        occupations_b = cistring.gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )
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
        np.testing.assert_allclose(out_slow, out_fast, atol=1e-8)


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
        np.testing.assert_allclose(out_slow, out_fast, atol=1e-8)


def test_contract_num_op_sum_spin_into_buffer_slow():
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
        coeffs = np.random.uniform(size=norb)
        vec = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
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
        np.testing.assert_allclose(out_slow, out_fast, atol=1e-8)
