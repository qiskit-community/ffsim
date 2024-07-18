# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for orbital rotation."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

import ffsim
from ffsim.states import slater_determinant


def _orbital_rotation_generator(mat: np.ndarray, spin: bool) -> ffsim.FermionOperator:
    norb, _ = mat.shape
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    cre, des = (ffsim.cre_b, ffsim.des_b) if spin else (ffsim.cre_a, ffsim.des_a)
    for i, j in itertools.product(range(norb), repeat=2):
        coeffs[cre(i), des(j)] = mat[i, j]
    return ffsim.FermionOperator(coeffs)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(4)))
def test_apply_orbital_rotation_one_body_linop(norb: int, nelec: tuple[int, int]):
    """Test applying orbital rotation is consistent one-body linear operator."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat = ffsim.random.random_unitary(norb, seed=rng)
        vec = ffsim.random.random_state_vector(dim, seed=rng)

        result = ffsim.apply_orbital_rotation(vec, mat, norb, nelec)
        if norb:
            op = ffsim.contract.one_body_linop(
                scipy.linalg.logm(mat), norb=norb, nelec=nelec
            )
            expected = scipy.sparse.linalg.expm_multiply(op, vec, traceA=0)
        else:
            expected = vec

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(4)))
def test_apply_orbital_rotation_random_spinless(norb: int, nocc: int):
    """Test applying random orbital rotation yields correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nocc)
    for _ in range(3):
        mat = ffsim.random.random_unitary(norb, seed=rng)
        vec = ffsim.random.random_state_vector(dim, seed=rng)
        if norb:
            gen = _orbital_rotation_generator(scipy.linalg.logm(mat), spin=False)
            op = ffsim.linear_operator(gen, norb=norb, nelec=(nocc, 0))
        expected = scipy.sparse.linalg.expm_multiply(op, vec, traceA=0) if norb else vec
        result = ffsim.apply_orbital_rotation(vec, mat, norb, nocc)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(4)))
def test_apply_orbital_rotation_random_spinful(norb: int, nelec: tuple[int, int]):
    """Test applying random orbital rotation yields correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat_a = ffsim.random.random_unitary(norb, seed=rng)
        mat_b = ffsim.random.random_unitary(norb, seed=rng)
        vec = ffsim.random.random_state_vector(dim, seed=rng)

        if norb:
            gen_a = _orbital_rotation_generator(scipy.linalg.logm(mat_a), spin=False)
            gen_b = _orbital_rotation_generator(scipy.linalg.logm(mat_b), spin=True)
            op_a = ffsim.linear_operator(gen_a, norb=norb, nelec=nelec)
            op_b = ffsim.linear_operator(gen_b, norb=norb, nelec=nelec)
            op_ab = ffsim.linear_operator(gen_a + gen_b, norb=norb, nelec=nelec)

        # (mat_a, mat_b)
        expected = (
            scipy.sparse.linalg.expm_multiply(op_ab, vec, traceA=0) if norb else vec
        )
        result = ffsim.apply_orbital_rotation(vec, (mat_a, mat_b), norb, nelec)
        np.testing.assert_allclose(result, expected)
        result = ffsim.apply_orbital_rotation(
            vec, np.stack((mat_a, mat_b)), norb, nelec
        )
        np.testing.assert_allclose(result, expected)

        # (mat_a, None)
        expected = (
            scipy.sparse.linalg.expm_multiply(op_a, vec, traceA=0) if norb else vec
        )
        result = ffsim.apply_orbital_rotation(vec, (mat_a, None), norb, nelec)
        np.testing.assert_allclose(result, expected)

        # (None, mat_b)
        expected = (
            scipy.sparse.linalg.expm_multiply(op_b, vec, traceA=0) if norb else vec
        )
        result = ffsim.apply_orbital_rotation(vec, (None, mat_b), norb, nelec)
        np.testing.assert_allclose(result, expected)

        # (None, None)
        expected = vec
        result = ffsim.apply_orbital_rotation(vec, (None, None), norb, nelec)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(4)))
def test_apply_orbital_rotation_no_side_effects_vec(norb: int, nelec: tuple[int, int]):
    """Test applying orbital basis change doesn't modify the original vector."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat = ffsim.random.random_unitary(norb, seed=rng)
        vec = ffsim.random.random_state_vector(dim, seed=rng)
        original_vec = vec.copy()
        _ = ffsim.apply_orbital_rotation(vec, mat, norb, nelec)
        np.testing.assert_allclose(vec, original_vec)


def test_apply_orbital_rotation_eigenstates():
    """Test applying orbital basis change prepares eigenstates of one-body tensor."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        nelec = ffsim.testing.random_nelec(norb, seed=rng)
        occupied_orbitals = ffsim.testing.random_occupied_orbitals(
            norb, nelec, seed=rng
        )
        occ_a, occ_b = occupied_orbitals
        one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
        eigs, vecs = scipy.linalg.eigh(one_body_tensor)
        eig = sum(eigs[occ_a]) + sum(eigs[occ_b])
        state = slater_determinant(norb, occupied_orbitals)
        original_state = state.copy()
        final_state = ffsim.apply_orbital_rotation(state, vecs, norb, nelec)
        np.testing.assert_allclose(np.linalg.norm(final_state), 1.0)
        result = ffsim.contract.contract_one_body(
            final_state, one_body_tensor, norb, nelec
        )
        expected = eig * final_state
        np.testing.assert_allclose(result, expected)
        # check that the state was not modified
        np.testing.assert_allclose(state, original_state)


def test_apply_orbital_rotation_compose():
    """Test composing orbital basis changes."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        basis_change_1 = ffsim.random.random_unitary(norb, seed=rng)
        basis_change_2 = ffsim.random.random_unitary(norb, seed=rng)

        nelec = ffsim.testing.random_nelec(norb, seed=rng)
        dim = ffsim.dim(norb, nelec)
        state = ffsim.random.random_state_vector(dim, seed=rng)

        result = ffsim.apply_orbital_rotation(state, basis_change_1, norb, nelec)
        result = ffsim.apply_orbital_rotation(
            result, basis_change_2 @ basis_change_1.T.conj(), norb, nelec
        )
        expected_state = ffsim.apply_orbital_rotation(
            state, basis_change_2, norb, nelec
        )

        np.testing.assert_allclose(result, expected_state)


def test_apply_orbital_rotation_special_case():
    """Test a special case that was found to cause issues."""
    datadir = Path(__file__).parent.parent / "test_data"
    filepath = datadir / "orbital_rotation-0.npy"

    with open(filepath, "rb") as f:
        mat = np.load(f)
    assert ffsim.linalg.is_unitary(mat, atol=1e-12)

    norb = 8
    nelec = 5, 5
    vec = ffsim.hartree_fock_state(norb, nelec)
    np.testing.assert_allclose(np.linalg.norm(vec), 1)

    result = ffsim.apply_orbital_rotation(vec, mat, norb, nelec)
    np.testing.assert_allclose(np.linalg.norm(result), 1)

    op = ffsim.contract.one_body_linop(scipy.linalg.logm(mat), norb=norb, nelec=nelec)
    expected = scipy.sparse.linalg.expm_multiply(op, vec, traceA=0)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_apply_orbital_rotation_no_side_effects_mat():
    """Test applying orbital rotation doesn't modify the original matrix."""
    norb = 8
    nelec = 5, 5
    vec = ffsim.hartree_fock_state(norb, nelec)

    rng = np.random.default_rng()
    for _ in range(5):
        mat = ffsim.random.random_unitary(norb, seed=rng)
        original_mat = mat.copy()
        _ = ffsim.apply_orbital_rotation(vec, mat, norb, nelec)

        assert ffsim.linalg.is_unitary(original_mat)
        assert ffsim.linalg.is_unitary(mat)
        np.testing.assert_allclose(mat, original_mat, atol=1e-12)


def test_apply_orbital_rotation_no_side_effects_special_case():
    """Test applying orbital rotation doesn't modify the original matrix."""
    datadir = Path(__file__).parent.parent / "test_data"
    filepath = datadir / "orbital_rotation-0.npy"

    with open(filepath, "rb") as f:
        mat = np.load(f)
    assert ffsim.linalg.is_unitary(mat, atol=1e-12)

    norb = 8
    nelec = 5, 5
    vec = ffsim.hartree_fock_state(norb, nelec)

    original_mat = mat.copy()
    _ = ffsim.apply_orbital_rotation(vec, mat, norb, nelec)

    assert ffsim.linalg.is_unitary(original_mat)
    assert ffsim.linalg.is_unitary(mat)
    np.testing.assert_allclose(mat, original_mat, atol=1e-12)
