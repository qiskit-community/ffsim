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


def _orbital_rotation_generator(
    mat: np.ndarray, spin: ffsim.Spin
) -> ffsim.FermionOperator:
    norb, _ = mat.shape
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for i, j in itertools.product(range(norb), repeat=2):
        if spin & ffsim.Spin.ALPHA:
            coeffs[ffsim.cre_a(i), ffsim.des_a(j)] = mat[i, j]
        if spin & ffsim.Spin.BETA:
            coeffs[ffsim.cre_b(i), ffsim.des_b(j)] = mat[i, j]
    return ffsim.FermionOperator(coeffs)


@pytest.mark.parametrize(
    "dtype, atol",
    [
        (np.complex128, 1e-12),
    ],
)
def test_apply_orbital_rotation(dtype: type, atol: float):
    """Test applying orbital basis change."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        nelec = ffsim.testing.random_nelec(norb, seed=rng)
        dim = ffsim.dim(norb, nelec)

        mat = ffsim.random.random_unitary(norb, seed=rng, dtype=dtype)
        vec = ffsim.random.random_statevector(dim, seed=rng, dtype=dtype)
        original_vec = vec.copy()

        result = ffsim.apply_orbital_rotation(vec, mat, norb, nelec)
        op = ffsim.contract.one_body_linop(
            scipy.linalg.logm(mat), norb=norb, nelec=nelec
        )
        expected = scipy.sparse.linalg.expm_multiply(op, original_vec, traceA=1)
        np.testing.assert_allclose(result, expected, atol=atol)


@pytest.mark.parametrize(
    "dtype, atol",
    [
        (np.complex128, 1e-12),
    ],
)
def test_apply_orbital_rotation_spin(dtype: type, atol: float):
    """Test applying orbital basis change to different spins."""
    norb = 4
    rng = np.random.default_rng()
    for _ in range(3):
        for spin in ffsim.Spin:
            nelec = ffsim.testing.random_nelec(norb, seed=rng)
            dim = ffsim.dim(norb, nelec)

            mat = ffsim.random.random_unitary(norb, seed=rng, dtype=dtype)
            vec = ffsim.random.random_statevector(dim, seed=rng, dtype=dtype)
            original_vec = vec.copy()

            result = ffsim.apply_orbital_rotation(vec, mat, norb, nelec, spin=spin)
            generator = _orbital_rotation_generator(scipy.linalg.logm(mat), spin)
            op = ffsim.linear_operator(generator, norb=norb, nelec=nelec)
            expected = scipy.sparse.linalg.expm_multiply(op, original_vec, traceA=1)
            np.testing.assert_allclose(result, expected, atol=atol)


def test_apply_orbital_rotation_no_side_effects_vec():
    """Test applying orbital basis change doesn't modify the original vector."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        nelec = ffsim.testing.random_nelec(norb, seed=rng)
        dim = ffsim.dim(norb, nelec)

        mat = -np.eye(norb)
        vec = ffsim.random.random_statevector(dim, seed=rng)
        original_vec = vec.copy()

        _ = ffsim.apply_orbital_rotation(vec, mat, norb, nelec)
        np.testing.assert_allclose(vec, original_vec)


@pytest.mark.parametrize(
    "dtype, atol",
    [
        (np.complex128, 1e-12),
    ],
)
def test_apply_orbital_rotation_lu(dtype: type, atol: float):
    """Test applying orbital basis change, LU decomposition method."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        nelec = ffsim.testing.random_nelec(norb, seed=rng)
        dim = ffsim.dim(norb, nelec)

        mat = ffsim.random.random_unitary(norb, seed=rng, dtype=dtype)
        vec = ffsim.random.random_statevector(dim, seed=rng, dtype=dtype)
        original_vec = vec.copy()

        result, perm = ffsim.apply_orbital_rotation(
            vec, mat, norb, nelec, allow_col_permutation=True, copy=True
        )
        np.testing.assert_allclose(np.linalg.norm(result), 1, atol=atol)
        op = ffsim.contract.one_body_linop(
            scipy.linalg.logm(mat @ perm), norb=norb, nelec=nelec
        )
        expected = scipy.sparse.linalg.expm_multiply(op, original_vec, traceA=1)
        np.testing.assert_allclose(result, expected, atol=atol)

        result, perm = ffsim.apply_orbital_rotation(
            vec, mat, norb, nelec, allow_row_permutation=True, copy=False
        )
        op = ffsim.contract.one_body_linop(
            scipy.linalg.logm(perm @ mat), norb=norb, nelec=nelec
        )
        expected = scipy.sparse.linalg.expm_multiply(op, original_vec, traceA=1)
        np.testing.assert_allclose(result, expected, atol=atol)


@pytest.mark.parametrize(
    "dtype, atol",
    [
        (np.complex128, 1e-12),
    ],
)
def test_apply_orbital_rotation_spin_lu(dtype: type, atol: float):
    """Test applying orbital basis, LU decomposition method to different spins."""
    norb = 4
    rng = np.random.default_rng()
    for _ in range(3):
        for spin in ffsim.Spin:
            nelec = ffsim.testing.random_nelec(norb, seed=rng)
            dim = ffsim.dim(norb, nelec)

            mat = ffsim.random.random_unitary(norb, seed=rng, dtype=dtype)
            vec = ffsim.random.random_statevector(dim, seed=rng, dtype=dtype)
            original_vec = vec.copy()

            result, perm = ffsim.apply_orbital_rotation(
                vec, mat, norb, nelec, spin, allow_col_permutation=True, copy=True
            )
            np.testing.assert_allclose(np.linalg.norm(result), 1, atol=atol)
            generator = _orbital_rotation_generator(
                scipy.linalg.logm(mat @ perm), spin=spin
            )
            op = ffsim.linear_operator(generator, norb=norb, nelec=nelec)
            expected = scipy.sparse.linalg.expm_multiply(op, original_vec, traceA=1)
            np.testing.assert_allclose(result, expected, atol=atol)

            result, perm = ffsim.apply_orbital_rotation(
                vec, mat, norb, nelec, spin, allow_row_permutation=True, copy=False
            )
            generator = _orbital_rotation_generator(
                scipy.linalg.logm(perm @ mat), spin=spin
            )
            op = ffsim.linear_operator(generator, norb=norb, nelec=nelec)
            expected = scipy.sparse.linalg.expm_multiply(op, original_vec, traceA=1)
            np.testing.assert_allclose(result, expected, atol=atol)


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
        eigs, vecs = np.linalg.eigh(one_body_tensor)
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


def test_apply_orbital_rotation_eigenstates_lu():
    """Test applying LU orbital basis change prepares eigenstates of one-body tensor."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        nelec = ffsim.testing.random_nelec(norb, seed=rng)
        occupied_orbitals = ffsim.testing.random_occupied_orbitals(
            norb, nelec, seed=rng
        )
        occ_a, occ_b = occupied_orbitals
        one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
        eigs, vecs = np.linalg.eigh(one_body_tensor)
        state = slater_determinant(norb, occupied_orbitals)
        original_state = state.copy()
        final_state, perm = ffsim.apply_orbital_rotation(
            state, vecs, norb, nelec, allow_col_permutation=True
        )
        eigs = eigs @ perm
        eig = sum(eigs[occ_a]) + sum(eigs[occ_b])
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
        state = ffsim.random.random_statevector(dim, seed=rng)

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
    expected = scipy.sparse.linalg.expm_multiply(op, vec, traceA=1)
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
