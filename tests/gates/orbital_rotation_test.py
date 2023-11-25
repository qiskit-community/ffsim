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

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

import ffsim
from ffsim.states import slater_determinant


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


def test_apply_orbital_rotation_no_side_effects():
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
def test_apply_orbital_rotation_permutation(dtype: type, atol: float):
    """Test applying orbital basis change."""
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


def test_apply_orbital_rotation_eigenstates_permutation():
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
