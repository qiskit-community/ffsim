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
from pyscf.fci.fci_slow import contract_1e

from ffsim.fci import get_dimension, one_body_tensor_to_linop
from ffsim.gates import apply_orbital_rotation
from ffsim.linalg import expm_multiply_taylor
from ffsim.random_utils import random_hermitian, random_statevector, random_unitary
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
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        nelec = (n_alpha, n_beta)
        dim = get_dimension(norb, nelec)

        mat = random_unitary(norb, seed=rng, dtype=dtype)
        vec = random_statevector(dim, seed=rng, dtype=dtype)
        original_vec = vec.copy()

        result = apply_orbital_rotation(vec, mat, norb, nelec)
        op = one_body_tensor_to_linop(scipy.linalg.logm(mat), norb=norb, nelec=nelec)
        expected = expm_multiply_taylor(original_vec, op)
        np.testing.assert_allclose(result, expected, atol=atol)


def test_apply_orbital_rotation_no_side_effects():
    """Test applying orbital basis change doesn't modify the original vector."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        nelec = (n_alpha, n_beta)
        dim = get_dimension(norb, nelec)

        mat = -np.eye(norb)
        vec = random_statevector(dim, seed=rng)
        original_vec = vec.copy()

        _ = apply_orbital_rotation(vec, mat, norb, nelec)
        np.testing.assert_allclose(vec, original_vec, atol=1e-12)


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
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        nelec = (n_alpha, n_beta)
        dim = get_dimension(norb, nelec)

        mat = random_unitary(norb, seed=rng, dtype=dtype)
        vec = random_statevector(dim, seed=rng, dtype=dtype)
        original_vec = vec.copy()

        result, perm = apply_orbital_rotation(
            vec, mat, norb, nelec, allow_col_permutation=True, copy=True
        )
        np.testing.assert_allclose(np.linalg.norm(result), 1, atol=atol)
        op = one_body_tensor_to_linop(
            scipy.linalg.logm(mat @ perm), norb=norb, nelec=nelec
        )
        expected = expm_multiply_taylor(original_vec, op)
        np.testing.assert_allclose(result, expected, atol=atol)

        result, perm = apply_orbital_rotation(
            vec, mat, norb, nelec, allow_row_permutation=True, copy=False
        )
        op = one_body_tensor_to_linop(
            scipy.linalg.logm(perm @ mat), norb=norb, nelec=nelec
        )
        expected = expm_multiply_taylor(original_vec, op)
        np.testing.assert_allclose(result, expected, atol=atol)


def test_apply_orbital_rotation_eigenstates():
    """Test applying orbital basis change prepares eigenstates of one-body tensor."""
    norb = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, norb + 1)
    n_beta = rng.integers(1, norb + 1)
    occupied_orbitals = (
        rng.choice(norb, n_alpha, replace=False),
        rng.choice(norb, n_beta, replace=False),
    )

    one_body_tensor = np.array(random_hermitian(norb, seed=rng))
    eigs, vecs = np.linalg.eigh(one_body_tensor)
    eig = sum(np.sum(eigs[orbs]) for orbs in occupied_orbitals)
    nelec = tuple(len(orbs) for orbs in occupied_orbitals)
    state = slater_determinant(norb, occupied_orbitals)
    original_state = state.copy()
    final_state = apply_orbital_rotation(state, vecs, norb, nelec)
    np.testing.assert_allclose(np.linalg.norm(final_state), 1.0, atol=1e-8)
    result = contract_1e(one_body_tensor, final_state, norb, nelec)
    expected = eig * final_state
    np.testing.assert_allclose(result, expected, atol=1e-8)
    # check that the state was not modified
    np.testing.assert_allclose(state, original_state)


def test_apply_orbital_rotation_eigenstates_permutation():
    """Test applying orbital basis change prepares eigenstates of one-body tensor."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        occupied_orbitals = (
            rng.choice(norb, n_alpha, replace=False),
            rng.choice(norb, n_beta, replace=False),
        )

        one_body_tensor = np.array(random_hermitian(norb, seed=rng))
        eigs, vecs = np.linalg.eigh(one_body_tensor)
        nelec = tuple(len(orbs) for orbs in occupied_orbitals)
        state = slater_determinant(norb, occupied_orbitals)
        original_state = state.copy()
        final_state, perm = apply_orbital_rotation(
            state, vecs, norb, nelec, allow_col_permutation=True
        )
        eigs = eigs @ perm
        eig = sum(np.sum(eigs[orbs]) for orbs in occupied_orbitals)
        np.testing.assert_allclose(np.linalg.norm(final_state), 1.0, atol=1e-8)
        result = contract_1e(one_body_tensor, final_state, norb, nelec)
        expected = eig * final_state
        np.testing.assert_allclose(result, expected, atol=1e-8)
        # check that the state was not modified
        np.testing.assert_allclose(state, original_state)


def test_apply_orbital_rotation_compose():
    """Test composing orbital basis changes."""
    norb = 5
    rng = np.random.default_rng()
    basis_change_1 = np.array(random_unitary(norb, seed=rng))
    basis_change_2 = np.array(random_unitary(norb, seed=rng))

    n_alpha = rng.integers(1, norb + 1)
    n_beta = rng.integers(1, norb + 1)
    nelec = (n_alpha, n_beta)
    dim = get_dimension(norb, nelec)
    state = np.array(random_statevector(dim, seed=rng))

    result = apply_orbital_rotation(state, basis_change_1, norb, nelec)
    result = apply_orbital_rotation(
        result, basis_change_2 @ basis_change_1.T.conj(), norb, nelec
    )
    expected_state = apply_orbital_rotation(state, basis_change_2, norb, nelec)

    np.testing.assert_allclose(result, expected_state, atol=1e-8)
