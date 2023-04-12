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

import itertools

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

from ffsim.fci import contract_1e, get_dimension, one_body_tensor_to_linop
from ffsim.gates import (
    apply_diag_coulomb_evolution,
    apply_givens_rotation,
    apply_num_interaction,
    apply_num_op_prod_interaction,
    apply_num_op_sum_evolution,
    apply_orbital_rotation,
    apply_tunneling_interaction,
)
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

        result = apply_orbital_rotation(mat, vec, norb, nelec)
        op = one_body_tensor_to_linop(scipy.linalg.logm(mat), nelec=nelec)
        expected = expm_multiply_taylor(op, original_vec)
        np.testing.assert_allclose(result, expected, atol=atol)


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
            mat, vec, norb, nelec, allow_col_permutation=True, copy=True
        )
        np.testing.assert_allclose(np.linalg.norm(result), 1, atol=atol)
        op = one_body_tensor_to_linop(scipy.linalg.logm(mat @ perm), nelec=nelec)
        expected = expm_multiply_taylor(op, original_vec)
        np.testing.assert_allclose(result, expected, atol=atol)

        result, perm = apply_orbital_rotation(
            mat, vec, norb, nelec, allow_row_permutation=True, copy=False
        )
        op = one_body_tensor_to_linop(scipy.linalg.logm(perm @ mat), nelec=nelec)
        expected = expm_multiply_taylor(op, original_vec)
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
    final_state = apply_orbital_rotation(vecs, state, norb, nelec)
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
            vecs, state, norb, nelec, allow_col_permutation=True
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

    result = apply_orbital_rotation(basis_change_1, state, norb, nelec)
    result = apply_orbital_rotation(
        basis_change_2 @ basis_change_1.T.conj(), result, norb, nelec
    )
    expected_state = apply_orbital_rotation(basis_change_2, state, norb, nelec)

    np.testing.assert_allclose(result, expected_state, atol=1e-8)


def test_apply_diag_coulomb_evolution():
    """Test applying time evolution of diagonal Coulomb operator."""
    norb = 5
    for _ in range(5):
        rng = np.random.default_rng()
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        occupied_orbitals = (
            rng.choice(norb, n_alpha, replace=False),
            rng.choice(norb, n_beta, replace=False),
        )
        nelec = tuple(len(orbs) for orbs in occupied_orbitals)
        state = slater_determinant(norb, occupied_orbitals)
        original_state = state.copy()

        core_tensor = np.real(np.array(random_hermitian(norb, seed=rng)))
        time = 0.6
        result = apply_diag_coulomb_evolution(core_tensor, state, time, norb, nelec)

        eig = 0
        for i, j in itertools.product(range(norb), repeat=2):
            for sigma, tau in itertools.product(range(2), repeat=2):
                if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                    eig += 0.5 * core_tensor[i, j]
        expected = np.exp(-1j * eig * time) * state

        np.testing.assert_allclose(result, expected, atol=1e-8)
        np.testing.assert_allclose(state, original_state)


def test_apply_num_op_sum_evolution():
    """Test applying time evolution of sum of number operators."""
    norb = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, norb + 1)
    n_beta = rng.integers(1, norb + 1)
    occupied_orbitals = (
        rng.choice(norb, n_alpha, replace=False),
        rng.choice(norb, n_beta, replace=False),
    )
    nelec = tuple(len(orbs) for orbs in occupied_orbitals)
    state = slater_determinant(norb, occupied_orbitals)
    original_state = state.copy()

    coeffs = rng.standard_normal(norb)
    time = 0.6
    result = apply_num_op_sum_evolution(coeffs, state, time, norb, nelec)

    eig = 0
    for i in range(norb):
        for sigma in range(2):
            if i in occupied_orbitals[sigma]:
                eig += coeffs[i]
    expected = np.exp(-1j * eig * time) * state

    np.testing.assert_allclose(result, expected, atol=1e-8)
    np.testing.assert_allclose(state, original_state)


def test_apply_quadratic_hamiltonian_evolution():
    """Test applying time evolution of a quadratic Hamiltonian."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        nelec = (n_alpha, n_beta)
        dim = get_dimension(norb, nelec)

        mat = random_hermitian(norb, seed=rng)
        eigs, vecs = np.linalg.eigh(mat)
        vec = random_statevector(dim, seed=rng)

        time = 0.6
        result = apply_num_op_sum_evolution(
            eigs, vec, time, norb, nelec, orbital_rotation=vecs
        )
        op = one_body_tensor_to_linop(mat, nelec=nelec)
        expected = expm_multiply_taylor(-1j * time * op, vec)
        np.testing.assert_allclose(result, expected, atol=1e-8)


def test_apply_givens_rotation():
    """Test applying Givens rotation."""
    norb = 5
    n_alpha = 3
    n_beta = 2
    nelec = (n_alpha, n_beta)

    dim = get_dimension(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(random_statevector(dim, seed=rng))
    original_vec = vec.copy()
    theta = rng.standard_normal()
    for i, j in itertools.product(range(norb), repeat=2):
        if i == j:
            continue
        result = apply_givens_rotation(
            theta,
            vec,
            (i, j),
            norb=norb,
            nelec=nelec,
        )
        generator = np.zeros((norb, norb))
        generator[i, j] = theta
        generator[j, i] = -theta
        linop = one_body_tensor_to_linop(generator, nelec=nelec)
        expected = expm_multiply_taylor(linop, vec)
        np.testing.assert_allclose(result, expected, atol=1e-8)
    np.testing.assert_allclose(vec, original_vec)


def test_apply_tunneling_interaction():
    """Test applying tunneling interaction."""
    norb = 5
    n_alpha = 3
    n_beta = 2
    nelec = (n_alpha, n_beta)

    dim = get_dimension(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(random_statevector(dim, seed=rng))
    theta = rng.standard_normal()
    for i, j in itertools.product(range(norb), repeat=2):
        if i == j:
            continue
        result = apply_tunneling_interaction(
            theta,
            vec,
            (i, j),
            norb=norb,
            nelec=nelec,
        )
        generator = np.zeros((norb, norb))
        generator[i, j] = theta
        generator[j, i] = theta
        linop = one_body_tensor_to_linop(generator, nelec=nelec)
        expected = expm_multiply_taylor(1j * linop, vec)
        np.testing.assert_allclose(result, expected, atol=1e-8)


def test_apply_num_interaction():
    """Test applying number interaction."""
    norb = 5
    n_alpha = 3
    n_beta = 2
    nelec = (n_alpha, n_beta)

    dim = get_dimension(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(random_statevector(dim, seed=rng))
    theta = rng.standard_normal()
    for target_orb in range(norb):
        result = apply_num_interaction(theta, vec, target_orb, norb=norb, nelec=nelec)
        generator = np.zeros((norb, norb))
        generator[target_orb, target_orb] = theta
        linop = one_body_tensor_to_linop(generator, nelec=nelec)
        expected = expm_multiply_taylor(
            1j * linop,
            vec,
        )
        np.testing.assert_allclose(result, expected, atol=1e-8)


def test_apply_num_num_interaction():
    """Test applying number-number interaction."""
    norb = 5

    rng = np.random.default_rng()
    n_alpha = 3
    n_beta = 2
    occupied_orbitals = (
        rng.choice(norb, n_alpha, replace=False),
        rng.choice(norb, n_beta, replace=False),
    )
    nelec = tuple(len(orbs) for orbs in occupied_orbitals)
    vec = slater_determinant(norb, occupied_orbitals)

    theta = rng.standard_normal()
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        for spin_i, spin_j in itertools.product(range(2), repeat=2):
            target_orbs = ((i, spin_i), (j, spin_j))
            result = apply_num_op_prod_interaction(
                theta,
                vec,
                target_orbs,
                norb=norb,
                nelec=nelec,
            )
            if i in occupied_orbitals[spin_i] and j in occupied_orbitals[spin_j]:
                eig = theta
            else:
                eig = 0
            expected = np.exp(1j * eig) * vec
            np.testing.assert_allclose(result, expected, atol=1e-8)
