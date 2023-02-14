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

import itertools

import numpy as np
import pytest
import scipy.sparse.linalg

from fqcsim.fci import contract_1e, get_dimension, one_body_tensor_to_linop
from fqcsim.gates import (
    apply_core_tensor_evolution,
    apply_givens_rotation_adjacent,
    apply_num_interaction,
    apply_num_num_interaction,
    apply_num_op_sum_evolution,
    apply_orbital_rotation,
    apply_tunneling_interaction_adjacent,
)
from fqcsim.random_utils import random_hermitian, random_statevector, random_unitary
from fqcsim.states import slater_determinant


def test_apply_orbital_rotation():
    """Test applying orbital basis change."""
    n_orbitals = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, n_orbitals + 1)
    n_beta = rng.integers(1, n_orbitals + 1)
    occupied_orbitals = (
        rng.choice(n_orbitals, n_alpha, replace=False),
        rng.choice(n_orbitals, n_beta, replace=False),
    )

    one_body_tensor = np.array(random_hermitian(n_orbitals, seed=rng))
    eigs, vecs = np.linalg.eigh(one_body_tensor)
    eig = sum(np.sum(eigs[orbs]) for orbs in occupied_orbitals)
    n_electrons = tuple(len(orbs) for orbs in occupied_orbitals)
    state = slater_determinant(n_orbitals, occupied_orbitals)
    original_state = state.copy()
    final_state = apply_orbital_rotation(vecs, state, n_orbitals, n_electrons)
    np.testing.assert_allclose(np.linalg.norm(final_state), 1.0, atol=1e-8)
    result = contract_1e(one_body_tensor, final_state, n_orbitals, n_electrons)
    expected = eig * final_state
    np.testing.assert_allclose(result, expected, atol=1e-8)
    # check that the state was not modified
    np.testing.assert_allclose(state, original_state)


def test_apply_orbital_rotation_compose():
    """Test composing orbital basis changes."""
    n_orbitals = 5
    rng = np.random.default_rng()
    basis_change_1 = np.array(random_unitary(n_orbitals, seed=rng))
    basis_change_2 = np.array(random_unitary(n_orbitals, seed=rng))

    n_electrons = (3, 2)
    dim = get_dimension(n_orbitals, n_electrons)
    state = np.array(random_statevector(dim, seed=rng))

    result = apply_orbital_rotation(basis_change_1, state, n_orbitals, n_electrons)
    result = apply_orbital_rotation(
        basis_change_2 @ basis_change_1.T.conj(), result, n_orbitals, n_electrons
    )
    expected_state = apply_orbital_rotation(
        basis_change_2, state, n_orbitals, n_electrons
    )

    np.testing.assert_allclose(result, expected_state, atol=1e-8)


def test_apply_core_tensor_evolution():
    """Test applying time evolution of core tensor."""
    n_orbitals = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, n_orbitals + 1)
    n_beta = rng.integers(1, n_orbitals + 1)
    occupied_orbitals = (
        rng.choice(n_orbitals, n_alpha, replace=False),
        rng.choice(n_orbitals, n_beta, replace=False),
    )
    n_electrons = tuple(len(orbs) for orbs in occupied_orbitals)
    state = slater_determinant(n_orbitals, occupied_orbitals)
    original_state = state.copy()

    core_tensor = np.real(np.array(random_hermitian(n_orbitals, seed=rng)))
    time = 0.5
    result = apply_core_tensor_evolution(
        core_tensor, state, time, n_orbitals, n_electrons
    )

    eig = 0
    for i, j in itertools.product(range(n_orbitals), repeat=2):
        for sigma, tau in itertools.product(range(2), repeat=2):
            if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                eig += 0.5 * core_tensor[i, j]
    expected = np.exp(-1j * eig * time) * state

    np.testing.assert_allclose(result, expected, atol=1e-8)
    np.testing.assert_allclose(state, original_state)


def test_apply_num_op_sum_evolution():
    """Test applying time evolution of sum of number operators."""
    n_orbitals = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, n_orbitals + 1)
    n_beta = rng.integers(1, n_orbitals + 1)
    occupied_orbitals = (
        rng.choice(n_orbitals, n_alpha, replace=False),
        rng.choice(n_orbitals, n_beta, replace=False),
    )
    n_electrons = tuple(len(orbs) for orbs in occupied_orbitals)
    state = slater_determinant(n_orbitals, occupied_orbitals)
    original_state = state.copy()

    coeffs = rng.standard_normal(n_orbitals)
    time = 0.5
    result = apply_num_op_sum_evolution(coeffs, state, time, n_orbitals, n_electrons)

    eig = 0
    for i in range(n_orbitals):
        for sigma in range(2):
            if i in occupied_orbitals[sigma]:
                eig += coeffs[i]
    expected = np.exp(-1j * eig * time) * state

    np.testing.assert_allclose(result, expected, atol=1e-8)
    np.testing.assert_allclose(state, original_state)


def test_apply_givens_rotation_adjacent():
    """Test applying Givens rotation to adjacent orbitals."""
    n_orbitals = 5
    n_alpha = 3
    n_beta = 2
    n_electrons = (n_alpha, n_beta)

    dim = get_dimension(n_orbitals, n_electrons)
    rng = np.random.default_rng()
    vec = np.array(random_statevector(dim, seed=rng))
    original_vec = vec.copy()
    theta = rng.standard_normal()
    for i in range(n_orbitals - 1):
        for target_orbitals in [(i, i + 1), (i + 1, i)]:
            result = apply_givens_rotation_adjacent(
                theta,
                vec,
                target_orbitals,
                n_orbitals=n_orbitals,
                n_electrons=n_electrons,
            )
            generator = np.zeros((n_orbitals, n_orbitals))
            j, k = target_orbitals
            generator[j, k] = theta
            generator[k, j] = -theta
            linop = one_body_tensor_to_linop(generator, n_electrons=n_electrons)
            expected = scipy.sparse.linalg.expm_multiply(linop, vec, traceA=theta)
            np.testing.assert_allclose(result, expected, atol=1e-8)
    np.testing.assert_allclose(vec, original_vec)


def test_apply_tunneling_interaction():
    """Test applying tunneling interaction to adjacent orbitals."""
    n_orbitals = 5
    n_alpha = 3
    n_beta = 2
    n_electrons = (n_alpha, n_beta)

    dim = get_dimension(n_orbitals, n_electrons)
    rng = np.random.default_rng()
    vec = np.array(random_statevector(dim, seed=rng))
    theta = rng.standard_normal()
    for i in range(n_orbitals - 1):
        for target_orbitals in [(i, i + 1), (i + 1, i)]:
            result = apply_tunneling_interaction_adjacent(
                theta,
                vec,
                target_orbitals,
                n_orbitals=n_orbitals,
                n_electrons=n_electrons,
            )
            generator = np.zeros((n_orbitals, n_orbitals))
            j, k = target_orbitals
            generator[j, k] = theta
            generator[k, j] = theta
            linop = one_body_tensor_to_linop(generator, n_electrons=n_electrons)
            expected = scipy.sparse.linalg.expm_multiply(1j * linop, vec, traceA=theta)
            np.testing.assert_allclose(result, expected, atol=1e-8)


def test_apply_num_interaction():
    """Test applying number interaction."""
    n_orbitals = 5
    n_alpha = 3
    n_beta = 2
    n_electrons = (n_alpha, n_beta)

    dim = get_dimension(n_orbitals, n_electrons)
    rng = np.random.default_rng()
    vec = np.array(random_statevector(dim, seed=rng))
    theta = rng.standard_normal()
    for target_orbital in range(n_orbitals):
        result = apply_num_interaction(
            theta, vec, target_orbital, n_orbitals=n_orbitals, n_electrons=n_electrons
        )
        generator = np.zeros((n_orbitals, n_orbitals))
        generator[target_orbital, target_orbital] = theta
        linop = one_body_tensor_to_linop(generator, n_electrons=n_electrons)
        expected = scipy.sparse.linalg.expm_multiply(1j * linop, vec, traceA=theta)
        np.testing.assert_allclose(result, expected, atol=1e-8)


def test_apply_num_num_interaction():
    """Test applying number interaction."""
    n_orbitals = 5

    rng = np.random.default_rng()
    n_alpha = 3
    n_beta = 2
    occupied_orbitals = (
        rng.choice(n_orbitals, n_alpha, replace=False),
        rng.choice(n_orbitals, n_beta, replace=False),
    )
    n_electrons = tuple(len(orbs) for orbs in occupied_orbitals)
    vec = slater_determinant(n_orbitals, occupied_orbitals)

    theta = rng.standard_normal()
    for i, j in itertools.combinations_with_replacement(range(n_orbitals), 2):
        for spin_i, spin_j in itertools.product(range(2), repeat=2):
            target_orbitals = ((i, spin_i), (j, spin_j))
            result = apply_num_num_interaction(
                theta,
                vec,
                target_orbitals,
                n_orbitals=n_orbitals,
                n_electrons=n_electrons,
            )
            if i in occupied_orbitals[spin_i] and j in occupied_orbitals[spin_j]:
                eig = theta
            else:
                eig = 0
            expected = np.exp(1j * eig) * vec
            np.testing.assert_allclose(result, expected, atol=1e-8)
