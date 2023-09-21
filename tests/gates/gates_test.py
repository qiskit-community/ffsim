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
import scipy.sparse.linalg

import ffsim


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
    ],
)
def test_apply_givens_rotation(norb: int, nelec: tuple[int, int]):
    """Test applying Givens rotation."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(ffsim.random.random_statevector(dim, seed=rng))
    original_vec = vec.copy()
    theta = rng.standard_normal()
    for i, j in itertools.product(range(norb), repeat=2):
        if i == j:
            continue
        result = ffsim.apply_givens_rotation(vec, theta, (i, j), norb=norb, nelec=nelec)
        generator = np.zeros((norb, norb))
        generator[i, j] = theta
        generator[j, i] = -theta
        linop = ffsim.contract.hamiltonian_linop(
            one_body_tensor=generator, norb=norb, nelec=nelec
        )
        expected = scipy.sparse.linalg.expm_multiply(
            linop, vec, traceA=np.sum(np.abs(generator))
        )
        np.testing.assert_allclose(result, expected, atol=1e-8)
    np.testing.assert_allclose(vec, original_vec)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
    ],
)
def test_apply_tunneling_interaction(norb: int, nelec: tuple[int, int]):
    """Test applying tunneling interaction."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(ffsim.random.random_statevector(dim, seed=rng))
    theta = rng.standard_normal()
    for i, j in itertools.product(range(norb), repeat=2):
        if i == j:
            continue
        result = ffsim.apply_tunneling_interaction(
            vec, theta, (i, j), norb=norb, nelec=nelec
        )
        generator = np.zeros((norb, norb))
        generator[i, j] = theta
        generator[j, i] = theta
        linop = ffsim.contract.hamiltonian_linop(
            one_body_tensor=generator, norb=norb, nelec=nelec
        )
        expected = scipy.sparse.linalg.expm_multiply(
            1j * linop, vec, traceA=np.sum(np.abs(generator))
        )
        np.testing.assert_allclose(result, expected, atol=1e-8)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
    ],
)
def test_apply_num_interaction(norb: int, nelec: tuple[int, int]):
    """Test applying number interaction."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = np.array(ffsim.random.random_statevector(dim, seed=rng))
    theta = rng.standard_normal()
    for target_orb in range(norb):
        result = ffsim.apply_num_interaction(
            vec, theta, target_orb, norb=norb, nelec=nelec
        )
        generator = np.zeros((norb, norb))
        generator[target_orb, target_orb] = theta
        linop = ffsim.contract.hamiltonian_linop(
            one_body_tensor=generator, norb=norb, nelec=nelec
        )
        expected = scipy.sparse.linalg.expm_multiply(
            1j * linop, vec, traceA=np.sum(np.abs(generator))
        )
        np.testing.assert_allclose(result, expected, atol=1e-8)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (1, 1)),
        (4, (2, 2)),
        (5, (3, 2)),
    ],
)
def test_apply_num_num_interaction(norb: int, nelec: tuple[int, int]):
    """Test applying number-number interaction."""
    rng = np.random.default_rng()
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)
    vec = ffsim.slater_determinant(norb, occupied_orbitals)

    theta = rng.standard_normal()
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        for spin_i, spin_j in itertools.product(range(2), repeat=2):
            target_orbs: tuple[set[int], set[int]] = (set(), set())
            target_orbs[spin_i].add(i)
            target_orbs[spin_j].add(j)
            alpha_orbs, beta_orbs = target_orbs
            result = ffsim.apply_num_op_prod_interaction(
                vec,
                theta,
                target_orbs=(list(alpha_orbs), list(beta_orbs)),
                norb=norb,
                nelec=nelec,
            )
            if i in occupied_orbitals[spin_i] and j in occupied_orbitals[spin_j]:
                eig = theta
            else:
                eig = 0
            expected = np.exp(1j * eig) * vec
            np.testing.assert_allclose(result, expected, atol=1e-8)
