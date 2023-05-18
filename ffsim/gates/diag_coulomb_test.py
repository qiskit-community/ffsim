# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for diagonal Coulomb evolution."""

from __future__ import annotations

import itertools

import numpy as np

from ffsim.gates import apply_diag_coulomb_evolution
from ffsim.random_utils import random_hermitian
from ffsim.states import slater_determinant


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

        mat = np.real(np.array(random_hermitian(norb, seed=rng)))
        time = 0.6
        result = apply_diag_coulomb_evolution(state, mat, time, norb, nelec)

        eig = 0
        for i, j in itertools.product(range(norb), repeat=2):
            for sigma, tau in itertools.product(range(2), repeat=2):
                if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                    eig += 0.5 * mat[i, j]
        expected = np.exp(-1j * eig * time) * state

        np.testing.assert_allclose(result, expected, atol=1e-8)
        np.testing.assert_allclose(state, original_state)


def test_apply_diag_coulomb_evolution_alpha_beta():
    """Test applying time evolution of diagonal Coulomb operator with alpha beta mat."""
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

        mat = np.real(np.array(random_hermitian(norb, seed=rng)))
        mat_alpha_beta = np.real(np.array(random_hermitian(norb, seed=rng)))
        time = 0.6
        result = apply_diag_coulomb_evolution(
            state, mat, time, norb, nelec, mat_alpha_beta=mat_alpha_beta
        )

        eig = 0
        for i, j in itertools.product(range(norb), repeat=2):
            for sigma, tau in itertools.product(range(2), repeat=2):
                if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                    this_mat = mat if sigma == tau else mat_alpha_beta
                    eig += 0.5 * this_mat[i, j]
        expected = np.exp(-1j * eig * time) * state

        np.testing.assert_allclose(result, expected, atol=1e-8)
        np.testing.assert_allclose(state, original_state)
