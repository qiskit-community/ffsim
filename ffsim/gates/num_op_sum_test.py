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

import numpy as np

from ffsim.fci import get_dimension, one_body_tensor_to_linop
from ffsim.gates import apply_num_op_sum_evolution
from ffsim.linalg import expm_multiply_taylor
from ffsim.random import random_hermitian, random_statevector
from ffsim.states import slater_determinant


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
    result = apply_num_op_sum_evolution(state, coeffs, time, norb, nelec)

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
            vec, eigs, time, norb, nelec, orbital_rotation=vecs
        )
        op = one_body_tensor_to_linop(mat, norb=norb, nelec=nelec)
        expected = expm_multiply_taylor(vec, -1j * time * op)
        np.testing.assert_allclose(result, expected, atol=1e-8)
