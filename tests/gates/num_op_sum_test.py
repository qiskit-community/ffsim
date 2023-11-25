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
import pytest
import scipy.sparse.linalg

import ffsim


def test_apply_num_op_sum_evolution():
    """Test applying time evolution of sum of number operators."""
    norb = 5
    rng = np.random.default_rng()
    nelec = tuple(rng.integers(1, norb + 1, size=2))
    n_alpha, n_beta = nelec
    occupied_orbitals = (
        rng.choice(norb, n_alpha, replace=False),
        rng.choice(norb, n_beta, replace=False),
    )
    state = ffsim.slater_determinant(norb, occupied_orbitals)
    original_state = state.copy()

    coeffs = rng.standard_normal(norb)
    time = 0.6
    result = ffsim.apply_num_op_sum_evolution(state, coeffs, time, norb, nelec)

    eig = 0
    for i in range(norb):
        for sigma in range(2):
            if i in occupied_orbitals[sigma]:
                eig += coeffs[i]
    expected = np.exp(-1j * eig * time) * state

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(state, original_state)


def test_apply_num_op_sum_evolution_wrong_coeffs_length():
    """Test passing wrong coeffs length raises correct error."""
    norb = 5
    nelec = (3, 2)
    n_alpha, n_beta = nelec
    occupied_orbitals = (range(n_alpha), range(n_beta))
    state = ffsim.slater_determinant(norb, occupied_orbitals)

    coeffs = np.ones(norb - 1)
    with pytest.raises(ValueError, match="length"):
        _ = ffsim.apply_num_op_sum_evolution(
            state, coeffs, time=1.0, norb=norb, nelec=nelec
        )


def test_apply_quadratic_hamiltonian_evolution():
    """Test applying time evolution of a quadratic Hamiltonian."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        nelec = tuple(rng.integers(1, norb + 1, size=2))
        dim = ffsim.dim(norb, nelec)

        mat = ffsim.random.random_hermitian(norb, seed=rng)
        eigs, vecs = np.linalg.eigh(mat)
        vec = ffsim.random.random_statevector(dim, seed=rng)

        time = 0.6
        result = ffsim.apply_num_op_sum_evolution(
            vec, eigs, time, norb, nelec, orbital_rotation=vecs
        )
        op = ffsim.contract.one_body_linop(mat, norb=norb, nelec=nelec)
        expected = scipy.sparse.linalg.expm_multiply(
            -1j * time * op, vec, traceA=np.sum(np.abs(mat))
        )
        np.testing.assert_allclose(result, expected)
