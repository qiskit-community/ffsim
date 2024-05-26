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
import scipy.linalg
import scipy.sparse.linalg

import ffsim
from ffsim.spin import pair_for_spin


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

    for spin in ffsim.Spin:
        result = ffsim.apply_num_op_sum_evolution(
            state, pair_for_spin(coeffs, spin), time, norb, nelec
        )

        eig = 0
        for i in range(norb):
            if spin & ffsim.Spin.ALPHA and i in occupied_orbitals[0]:
                eig += coeffs[i]
            if spin & ffsim.Spin.BETA and i in occupied_orbitals[1]:
                eig += coeffs[i]
        expected = np.exp(-1j * eig * time) * state

        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(state, original_state)


def test_apply_quadratic_hamiltonian_evolution():
    """Test applying time evolution of a quadratic Hamiltonian."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        nelec = tuple(rng.integers(1, norb + 1, size=2))
        dim = ffsim.dim(norb, nelec)

        mat = ffsim.random.random_hermitian(norb, seed=rng)
        eigs, vecs = scipy.linalg.eigh(mat)
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
