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
import pytest
import scipy.linalg
import scipy.sparse.linalg

import ffsim
from ffsim.spin import pair_for_spin


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(5)))
def test_apply_num_op_sum_evolution_spinless(norb: int, nelec: int):
    """Test applying time evolution of sum of number operators, spinless."""
    rng = np.random.default_rng()
    coeffs = rng.standard_normal(norb)
    time = rng.standard_normal()
    for occupied_orbitals in itertools.combinations(range(norb), nelec):
        state = ffsim.slater_determinant(norb, occupied_orbitals)
        original_state = state.copy()
        result = ffsim.apply_num_op_sum_evolution(state, coeffs, time, norb, nelec)
        eig = sum(coeffs[list(occupied_orbitals)])
        expected = np.exp(-1j * eig * time) * state
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(state, original_state)


@pytest.mark.parametrize(
    "norb, nelec, spin", ffsim.testing.generate_norb_nelec_spin(range(5))
)
def test_apply_num_op_sum_evolution_spinful(
    norb: int, nelec: tuple[int, int], spin: ffsim.Spin
):
    """Test applying time evolution of sum of number operators."""
    rng = np.random.default_rng()
    coeffs = rng.standard_normal(norb)
    time = rng.standard_normal()
    n_alpha, n_beta = nelec
    for alpha_orbitals in itertools.combinations(range(norb), n_alpha):
        for beta_orbitals in itertools.combinations(range(norb), n_beta):
            occupied_orbitals = (alpha_orbitals, beta_orbitals)
            state = ffsim.slater_determinant(norb, occupied_orbitals)
            original_state = state.copy()
            result = ffsim.apply_num_op_sum_evolution(
                state, pair_for_spin(coeffs, spin), time, norb, nelec
            )
            eig = 0
            if spin & ffsim.Spin.ALPHA:
                eig += sum(coeffs[list(occupied_orbitals[0])])
            if spin & ffsim.Spin.BETA:
                eig += sum(coeffs[list(occupied_orbitals[1])])
            expected = np.exp(-1j * eig * time) * state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_apply_quadratic_hamiltonian_evolution(norb: int, nelec: tuple[int, int]):
    """Test applying time evolution of a quadratic Hamiltonian."""
    rng = np.random.default_rng()
    mat = ffsim.random.random_hermitian(norb, seed=rng)
    eigs, vecs = scipy.linalg.eigh(mat)
    time = rng.standard_normal()
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=rng)
    result = ffsim.apply_num_op_sum_evolution(
        vec, eigs, time, norb, nelec, orbital_rotation=vecs
    )
    op = ffsim.contract.one_body_linop(mat, norb=norb, nelec=nelec)
    expected = scipy.sparse.linalg.expm_multiply(
        -1j * time * op, vec, traceA=-1j * time * np.sum(np.abs(mat))
    )
    np.testing.assert_allclose(result, expected)
