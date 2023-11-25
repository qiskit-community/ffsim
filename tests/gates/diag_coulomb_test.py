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


@pytest.mark.parametrize("z_representation", [False, True])
def test_apply_diag_coulomb_evolution(z_representation: bool):
    """Test applying time evolution of diagonal Coulomb operator."""
    rng = np.random.default_rng()
    norb = 5
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        nelec = (n_alpha, n_beta)
        dim = ffsim.dim(norb, nelec)

        mat = np.real(np.array(ffsim.random.random_hermitian(norb, seed=rng)))
        orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
        vec = ffsim.random.random_statevector(dim, seed=rng)
        time = rng.uniform()
        result = ffsim.apply_diag_coulomb_evolution(
            vec,
            mat,
            time,
            norb,
            nelec,
            orbital_rotation=orbital_rotation,
            z_representation=z_representation,
        )

        op = ffsim.contract.diag_coulomb_linop(
            mat, norb=norb, nelec=nelec, z_representation=z_representation
        )
        orbital_op = ffsim.contract.one_body_linop(
            scipy.linalg.logm(orbital_rotation), norb=norb, nelec=nelec
        )
        expected = scipy.sparse.linalg.expm_multiply(
            -orbital_op, vec, traceA=np.sum(np.abs(orbital_rotation))
        )
        expected = scipy.sparse.linalg.expm_multiply(
            -1j * time * op, expected, traceA=np.sum(np.abs(mat))
        )
        expected = scipy.sparse.linalg.expm_multiply(
            orbital_op, expected, traceA=np.sum(np.abs(orbital_rotation))
        )

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("z_representation", [False, True])
def test_apply_diag_coulomb_evolution_alpha_beta(z_representation: bool):
    """Test applying time evolution of diagonal Coulomb operator with alpha beta mat."""
    rng = np.random.default_rng()
    norb = 5
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        nelec = (n_alpha, n_beta)
        dim = ffsim.dim(norb, nelec)

        mat = np.real(np.array(ffsim.random.random_hermitian(norb, seed=rng)))
        mat_alpha_beta = np.real(
            np.array(ffsim.random.random_hermitian(norb, seed=rng))
        )
        orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
        vec = ffsim.random.random_statevector(dim, seed=rng)
        time = rng.uniform()
        result = ffsim.apply_diag_coulomb_evolution(
            vec,
            mat,
            time,
            norb,
            nelec,
            mat_alpha_beta=mat_alpha_beta,
            orbital_rotation=orbital_rotation,
            z_representation=z_representation,
        )

        op = ffsim.contract.diag_coulomb_linop(
            mat,
            norb=norb,
            nelec=nelec,
            mat_alpha_beta=mat_alpha_beta,
            z_representation=z_representation,
        )
        orbital_op = ffsim.contract.one_body_linop(
            scipy.linalg.logm(orbital_rotation), norb=norb, nelec=nelec
        )
        expected = scipy.sparse.linalg.expm_multiply(
            -orbital_op, vec, traceA=np.sum(np.abs(orbital_rotation))
        )
        expected = scipy.sparse.linalg.expm_multiply(
            -1j * time * op, expected, traceA=np.sum(np.abs(mat))
        )
        expected = scipy.sparse.linalg.expm_multiply(
            orbital_op, expected, traceA=np.sum(np.abs(orbital_rotation))
        )

        np.testing.assert_allclose(result, expected)


def test_apply_diag_coulomb_evolution_eigenvalue():
    """Test applying diagonal Coulomb evolution with alpha beta mat to eigenvector."""
    rng = np.random.default_rng()
    norb = 5
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        occupied_orbitals = (
            rng.choice(norb, n_alpha, replace=False),
            rng.choice(norb, n_beta, replace=False),
        )
        nelec = tuple(len(orbs) for orbs in occupied_orbitals)
        state = ffsim.slater_determinant(norb, occupied_orbitals)
        original_state = state.copy()

        mat = np.real(np.array(ffsim.random.random_hermitian(norb, seed=rng)))
        mat_alpha_beta = np.real(
            np.array(ffsim.random.random_hermitian(norb, seed=rng))
        )
        time = 0.6
        result = ffsim.apply_diag_coulomb_evolution(
            state, mat, time, norb, nelec, mat_alpha_beta=mat_alpha_beta
        )

        eig = 0
        for i, j in itertools.product(range(norb), repeat=2):
            for sigma, tau in itertools.product(range(2), repeat=2):
                if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                    this_mat = mat if sigma == tau else mat_alpha_beta
                    eig += 0.5 * this_mat[i, j]
        expected = np.exp(-1j * eig * time) * state

        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(state, original_state)
