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


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(4)))
def test_apply_diag_coulomb_evolution_random_spinless(norb: int, nelec: int):
    """Test applying time evolution of random diagonal Coulomb operator."""
    rng = np.random.default_rng(4305)
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
        vec = ffsim.random.random_state_vector(dim, seed=rng)
        time = rng.uniform()

        result = ffsim.apply_diag_coulomb_evolution(
            vec,
            mat,
            time,
            norb,
            nelec,
            orbital_rotation=orbital_rotation,
        )

        op = ffsim.contract.diag_coulomb_linop(mat, norb=norb, nelec=(nelec, 0))
        if norb:
            orbital_op = ffsim.contract.one_body_linop(
                scipy.linalg.logm(orbital_rotation), norb=norb, nelec=(nelec, 0)
            )
            expected = scipy.sparse.linalg.expm_multiply(-orbital_op, vec, traceA=0)
            expected = scipy.sparse.linalg.expm_multiply(
                -1j * time * op, expected, traceA=-1j * time * np.sum(np.abs(mat))
            )
            expected = scipy.sparse.linalg.expm_multiply(orbital_op, expected, traceA=0)
        else:
            expected = vec

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec, z_representation",
    [
        (norb, nelec, z_representation)
        for (norb, nelec), z_representation in itertools.product(
            ffsim.testing.generate_norb_nelec(range(4)), [False, True]
        )
    ],
)
def test_apply_diag_coulomb_evolution_random_symmetric_spin(
    norb: int, nelec: tuple[int, int], z_representation: bool
):
    """Test applying time evolution of random diagonal Coulomb operator."""
    rng = np.random.default_rng(4305)
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
        vec = ffsim.random.random_state_vector(dim, seed=rng)
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
        if norb:
            orbital_op = ffsim.contract.one_body_linop(
                scipy.linalg.logm(orbital_rotation), norb=norb, nelec=nelec
            )
            expected = scipy.sparse.linalg.expm_multiply(-orbital_op, vec, traceA=0)
            expected = scipy.sparse.linalg.expm_multiply(
                -1j * time * op, expected, traceA=-1j * time * np.sum(np.abs(mat))
            )
            expected = scipy.sparse.linalg.expm_multiply(orbital_op, expected, traceA=0)
        else:
            expected = vec

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec, z_representation",
    [
        (norb, nelec, z_representation)
        for (norb, nelec), z_representation in itertools.product(
            ffsim.testing.generate_norb_nelec(range(5)), [False, True]
        )
    ],
)
def test_apply_diag_coulomb_evolution_conserves_spin_squared(
    norb: int, nelec: tuple[int, int], z_representation: bool
):
    """Test diagonal Coulomb evolution conserves spin squared."""
    rng = np.random.default_rng(8222)
    dim = ffsim.dim(norb, nelec)

    for _ in range(3):
        mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
        vec = ffsim.random.random_state_vector(dim, seed=rng)
        time = rng.uniform()

        spin_squared_init = ffsim.spin_square(vec, norb=norb, nelec=nelec)

        result = ffsim.apply_diag_coulomb_evolution(
            vec,
            mat,
            time,
            norb,
            nelec,
            orbital_rotation=orbital_rotation,
            z_representation=z_representation,
        )
        spin_squared_result = ffsim.spin_square(result, norb=norb, nelec=nelec)

        np.testing.assert_allclose(spin_squared_result, spin_squared_init)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(6)))
def test_apply_diag_coulomb_evolution_num_rep_asymmetric_spin(
    norb: int, nelec: tuple[int, int]
):
    """Test applying diagonal Coulomb evolution with asymmetric spin action."""
    rng = np.random.default_rng()
    n_alpha, n_beta = nelec
    time = rng.uniform(-10, 10)
    mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    mat_ab = rng.standard_normal((norb, norb))
    mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    for alpha_orbitals in itertools.combinations(range(norb), n_alpha):
        for beta_orbitals in itertools.combinations(range(norb), n_beta):
            occupied_orbitals = (alpha_orbitals, beta_orbitals)
            state = ffsim.slater_determinant(norb, occupied_orbitals)
            original_state = state.copy()

            # (mat_aa, mat_ab, mat_bb)
            result = ffsim.apply_diag_coulomb_evolution(
                state, (mat_aa, mat_ab, mat_bb), time, norb, nelec
            )
            eig = 0
            for i, j in itertools.product(range(norb), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                        if (sigma, tau) == (0, 0):
                            eig += 0.5 * mat_aa[i, j]
                        elif (sigma, tau) == (1, 1):
                            eig += 0.5 * mat_bb[i, j]
                        elif (sigma, tau) == (0, 1):
                            eig += 0.5 * mat_ab[i, j]
                        elif (sigma, tau) == (1, 0):
                            eig += 0.5 * mat_ab[j, i]
            expected = np.exp(-1j * eig * time) * state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)
            # Numpy array input
            result = ffsim.apply_diag_coulomb_evolution(
                state, np.stack((mat_aa, mat_ab, mat_bb)), time, norb, nelec
            )
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)

            # (mat_aa, None, None)
            result = ffsim.apply_diag_coulomb_evolution(
                state, (mat_aa, None, None), time, norb, nelec
            )
            eig = 0
            for i, j in itertools.product(range(norb), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                        if (sigma, tau) == (0, 0):
                            eig += 0.5 * mat_aa[i, j]
            expected = np.exp(-1j * eig * time) * state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)

            # (None, mat_ab, None)
            result = ffsim.apply_diag_coulomb_evolution(
                state, (None, mat_ab, None), time, norb, nelec
            )
            eig = 0
            for i, j in itertools.product(range(norb), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                        if (sigma, tau) == (0, 1):
                            eig += 0.5 * mat_ab[i, j]
                        elif (sigma, tau) == (1, 0):
                            eig += 0.5 * mat_ab[j, i]
            expected = np.exp(-1j * eig * time) * state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)

            # (None, None, mat_bb)
            result = ffsim.apply_diag_coulomb_evolution(
                state, (None, None, mat_bb), time, norb, nelec
            )
            eig = 0
            for i, j in itertools.product(range(norb), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                        if (sigma, tau) == (1, 1):
                            eig += 0.5 * mat_bb[i, j]
            expected = np.exp(-1j * eig * time) * state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)

            # (None, None, None)
            result = ffsim.apply_diag_coulomb_evolution(
                state, (None, None, None), time, norb, nelec
            )
            expected = state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(6)))
def test_apply_diag_coulomb_evolution_z_rep_asymmetric_spin(
    norb: int, nelec: tuple[int, int]
):
    """Test applying diagonal Coulomb evolution with asymmetric spin action."""
    rng = np.random.default_rng()
    n_alpha, n_beta = nelec
    time = rng.uniform(-10, 10)
    mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    mat_ab = rng.standard_normal((norb, norb))
    mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    for alpha_orbitals in itertools.combinations(range(norb), n_alpha):
        for beta_orbitals in itertools.combinations(range(norb), n_beta):
            occupied_orbitals = (alpha_orbitals, beta_orbitals)
            state = ffsim.slater_determinant(norb, occupied_orbitals)
            original_state = state.copy()

            # (mat_aa, mat_ab, mat_bb)
            result = ffsim.apply_diag_coulomb_evolution(
                state,
                (mat_aa, mat_ab, mat_bb),
                time,
                norb,
                nelec,
                z_representation=True,
            )
            eig = 0
            for a, b in itertools.combinations(range(2 * norb), 2):
                sigma, i = divmod(a, norb)
                tau, j = divmod(b, norb)
                sign_i = -1 if i in occupied_orbitals[sigma] else 1
                sign_j = -1 if j in occupied_orbitals[tau] else 1
                if (sigma, tau) == (0, 0):
                    eig += 0.25 * sign_i * sign_j * mat_aa[i, j]
                elif (sigma, tau) == (1, 1):
                    eig += 0.25 * sign_i * sign_j * mat_bb[i, j]
                else:
                    eig += 0.25 * sign_i * sign_j * mat_ab[i, j]
            expected = np.exp(-1j * eig * time) * state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)

            # (mat_aa, None, None)
            result = ffsim.apply_diag_coulomb_evolution(
                state, (mat_aa, None, None), time, norb, nelec, z_representation=True
            )
            eig = 0
            for a, b in itertools.combinations(range(2 * norb), 2):
                sigma, i = divmod(a, norb)
                tau, j = divmod(b, norb)
                sign_i = -1 if i in occupied_orbitals[sigma] else 1
                sign_j = -1 if j in occupied_orbitals[tau] else 1
                if (sigma, tau) == (0, 0):
                    eig += 0.25 * sign_i * sign_j * mat_aa[i, j]
            expected = np.exp(-1j * eig * time) * state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)

            # (None, mat_ab, None)
            result = ffsim.apply_diag_coulomb_evolution(
                state, (None, mat_ab, None), time, norb, nelec, z_representation=True
            )
            eig = 0
            for a, b in itertools.combinations(range(2 * norb), 2):
                sigma, i = divmod(a, norb)
                tau, j = divmod(b, norb)
                sign_i = -1 if i in occupied_orbitals[sigma] else 1
                sign_j = -1 if j in occupied_orbitals[tau] else 1
                if sigma != tau:
                    eig += 0.25 * sign_i * sign_j * mat_ab[i, j]
            expected = np.exp(-1j * eig * time) * state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)

            # (None, None, mat_bb)
            result = ffsim.apply_diag_coulomb_evolution(
                state, (None, None, mat_bb), time, norb, nelec, z_representation=True
            )
            eig = 0
            for a, b in itertools.combinations(range(2 * norb), 2):
                sigma, i = divmod(a, norb)
                tau, j = divmod(b, norb)
                sign_i = -1 if i in occupied_orbitals[sigma] else 1
                sign_j = -1 if j in occupied_orbitals[tau] else 1
                if (sigma, tau) == (1, 1):
                    eig += 0.25 * sign_i * sign_j * mat_bb[i, j]
            expected = np.exp(-1j * eig * time) * state
            np.testing.assert_allclose(result, expected)
            np.testing.assert_allclose(state, original_state)
