# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for diagonal Coulomb contraction."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import ffsim


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(6)))
def test_contract_diag_coulomb_num_rep_symmetric_spin(
    norb: int, nelec: tuple[int, int]
):
    """Test contracting a diagonal Coulomb matrix, symmetric spin."""
    rng = np.random.default_rng()
    n_alpha, n_beta = nelec
    mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    for alpha_orbitals in itertools.combinations(range(norb), n_alpha):
        for beta_orbitals in itertools.combinations(range(norb), n_beta):
            occupied_orbitals = (alpha_orbitals, beta_orbitals)
            state = ffsim.slater_determinant(norb, occupied_orbitals)
            result = ffsim.contract.contract_diag_coulomb(
                state,
                mat,
                norb=norb,
                nelec=nelec,
            )
            eig = 0
            for i, j in itertools.product(range(norb), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                        eig += 0.5 * mat[i, j]
            expected = eig * state
            np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(6)))
def test_contract_diag_coulomb_num_rep_asymmetric_spin(
    norb: int, nelec: tuple[int, int]
):
    """Test contracting a diagonal Coulomb matrix, asymmetric spin."""
    rng = np.random.default_rng()
    n_alpha, n_beta = nelec
    mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    mat_ab = rng.standard_normal((norb, norb))
    mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    for alpha_orbitals in itertools.combinations(range(norb), n_alpha):
        for beta_orbitals in itertools.combinations(range(norb), n_beta):
            occupied_orbitals = (alpha_orbitals, beta_orbitals)
            state = ffsim.slater_determinant(norb, occupied_orbitals)

            # (mat_aa, mat_ab, mat_bb)
            result = ffsim.contract.contract_diag_coulomb(
                state, (mat_aa, mat_ab, mat_bb), norb=norb, nelec=nelec
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
            expected = eig * state
            np.testing.assert_allclose(result, expected)

            # (mat_aa, None, None)
            result = ffsim.contract.contract_diag_coulomb(
                state, (mat_aa, None, None), norb=norb, nelec=nelec
            )
            eig = 0
            for i, j in itertools.product(range(norb), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                        if (sigma, tau) == (0, 0):
                            eig += 0.5 * mat_aa[i, j]
            expected = eig * state
            np.testing.assert_allclose(result, expected)

            # (None, mat_ab, None)
            result = ffsim.contract.contract_diag_coulomb(
                state, (None, mat_ab, None), norb=norb, nelec=nelec
            )
            eig = 0
            for i, j in itertools.product(range(norb), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                        if (sigma, tau) == (0, 1):
                            eig += 0.5 * mat_ab[i, j]
                        elif (sigma, tau) == (1, 0):
                            eig += 0.5 * mat_ab[j, i]
            expected = eig * state
            np.testing.assert_allclose(result, expected)

            # (None, None, mat_bb)
            result = ffsim.contract.contract_diag_coulomb(
                state, (None, None, mat_bb), norb=norb, nelec=nelec
            )
            eig = 0
            for i, j in itertools.product(range(norb), repeat=2):
                for sigma, tau in itertools.product(range(2), repeat=2):
                    if i in occupied_orbitals[sigma] and j in occupied_orbitals[tau]:
                        if (sigma, tau) == (1, 1):
                            eig += 0.5 * mat_bb[i, j]
            expected = eig * state
            np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(6)))
def test_contract_diag_coulomb_z_rep_symmetric_spin(norb: int, nelec: tuple[int, int]):
    """Test contracting a diagonal Coulomb matrix in the Z representation."""
    rng = np.random.default_rng()
    n_alpha, n_beta = nelec
    mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    for alpha_orbitals in itertools.combinations(range(norb), n_alpha):
        for beta_orbitals in itertools.combinations(range(norb), n_beta):
            occupied_orbitals = (alpha_orbitals, beta_orbitals)
            state = ffsim.slater_determinant(norb, occupied_orbitals)
            result = ffsim.contract.contract_diag_coulomb(
                state,
                mat,
                norb=norb,
                nelec=nelec,
                z_representation=True,
            )
            eig = 0
            for a, b in itertools.combinations(range(2 * norb), 2):
                sigma, i = divmod(a, norb)
                tau, j = divmod(b, norb)
                sign_i = -1 if i in occupied_orbitals[sigma] else 1
                sign_j = -1 if j in occupied_orbitals[tau] else 1
                eig += 0.25 * sign_i * sign_j * mat[i, j]
            expected = eig * state
            np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(6)))
def test_contract_diag_coulomb_z_rep_asymmetric_spin(norb: int, nelec: tuple[int, int]):
    """Test contracting a diagonal Coulomb matrix in the Z representation."""
    rng = np.random.default_rng()
    n_alpha, n_beta = nelec
    mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    mat_ab = rng.standard_normal((norb, norb))
    mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    for alpha_orbitals in itertools.combinations(range(norb), n_alpha):
        for beta_orbitals in itertools.combinations(range(norb), n_beta):
            occupied_orbitals = (alpha_orbitals, beta_orbitals)
            state = ffsim.slater_determinant(norb, occupied_orbitals)

            # (mat_aa, mat_ab, mat_bb)
            result = ffsim.contract.contract_diag_coulomb(
                state,
                (mat_aa, mat_ab, mat_bb),
                norb=norb,
                nelec=nelec,
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
            expected = eig * state
            np.testing.assert_allclose(result, expected)

            # (mat_aa, None, None)
            result = ffsim.contract.contract_diag_coulomb(
                state,
                (mat_aa, None, None),
                norb=norb,
                nelec=nelec,
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
            expected = eig * state
            np.testing.assert_allclose(result, expected)

            # (None, mat_ab, None)
            result = ffsim.contract.contract_diag_coulomb(
                state,
                (None, mat_ab, None),
                norb=norb,
                nelec=nelec,
                z_representation=True,
            )
            eig = 0
            for a, b in itertools.combinations(range(2 * norb), 2):
                sigma, i = divmod(a, norb)
                tau, j = divmod(b, norb)
                sign_i = -1 if i in occupied_orbitals[sigma] else 1
                sign_j = -1 if j in occupied_orbitals[tau] else 1
                if sigma != tau:
                    eig += 0.25 * sign_i * sign_j * mat_ab[i, j]
            expected = eig * state
            np.testing.assert_allclose(result, expected)

            # (None, None, mat_bb)
            result = ffsim.contract.contract_diag_coulomb(
                state,
                (None, None, mat_bb),
                norb=norb,
                nelec=nelec,
                z_representation=True,
            )
            eig = 0
            for a, b in itertools.combinations(range(2 * norb), 2):
                sigma, i = divmod(a, norb)
                tau, j = divmod(b, norb)
                sign_i = -1 if i in occupied_orbitals[sigma] else 1
                sign_j = -1 if j in occupied_orbitals[tau] else 1
                if (sigma, tau) == (1, 1):
                    eig += 0.25 * sign_i * sign_j * mat_bb[i, j]
            expected = eig * state
            np.testing.assert_allclose(result, expected)


def test_diag_coulomb_to_linop():
    """Test converting a diagonal Coulomb matrix to a linear operator."""
    norb = 5
    rng = np.random.default_rng()
    n_alpha = rng.integers(1, norb + 1)
    n_beta = rng.integers(1, norb + 1)
    nelec = (n_alpha, n_beta)
    dim = ffsim.dim(norb, nelec)

    mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    vec = ffsim.random.random_state_vector(dim, seed=rng)

    linop = ffsim.contract.diag_coulomb_linop(
        mat, norb=norb, nelec=nelec, orbital_rotation=orbital_rotation
    )
    result = linop @ vec

    expected = ffsim.apply_orbital_rotation(
        vec, orbital_rotation.T.conj(), norb=norb, nelec=nelec
    )
    expected = ffsim.contract.contract_diag_coulomb(
        expected, mat, norb=norb, nelec=nelec
    )
    expected = ffsim.apply_orbital_rotation(
        expected, orbital_rotation, norb=norb, nelec=nelec
    )

    np.testing.assert_allclose(result, expected)
