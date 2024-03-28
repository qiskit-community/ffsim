# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Fermi-Hubbard model Hamiltonian."""

from __future__ import annotations

from math import comb

import numpy as np
import scipy

import ffsim
from ffsim.operators.fermi_hubbard import fermi_hubbard


def test_non_interacting_FH_model():
    """Test non-interacting Fermi-Hubbard model Hamiltonian."""

    # open boundary conditions
    op = fermi_hubbard(norb=4, tunneling=1, interaction=0)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -4.472135955000)

    # periodic boundary conditions
    op_periodic = fermi_hubbard(norb=4, tunneling=1, interaction=0, periodic=True)
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -4.000000000000)


def test_FH_model_with_U():
    """Test Fermi-Hubbard model Hamiltonian with onsite interaction."""

    # open boundary conditions
    op = fermi_hubbard(norb=4, tunneling=1, interaction=2)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -2.875942809005)

    # periodic boundary conditions
    op_periodic = fermi_hubbard(norb=4, tunneling=1, interaction=2, periodic=True)
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -2.828427124746)


def test_FH_model_with_U_mu():
    """Test Fermi-Hubbard model Hamiltonian with onsite interaction and chemical
    potential."""

    # open boundary conditions
    op = fermi_hubbard(norb=4, tunneling=1, interaction=2, chemical_potential=3)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -14.875942809005)

    # periodic boundary conditions
    op_periodic = fermi_hubbard(
        norb=4, tunneling=1, interaction=2, chemical_potential=3, periodic=True
    )
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -14.828427124746)


def test_FH_model_with_U_mu_V():
    """Test Fermi-Hubbard model Hamiltonian with onsite interaction, chemical potential,
    and nearest-neighbor interaction."""

    # open boundary conditions
    op = fermi_hubbard(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
    )
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -9.961978205599)

    # periodic boundary conditions
    op_periodic = fermi_hubbard(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
        periodic=True,
    )
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -8.781962448006)


def test_FH_model_with_unequal_filling():
    """Test Fermi-Hubbard model Hamiltonian with unequal filling."""

    # open boundary conditions
    op = fermi_hubbard(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
    )
    ham = ffsim.linear_operator(op, norb=4, nelec=(1, 3))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -6.615276287167)

    # periodic boundary conditions
    op_periodic = fermi_hubbard(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
        periodic=True,
    )
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(1, 3))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -0.828427124746)


def test_FH_model_hermiticity():
    """Test Fermi-Hubbard model Hamiltonian hermiticity."""

    n_orbitals, n_alpha, n_beta = 4, 3, 1
    dim = comb(n_orbitals, n_alpha) * comb(n_orbitals, n_beta)

    # open boundary conditions
    op = fermi_hubbard(
        norb=n_orbitals,
        tunneling=0.1,
        interaction=0.2,
        chemical_potential=0.3,
        nearest_neighbor_interaction=0.4,
    )
    ham = ffsim.linear_operator(op, norb=n_orbitals, nelec=(n_alpha, n_beta))
    np.testing.assert_allclose(ham.dot(np.eye(dim)), ham.H.dot(np.eye(dim)))

    # periodic boundary conditions
    op_periodic = fermi_hubbard(
        norb=n_orbitals,
        tunneling=0.1,
        interaction=0.2,
        chemical_potential=0.3,
        nearest_neighbor_interaction=0.4,
        periodic=True,
    )
    ham_periodic = ffsim.linear_operator(
        op_periodic, norb=n_orbitals, nelec=(n_alpha, n_beta)
    )
    np.testing.assert_allclose(
        ham_periodic.dot(np.eye(dim)), ham_periodic.H.dot(np.eye(dim))
    )
