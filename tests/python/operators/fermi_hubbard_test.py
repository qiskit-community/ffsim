# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the Fermi-Hubbard model Hamiltonian."""

from __future__ import annotations

import numpy as np
import scipy

import ffsim
from ffsim.operators.fermi_hubbard import fermi_hubbard_1d
from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b


def test_fermi_hubbard_1d():
    """Test terms of the one-dimensional Fermi-Hubbard model Hamiltonian."""

    # open boundary conditions
    op = fermi_hubbard_1d(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
    )
    np.testing.assert_equal(
        dict(op),
        {
            (cre_a(0), des_a(1)): -1,
            (cre_b(0), des_b(1)): -1,
            (cre_a(1), des_a(0)): -1,
            (cre_b(1), des_b(0)): -1,
            (cre_a(1), des_a(2)): -1,
            (cre_b(1), des_b(2)): -1,
            (cre_a(2), des_a(1)): -1,
            (cre_b(2), des_b(1)): -1,
            (cre_a(2), des_a(3)): -1,
            (cre_b(2), des_b(3)): -1,
            (cre_a(3), des_a(2)): -1,
            (cre_b(3), des_b(2)): -1,
            (cre_a(0), des_a(0), cre_b(0), des_b(0)): 2,
            (cre_a(1), des_a(1), cre_b(1), des_b(1)): 2,
            (cre_a(2), des_a(2), cre_b(2), des_b(2)): 2,
            (cre_a(3), des_a(3), cre_b(3), des_b(3)): 2,
            (cre_a(0), des_a(0)): -3,
            (cre_b(0), des_b(0)): -3,
            (cre_a(1), des_a(1)): -3,
            (cre_b(1), des_b(1)): -3,
            (cre_a(2), des_a(2)): -3,
            (cre_b(2), des_b(2)): -3,
            (cre_a(3), des_a(3)): -3,
            (cre_b(3), des_b(3)): -3,
            (cre_a(0), des_a(0), cre_a(1), des_a(1)): 4,
            (cre_a(0), des_a(0), cre_b(1), des_b(1)): 4,
            (cre_b(0), des_b(0), cre_a(1), des_a(1)): 4,
            (cre_b(0), des_b(0), cre_b(1), des_b(1)): 4,
            (cre_a(1), des_a(1), cre_a(2), des_a(2)): 4,
            (cre_a(1), des_a(1), cre_b(2), des_b(2)): 4,
            (cre_b(1), des_b(1), cre_a(2), des_a(2)): 4,
            (cre_b(1), des_b(1), cre_b(2), des_b(2)): 4,
            (cre_a(2), des_a(2), cre_a(3), des_a(3)): 4,
            (cre_a(2), des_a(2), cre_b(3), des_b(3)): 4,
            (cre_b(2), des_b(2), cre_a(3), des_a(3)): 4,
            (cre_b(2), des_b(2), cre_b(3), des_b(3)): 4,
        },
    )

    # periodic boundary conditions
    op_periodic = fermi_hubbard_1d(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
        periodic=True,
    )
    np.testing.assert_equal(
        dict(op_periodic),
        {
            (cre_a(0), des_a(1)): -1,
            (cre_b(0), des_b(1)): -1,
            (cre_a(1), des_a(0)): -1,
            (cre_b(1), des_b(0)): -1,
            (cre_a(1), des_a(2)): -1,
            (cre_b(1), des_b(2)): -1,
            (cre_a(2), des_a(1)): -1,
            (cre_b(2), des_b(1)): -1,
            (cre_a(2), des_a(3)): -1,
            (cre_b(2), des_b(3)): -1,
            (cre_a(3), des_a(2)): -1,
            (cre_b(3), des_b(2)): -1,
            (cre_a(3), des_a(0)): -1,
            (cre_b(3), des_b(0)): -1,
            (cre_a(0), des_a(3)): -1,
            (cre_b(0), des_b(3)): -1,
            (cre_a(0), des_a(0), cre_b(0), des_b(0)): 2,
            (cre_a(1), des_a(1), cre_b(1), des_b(1)): 2,
            (cre_a(2), des_a(2), cre_b(2), des_b(2)): 2,
            (cre_a(3), des_a(3), cre_b(3), des_b(3)): 2,
            (cre_a(0), des_a(0)): -3,
            (cre_b(0), des_b(0)): -3,
            (cre_a(1), des_a(1)): -3,
            (cre_b(1), des_b(1)): -3,
            (cre_a(2), des_a(2)): -3,
            (cre_b(2), des_b(2)): -3,
            (cre_a(3), des_a(3)): -3,
            (cre_b(3), des_b(3)): -3,
            (cre_a(0), des_a(0), cre_a(1), des_a(1)): 4,
            (cre_a(0), des_a(0), cre_b(1), des_b(1)): 4,
            (cre_b(0), des_b(0), cre_a(1), des_a(1)): 4,
            (cre_b(0), des_b(0), cre_b(1), des_b(1)): 4,
            (cre_a(1), des_a(1), cre_a(2), des_a(2)): 4,
            (cre_a(1), des_a(1), cre_b(2), des_b(2)): 4,
            (cre_b(1), des_b(1), cre_a(2), des_a(2)): 4,
            (cre_b(1), des_b(1), cre_b(2), des_b(2)): 4,
            (cre_a(2), des_a(2), cre_a(3), des_a(3)): 4,
            (cre_a(2), des_a(2), cre_b(3), des_b(3)): 4,
            (cre_b(2), des_b(2), cre_a(3), des_a(3)): 4,
            (cre_b(2), des_b(2), cre_b(3), des_b(3)): 4,
            (cre_a(3), des_a(3), cre_a(0), des_a(0)): 4,
            (cre_a(3), des_a(3), cre_b(0), des_b(0)): 4,
            (cre_b(3), des_b(3), cre_a(0), des_a(0)): 4,
            (cre_b(3), des_b(3), cre_b(0), des_b(0)): 4,
        },
    )

    # periodic boundary conditions (edge case)
    op_periodic_edge = fermi_hubbard_1d(
        norb=2,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
        periodic=True,
    )
    np.testing.assert_equal(
        dict(op_periodic_edge),
        {
            (cre_a(0), des_a(1)): -2,
            (cre_b(0), des_b(1)): -2,
            (cre_a(1), des_a(0)): -2,
            (cre_b(1), des_b(0)): -2,
            (cre_a(0), des_a(0), cre_b(0), des_b(0)): 2,
            (cre_a(1), des_a(1), cre_b(1), des_b(1)): 2,
            (cre_a(0), des_a(0)): -3,
            (cre_b(0), des_b(0)): -3,
            (cre_a(1), des_a(1)): -3,
            (cre_b(1), des_b(1)): -3,
            (cre_a(0), des_a(0), cre_a(1), des_a(1)): 4,
            (cre_a(0), des_a(0), cre_b(1), des_b(1)): 4,
            (cre_b(0), des_b(0), cre_a(1), des_a(1)): 4,
            (cre_b(0), des_b(0), cre_b(1), des_b(1)): 4,
            (cre_a(1), des_a(1), cre_a(0), des_a(0)): 4,
            (cre_a(1), des_a(1), cre_b(0), des_b(0)): 4,
            (cre_b(1), des_b(1), cre_a(0), des_a(0)): 4,
            (cre_b(1), des_b(1), cre_b(0), des_b(0)): 4,
        },
    )


def test_non_interacting_fermi_hubbard_1d_eigenvalue():
    """Test ground-state eigenvalue of the non-interacting one-dimensional Fermi-Hubbard
    model Hamiltonian."""

    # open boundary conditions
    op = fermi_hubbard_1d(norb=4, tunneling=1, interaction=0)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -4.472135955000)

    # periodic boundary conditions
    op_periodic = fermi_hubbard_1d(norb=4, tunneling=1, interaction=0, periodic=True)
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -4.000000000000)

    # periodic boundary conditions (edge case)
    op_periodic_edge = fermi_hubbard_1d(
        norb=2, tunneling=1, interaction=0, periodic=True
    )
    ham_periodic = ffsim.linear_operator(op_periodic_edge, norb=2, nelec=(1, 1))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -4.000000000000)


def test_fermi_hubbard_1d_with_interaction_eigenvalue():
    """Test ground-state eigenvalue of the one-dimensional Fermi-Hubbard model
    Hamiltonian with onsite interaction."""

    # open boundary conditions
    op = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -2.875942809005)

    # periodic boundary conditions
    op_periodic = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2, periodic=True)
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -2.828427124746)

    # periodic boundary conditions (edge case)
    op_periodic_edge = fermi_hubbard_1d(
        norb=2, tunneling=1, interaction=2, periodic=True
    )
    ham_periodic = ffsim.linear_operator(op_periodic_edge, norb=2, nelec=(1, 1))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -3.123105625618)


def test_fermi_hubbard_1d_with_chemical_potential_eigenvalue():
    """Test ground-state eigenvalue of the one-dimensional Fermi-Hubbard model
    Hamiltonian with onsite interaction and chemical potential."""

    # open boundary conditions
    op = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2, chemical_potential=3)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -14.875942809005)

    # periodic boundary conditions
    op_periodic = fermi_hubbard_1d(
        norb=4, tunneling=1, interaction=2, chemical_potential=3, periodic=True
    )
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -14.828427124746)

    # periodic boundary conditions (edge case)
    op_periodic_edge = fermi_hubbard_1d(
        norb=2, tunneling=1, interaction=2, chemical_potential=3, periodic=True
    )
    ham_periodic = ffsim.linear_operator(op_periodic_edge, norb=2, nelec=(1, 1))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -9.123105625618)


def test_fermi_hubbard_1d_with_nearest_neighbor_interaction_eigenvalue():
    """Test ground-state eigenvalue of the one-dimensional Fermi-Hubbard model
    Hamiltonian with onsite interaction, chemical potential, and nearest-neighbor
    interaction."""

    # open boundary conditions
    op = fermi_hubbard_1d(
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
    op_periodic = fermi_hubbard_1d(
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

    # periodic boundary conditions (edge case)
    op_periodic_edge = fermi_hubbard_1d(
        norb=2,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
        periodic=True,
    )
    ham_periodic = ffsim.linear_operator(op_periodic_edge, norb=2, nelec=(1, 1))
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -6.000000000000)


def test_fermi_hubbard_1d_with_unequal_filling_eigenvalue():
    """Test ground-state eigenvalue of the one-dimensional Fermi-Hubbard model
    Hamiltonian with unequal filling."""

    # open boundary conditions
    op = fermi_hubbard_1d(
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
    op_periodic = fermi_hubbard_1d(
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


def test_fermi_hubbard_1d_hermiticity():
    """Test hermiticity of the one-dimensional Fermi-Hubbard model Hamiltonian."""

    n_orbitals = 4
    n_electrons = (3, 1)
    dim = ffsim.dim(n_orbitals, n_electrons)

    # open boundary conditions
    op = fermi_hubbard_1d(
        norb=n_orbitals,
        tunneling=1.1,
        interaction=1.2,
        chemical_potential=1.3,
        nearest_neighbor_interaction=1.4,
    )
    ham = ffsim.linear_operator(op, norb=n_orbitals, nelec=n_electrons)
    np.testing.assert_allclose(ham @ np.eye(dim), ham.H @ np.eye(dim))

    # periodic boundary conditions
    op_periodic = fermi_hubbard_1d(
        norb=n_orbitals,
        tunneling=1.1,
        interaction=1.2,
        chemical_potential=1.3,
        nearest_neighbor_interaction=1.4,
        periodic=True,
    )
    ham_periodic = ffsim.linear_operator(
        op_periodic, norb=n_orbitals, nelec=n_electrons
    )
    np.testing.assert_allclose(ham_periodic @ np.eye(dim), ham_periodic.H @ np.eye(dim))
