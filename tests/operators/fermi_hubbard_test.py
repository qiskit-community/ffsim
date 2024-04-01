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

import numpy as np
import scipy

import ffsim
from ffsim._lib import FermionOperator
from ffsim.operators.fermi_hubbard import fermi_hubbard_1d
from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b


def test_non_interacting_fermi_hubbard_1d():
    """Test non-interacting one-dimensional Fermi-Hubbard model Hamiltonian."""

    # open boundary conditions
    op = fermi_hubbard_1d(norb=4, tunneling=1, interaction=0)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for p in range(3):
        coeffs[(cre_a(p), des_a((p + 1) % 4))] = -1
        coeffs[(cre_b(p), des_b((p + 1) % 4))] = -1
        coeffs[(cre_a((p + 1) % 4), des_a(p))] = -1
        coeffs[(cre_b((p + 1) % 4), des_b(p))] = -1
    np.testing.assert_equal(FermionOperator(coeffs), op)
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -4.472135955000)

    # periodic boundary conditions
    op_periodic = fermi_hubbard_1d(norb=4, tunneling=1, interaction=0, periodic=True)
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    coeffs_periodic: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for p in range(4):
        coeffs_periodic[(cre_a(p), des_a((p + 1) % 4))] = -1
        coeffs_periodic[(cre_b(p), des_b((p + 1) % 4))] = -1
        coeffs_periodic[(cre_a((p + 1) % 4), des_a(p))] = -1
        coeffs_periodic[(cre_b((p + 1) % 4), des_b(p))] = -1
    np.testing.assert_equal(FermionOperator(coeffs_periodic), op_periodic)
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -4.000000000000)


def test_fermi_hubbard_1d_with_interaction():
    """Test one-dimensional Fermi-Hubbard model Hamiltonian with onsite interaction."""

    # open boundary conditions
    op = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for p in range(3):
        coeffs[(cre_a(p), des_a((p + 1) % 4))] = -1
        coeffs[(cre_b(p), des_b((p + 1) % 4))] = -1
        coeffs[(cre_a((p + 1) % 4), des_a(p))] = -1
        coeffs[(cre_b((p + 1) % 4), des_b(p))] = -1
    for p in range(4):
        coeffs[(cre_a(p), des_a(p), cre_b(p), des_b(p))] = 1
        coeffs[(cre_b(p), des_b(p), cre_a(p), des_a(p))] = 1
    np.testing.assert_equal(FermionOperator(coeffs), op)
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -2.875942809005)

    # periodic boundary conditions
    op_periodic = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2, periodic=True)
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    coeffs_periodic: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for p in range(4):
        coeffs_periodic[(cre_a(p), des_a((p + 1) % 4))] = -1
        coeffs_periodic[(cre_b(p), des_b((p + 1) % 4))] = -1
        coeffs_periodic[(cre_a((p + 1) % 4), des_a(p))] = -1
        coeffs_periodic[(cre_b((p + 1) % 4), des_b(p))] = -1
        coeffs_periodic[(cre_a(p), des_a(p), cre_b(p), des_b(p))] = 1
        coeffs_periodic[(cre_b(p), des_b(p), cre_a(p), des_a(p))] = 1
    np.testing.assert_equal(FermionOperator(coeffs_periodic), op_periodic)
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -2.828427124746)


def test_fermi_hubbard_1d_with_chemical_potential():
    """Test one-dimensional Fermi-Hubbard model Hamiltonian with onsite interaction and
    chemical potential."""

    # open boundary conditions
    op = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2, chemical_potential=3)
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for p in range(3):
        coeffs[(cre_a(p), des_a((p + 1) % 4))] = -1
        coeffs[(cre_b(p), des_b((p + 1) % 4))] = -1
        coeffs[(cre_a((p + 1) % 4), des_a(p))] = -1
        coeffs[(cre_b((p + 1) % 4), des_b(p))] = -1
    for p in range(4):
        coeffs[(cre_a(p), des_a(p), cre_b(p), des_b(p))] = 1
        coeffs[(cre_b(p), des_b(p), cre_a(p), des_a(p))] = 1
        coeffs[(cre_a(p), des_a(p))] = -3
        coeffs[(cre_b(p), des_b(p))] = -3
    np.testing.assert_equal(FermionOperator(coeffs), op)
    eigs, _ = scipy.sparse.linalg.eigsh(ham, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -14.875942809005)

    # periodic boundary conditions
    op_periodic = fermi_hubbard_1d(
        norb=4, tunneling=1, interaction=2, chemical_potential=3, periodic=True
    )
    ham_periodic = ffsim.linear_operator(op_periodic, norb=4, nelec=(2, 2))
    coeffs_periodic: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for p in range(4):
        coeffs_periodic[(cre_a(p), des_a((p + 1) % 4))] = -1
        coeffs_periodic[(cre_b(p), des_b((p + 1) % 4))] = -1
        coeffs_periodic[(cre_a((p + 1) % 4), des_a(p))] = -1
        coeffs_periodic[(cre_b((p + 1) % 4), des_b(p))] = -1
        coeffs_periodic[(cre_a(p), des_a(p), cre_b(p), des_b(p))] = 1
        coeffs_periodic[(cre_b(p), des_b(p), cre_a(p), des_a(p))] = 1
        coeffs_periodic[(cre_a(p), des_a(p))] = -3
        coeffs_periodic[(cre_b(p), des_b(p))] = -3
    np.testing.assert_equal(FermionOperator(coeffs_periodic), op_periodic)
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -14.828427124746)


def test_fermi_hubbard_1d_with_nearest_neighbor_interaction():
    """Test one-dimensional Fermi-Hubbard model Hamiltonian with onsite interaction,
    chemical potential, and nearest-neighbor interaction."""

    # open boundary conditions
    op = fermi_hubbard_1d(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
    )
    ham = ffsim.linear_operator(op, norb=4, nelec=(2, 2))
    coeffs: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for p in range(3):
        coeffs[(cre_a(p), des_a((p + 1) % 4))] = -1
        coeffs[(cre_b(p), des_b((p + 1) % 4))] = -1
        coeffs[(cre_a((p + 1) % 4), des_a(p))] = -1
        coeffs[(cre_b((p + 1) % 4), des_b(p))] = -1
        coeffs[cre_a(p), des_a(p), cre_a((p + 1) % 4), des_a((p + 1) % 4)] = 4
        coeffs[cre_a(p), des_a(p), cre_b((p + 1) % 4), des_b((p + 1) % 4)] = 4
        coeffs[cre_b(p), des_b(p), cre_a((p + 1) % 4), des_a((p + 1) % 4)] = 4
        coeffs[cre_b(p), des_b(p), cre_b((p + 1) % 4), des_b((p + 1) % 4)] = 4
    for p in range(4):
        coeffs[(cre_a(p), des_a(p), cre_b(p), des_b(p))] = 1
        coeffs[(cre_b(p), des_b(p), cre_a(p), des_a(p))] = 1
        coeffs[(cre_a(p), des_a(p))] = -3
        coeffs[(cre_b(p), des_b(p))] = -3
    np.testing.assert_equal(FermionOperator(coeffs), op)
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
    coeffs_periodic: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for p in range(4):
        coeffs_periodic[(cre_a(p), des_a((p + 1) % 4))] = -1
        coeffs_periodic[(cre_b(p), des_b((p + 1) % 4))] = -1
        coeffs_periodic[(cre_a((p + 1) % 4), des_a(p))] = -1
        coeffs_periodic[(cre_b((p + 1) % 4), des_b(p))] = -1
        coeffs_periodic[cre_a(p), des_a(p), cre_a((p + 1) % 4), des_a((p + 1) % 4)] = 4
        coeffs_periodic[cre_a(p), des_a(p), cre_b((p + 1) % 4), des_b((p + 1) % 4)] = 4
        coeffs_periodic[cre_b(p), des_b(p), cre_a((p + 1) % 4), des_a((p + 1) % 4)] = 4
        coeffs_periodic[cre_b(p), des_b(p), cre_b((p + 1) % 4), des_b((p + 1) % 4)] = 4
        coeffs_periodic[(cre_a(p), des_a(p), cre_b(p), des_b(p))] = 1
        coeffs_periodic[(cre_b(p), des_b(p), cre_a(p), des_a(p))] = 1
        coeffs_periodic[(cre_a(p), des_a(p))] = -3
        coeffs_periodic[(cre_b(p), des_b(p))] = -3
    np.testing.assert_equal(FermionOperator(coeffs_periodic), op_periodic)
    eigs_periodic, _ = scipy.sparse.linalg.eigsh(ham_periodic, which="SA", k=1)
    np.testing.assert_allclose(eigs_periodic[0], -8.781962448006)


def test_fermi_hubbard_1d_with_unequal_filling():
    """Test one-dimensional Fermi-Hubbard model Hamiltonian with unequal filling."""

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
    """Test one-dimensional Fermi-Hubbard model Hamiltonian hermiticity."""

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
