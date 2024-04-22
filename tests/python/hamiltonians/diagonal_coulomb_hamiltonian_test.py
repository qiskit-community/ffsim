# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for diagonal Coulomb Hamiltonian."""

from __future__ import annotations

import numpy as np

import ffsim
from ffsim.hamiltonians.diagonal_coulomb_hamiltonian import DiagonalCoulombHamiltonian
from ffsim.operators.fermi_hubbard import fermi_hubbard_1d


def test_from_fermion_operator():

    #########################
    # one_body_tensor tests #
    #########################

    # non-interacting one-dimensional Fermi-Hubbard model Hamiltonian
    operator = fermi_hubbard_1d(norb=4, tunneling=1, interaction=0)
    hamiltonian = ffsim.linear_operator(operator, norb=4, nelec=(1, 0))
    one_body_tensor_1 = hamiltonian.matmat(np.eye(4))
    one_body_tensor_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
        operator
    ).one_body_tensor
    np.testing.assert_allclose(one_body_tensor_1, one_body_tensor_2)

    # one-dimensional Fermi-Hubbard model Hamiltonian with onsite interaction
    operator = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2)
    hamiltonian = ffsim.linear_operator(operator, norb=4, nelec=(1, 0))
    one_body_tensor_1 = hamiltonian.matmat(np.eye(4))
    one_body_tensor_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
        operator
    ).one_body_tensor
    np.testing.assert_allclose(one_body_tensor_1, one_body_tensor_2)

    # one-dimensional Fermi-Hubbard model Hamiltonian with onsite interaction and
    # chemical potential
    operator = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2,
                                chemical_potential=3)
    hamiltonian = ffsim.linear_operator(operator, norb=4, nelec=(1, 0))
    one_body_tensor_1 = hamiltonian.matmat(np.eye(4))
    one_body_tensor_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
        operator
    ).one_body_tensor
    np.testing.assert_allclose(one_body_tensor_1, one_body_tensor_2)

    # one-dimensional Fermi-Hubbard model Hamiltonian with onsite interaction, chemical
    # potential, and nearest-neighbor interaction
    operator = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2,
                                chemical_potential=3, nearest_neighbor_interaction=4)
    hamiltonian = ffsim.linear_operator(operator, norb=4, nelec=(1, 0))
    one_body_tensor_1 = hamiltonian.matmat(np.eye(4))
    one_body_tensor_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
        operator
    ).one_body_tensor
    np.testing.assert_allclose(one_body_tensor_1, one_body_tensor_2)
