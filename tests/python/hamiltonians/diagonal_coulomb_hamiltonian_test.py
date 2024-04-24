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

import pyscf
import pyscf.mcscf
import scipy.sparse.linalg
import pytest
import ffsim
from ffsim.hamiltonians.diagonal_coulomb_hamiltonian import DiagonalCoulombHamiltonian
from ffsim.operators.fermi_hubbard import fermi_hubbard_1d


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (3, (1, 0)),
    ],
)
def test_fermion_operator(norb: int, nelec: tuple[int, int]):
    """Test FermionOperator."""
    rng = np.random.default_rng()

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    print("random one_body_tensor = ", one_body_tensor)

    diag_coulomb_mat = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    print("random diag_coulomb_mat = ", diag_coulomb_mat)

    constant = rng.standard_normal()
    print("random constant = ", constant)

    diag_coulomb_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mat, constant=constant
    )
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    op = ffsim.fermion_operator(diag_coulomb_hamiltonian)

    linop = ffsim.linear_operator(op, norb, nelec)
    expected_linop = ffsim.linear_operator(diag_coulomb_hamiltonian, norb, nelec)

    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)


# def test_from_fermion_operator():
#
#     #########################
#     # one_body_tensor tests #
#     #########################
#
#     # non-interacting one-dimensional Fermi-Hubbard model Hamiltonian
#     operator = fermi_hubbard_1d(norb=4, tunneling=1, interaction=0)
#     hamiltonian = ffsim.linear_operator(operator, norb=4, nelec=(1, 0))
#     one_body_tensor_1 = hamiltonian.matmat(np.eye(4))
#     one_body_tensor_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
#         operator
#     ).one_body_tensor
#     np.testing.assert_allclose(one_body_tensor_1, one_body_tensor_2)
#
#     # one-dimensional Fermi-Hubbard model Hamiltonian with onsite interaction
#     operator = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2)
#     hamiltonian = ffsim.linear_operator(operator, norb=4, nelec=(1, 0))
#     one_body_tensor_1 = hamiltonian.matmat(np.eye(4))
#     one_body_tensor_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
#         operator
#     ).one_body_tensor
#     np.testing.assert_allclose(one_body_tensor_1, one_body_tensor_2)
#
#     # one-dimensional Fermi-Hubbard model Hamiltonian with onsite interaction and
#     # chemical potential
#     operator = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2,
#                                 chemical_potential=3)
#     hamiltonian = ffsim.linear_operator(operator, norb=4, nelec=(1, 0))
#     one_body_tensor_1 = hamiltonian.matmat(np.eye(4))
#     one_body_tensor_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
#         operator
#     ).one_body_tensor
#     np.testing.assert_allclose(one_body_tensor_1, one_body_tensor_2)
#
#     # one-dimensional Fermi-Hubbard model Hamiltonian with onsite interaction, chemical
#     # potential, and nearest-neighbor interaction
#     operator = fermi_hubbard_1d(norb=4, tunneling=1, interaction=2,
#                                 chemical_potential=3, nearest_neighbor_interaction=4)
#     hamiltonian = ffsim.linear_operator(operator, norb=4, nelec=(1, 0))
#     one_body_tensor_1 = hamiltonian.matmat(np.eye(4))
#     one_body_tensor_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
#         operator
#     ).one_body_tensor
#     np.testing.assert_allclose(one_body_tensor_1, one_body_tensor_2)
