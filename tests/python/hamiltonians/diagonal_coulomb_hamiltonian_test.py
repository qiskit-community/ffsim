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


# @pytest.mark.parametrize(
#     "norb, nelec",
#     [
#         (4, (2, 2)),
#     ],
# )
#
# def test_fermion_operator(norb: int, nelec: tuple[int, int]):
#     """Test FermionOperator."""
#     rng = np.random.default_rng()
#
#     one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
#     # TODO remove dtype=float after adding support for complex
#     diag_coulomb_mat = ffsim.random.random_hermitian(norb, seed=rng)
#     constant = rng.standard_normal()
#
#     print("one_body_tensor = ", one_body_tensor)
#     print("diag_coulomb_mat = ", diag_coulomb_mat)
#     print("constant = ", constant)
#
#     diag_coulomb_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
#         one_body_tensor, diag_coulomb_mat, constant=constant
#     )
#     vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)
#
#     op = ffsim.fermion_operator(diag_coulomb_hamiltonian)
#     linop = ffsim.linear_operator(op, norb, nelec)
#     expected_linop = ffsim.linear_operator(diag_coulomb_hamiltonian, norb, nelec)  #
#
#     actual = linop @ vec
#     expected = expected_linop @ vec
#     np.testing.assert_allclose(actual, expected)


# def test_rotated():
#     """Test rotating orbitals."""
#     norb = 5
#     nelec = (3, 2)
#
#     rng = np.random.default_rng()
#
#     # generate a random molecular Hamiltonian
#     one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
#     # TODO remove dtype=float after adding support for complex
#     diag_coulomb_mat = ffsim.random.random_diag_coulomb_mat(norb, seed=rng, dtype=float)
#     constant = rng.standard_normal()
#     diag_coulomb_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
#         one_body_tensor, diag_coulomb_mat, constant=constant
#     )
#
#     # generate a random orbital rotation
#     orbital_rotation = ffsim.random.random_orthogonal(norb, seed=rng)
#
#     # rotate the Hamiltonian
#     diag_coulomb_hamiltonian_rotated = diag_coulomb_hamiltonian.rotated(orbital_rotation)
#
#     # convert the original and rotated Hamiltonians to linear operators
#     linop = ffsim.linear_operator(diag_coulomb_hamiltonian, norb, nelec)
#     linop_rotated = ffsim.linear_operator(diag_coulomb_hamiltonian_rotated, norb, nelec)
#
#     # generate a random statevector
#     vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)
#
#     # rotate the statevector
#     rotated_vec = ffsim.apply_orbital_rotation(vec, orbital_rotation, norb, nelec)
#
#     # test definition
#     actual = linop_rotated @ vec
#     expected = ffsim.apply_orbital_rotation(vec, orbital_rotation.T.conj(), norb, nelec)
#     expected = linop @ expected
#     expected = ffsim.apply_orbital_rotation(expected, orbital_rotation, norb, nelec)
#     np.testing.assert_allclose(actual, expected)
#
#     # test expectation is preserved
#     original_expectation = np.vdot(vec, linop @ vec)
#     rotated_expectation = np.vdot(rotated_vec, linop_rotated @ rotated_vec)
#     np.testing.assert_allclose(original_expectation, rotated_expectation)


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
