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
import itertools

import pyscf
import pyscf.mcscf
import scipy.sparse.linalg
import pytest
import ffsim
from ffsim.hamiltonians.diagonal_coulomb_hamiltonian import DiagonalCoulombHamiltonian


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
def test_linear_operator(norb: int, nelec: tuple[int, int]):
    """Test linear_operator method."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    real_sym_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng, dtype=float)

    # expand dimensions of real_sym_mat
    diag_coulomb_mat = np.zeros((norb, norb, norb, norb), dtype=float)
    for p, q in itertools.product(range(norb), repeat=2):
        diag_coulomb_mat[p, p, q, q] = real_sym_mat[p, q]

    constant = rng.standard_normal()
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, diag_coulomb_mat, constant
    )
    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mat, constant
    )

    linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)

    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    vec = ffsim.random.random_statevector(dim, seed=rng)
    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
def test_fermion_operator(norb: int, nelec: tuple[int, int]):
    """Test fermion_operator method."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    real_sym_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng, dtype=float)

    # expand dimensions of real_sym_mat
    diag_coulomb_mat = np.zeros((norb, norb, norb, norb), dtype=float)
    for p, q in itertools.product(range(norb), repeat=2):
        diag_coulomb_mat[p, p, q, q] = real_sym_mat[p, q]

    constant = rng.standard_normal()
    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mat, constant=constant
    )

    op = ffsim.fermion_operator(dc_hamiltonian)
    linop = ffsim.linear_operator(op, norb, nelec)

    expected_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)

    vec = ffsim.random.random_statevector(dim, seed=rng)
    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
def test_from_fermion_operator(norb: int, nelec: tuple[int, int]):
    """Test from_fermion_operator method."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    real_sym_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng, dtype=float)

    # expand dimensions of real_sym_mat
    diag_coulomb_mat = np.zeros((norb, norb, norb, norb), dtype=float)
    for p, q in itertools.product(range(norb), repeat=2):
        diag_coulomb_mat[p, p, q, q] = real_sym_mat[p, q]

    constant = rng.standard_normal()
    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mat, constant=constant
    )

    op = ffsim.fermion_operator(dc_hamiltonian)
    expected_linop = ffsim.linear_operator(op, norb, nelec)

    dc_hamiltonian_from_op = DiagonalCoulombHamiltonian.from_fermion_operator(op)
    linop = ffsim.linear_operator(dc_hamiltonian_from_op, norb, nelec)

    vec = ffsim.random.random_statevector(dim, seed=rng)
    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)


############
# old code #
############

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


# print("overall_tensor_1 = ", overall_tensor_1)
#
# one_body_tensor_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
#     operator
# ).one_body_tensor
# diag_coulomb_mat_2 = DiagonalCoulombHamiltonian.from_fermion_operator(
#     operator
# ).diag_coulomb_mat
#
# hamiltonian_2 = DiagonalCoulombHamiltonian.from_fermion_operator(operator)
# exp_linop = ffsim.linear_operator(hamiltonian_2, norb=4, nelec=(1, 1))
# overall_tensor_2 = hamiltonian_2.matvec(np.eye(16))
#
# print("overall_tensor_2 = ", overall_tensor_2)
#
# # print("one_body_tensor_2 = ", one_body_tensor_2)
# # print("diag_coulomb_mat_2 = ", diag_coulomb_mat_2)
#
# np.testing.assert_allclose(diag_coulomb_mat_1, diag_coulomb_mat_2)


# # non-interacting one-dimensional Fermi-Hubbard model Hamiltonian
    # norb = 4
    # nelec = (1, 0)
    # rng = np.random.default_rng()
    # dim = ffsim.dim(norb, nelec)
    #
    # operator = fermi_hubbard_1d(norb=norb, tunneling=1, interaction=0)
    # linop = ffsim.linear_operator(operator, norb=norb, nelec=nelec)
    #
    # dc_hamiltonian = DiagonalCoulombHamiltonian.from_fermion_operator(operator)
    # expected_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)
    #
    # vec = ffsim.random.random_statevector(dim, seed=rng)
    #
    # actual = linop @ vec
    # expected = expected_linop @ vec
    # np.testing.assert_allclose(actual, expected)

# def test_from_fermion_operator():
#     """Test from_fermion_operator."""
#     # non-interacting one-dimensional Fermi-Hubbard model Hamiltonian
#     norb = 4
#     nelec = (1, 0)
#     rng = np.random.default_rng()
#     dim = ffsim.dim(norb, nelec)
#
#     operator = fermi_hubbard_1d(norb=norb, tunneling=1, interaction=0)
#     linop = ffsim.linear_operator(operator, norb=norb, nelec=nelec)
#
#     dc_hamiltonian = DiagonalCoulombHamiltonian.from_fermion_operator(operator)
#     expected_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)
#
#     vec = ffsim.random.random_statevector(dim, seed=rng)
#
#     actual = linop @ vec
#     expected = expected_linop @ vec
#     np.testing.assert_allclose(actual, expected)
