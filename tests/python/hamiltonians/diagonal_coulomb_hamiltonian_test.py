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

import itertools

import numpy as np
import pytest
import scipy

import ffsim
from ffsim.hamiltonians.diagonal_coulomb_hamiltonian import DiagonalCoulombHamiltonian
from ffsim import fermi_hubbard_1d, fermi_hubbard_2d


# @pytest.mark.parametrize(
#     "norb, nelec",
#     [
#         (4, (2, 2)),
#         (4, (1, 2)),
#         (4, (0, 2)),
#         (4, (0, 0)),
#     ],
# )
# def test_linear_operator(norb: int, nelec: tuple[int, int]):
#     """Test linear_operator method."""
#     rng = np.random.default_rng()
#     dim = ffsim.dim(norb, nelec)
#
#     one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
#     diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
#         norb, seed=rng, dtype=float
#     )
#
#     constant = rng.standard_normal()
#     dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
#         one_body_tensor, diag_coulomb_mat, constant
#     )
#     df_hamiltonian = ffsim.DoubleFactorizedHamiltonian(
#         one_body_tensor, np.array([diag_coulomb_mat]),
#         np.array([np.eye(norb)]), constant, z_representation=False,
#     )
#
#     actual_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)
#     expected_linop = ffsim.linear_operator(df_hamiltonian, norb, nelec)
#
#     vec = ffsim.random.random_statevector(dim, seed=rng)
#     actual = actual_linop @ vec
#     expected = expected_linop @ vec
#     np.testing.assert_allclose(actual, expected)


# @pytest.mark.parametrize(
#     "norb, nelec",
#     [
#         (4, (2, 2)),
#         (4, (1, 2)),
#         (4, (0, 2)),
#         (4, (0, 0)),
#     ],
# )
# def test_fermion_operator(norb: int, nelec: tuple[int, int]):
#     """Test fermion_operator method."""
#     rng = np.random.default_rng()
#     dim = ffsim.dim(norb, nelec)
#
#     one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
#     diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
#         norb, seed=rng, dtype=float
#     )
#
#     constant = rng.standard_normal()
#     dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
#         one_body_tensor, diag_coulomb_mat, constant=constant
#     )
#
#     op = ffsim.fermion_operator(dc_hamiltonian)
#     actual_linop = ffsim.linear_operator(op, norb, nelec)
#     expected_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)
#
#     vec = ffsim.random.random_statevector(dim, seed=rng)
#     actual = actual_linop @ vec
#     expected = expected_linop @ vec
#     np.testing.assert_allclose(actual, expected)


# @pytest.mark.parametrize(
#     "norb, nelec",
#     [
#         (4, (2, 2)),
#         (4, (1, 2)),
#         (4, (0, 2)),
#         (4, (0, 0)),
#     ],
# )
# def test_from_fermion_operator(norb: int, nelec: tuple[int, int]):
#     """Test from_fermion_operator method."""
#     rng = np.random.default_rng()
#     dim = ffsim.dim(norb, nelec)
#
#     one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
#     diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
#         norb, seed=rng, dtype=float
#     )
#
#     constant = rng.standard_normal()
#     dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
#         one_body_tensor, diag_coulomb_mat, constant=constant
#     )
#
#     op = ffsim.fermion_operator(dc_hamiltonian)
#
#     dc_hamiltonian_from_op = DiagonalCoulombHamiltonian.from_fermion_operator(op)
#
#     # print("dc_hamiltonian = ", dc_hamiltonian)
#     # print("dc_hamiltonian from op = ", dc_hamiltonian_from_op)
#
#     actual_linop = ffsim.linear_operator(dc_hamiltonian_from_op, norb, nelec)
#     expected_linop = ffsim.linear_operator(op, norb, nelec)
#
#     vec = ffsim.random.random_statevector(dim, seed=rng)
#     actual = actual_linop @ vec
#     expected = expected_linop @ vec
#     np.testing.assert_allclose(actual, expected)


# @pytest.mark.parametrize(
#     "norb, nelec",
#     [
#         (4, (2, 2)),
#         (4, (1, 2)),
#         (4, (0, 2)),
#         (4, (0, 0)),
#     ],
# )
# def test_from_fermion_operator_failure(norb: int, nelec: tuple[int, int]):
#     """Test from_fermion_operator method failure."""
#     rng = np.random.default_rng()
#
#     one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
#     two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
#
#     constant = rng.standard_normal()
#     mol_hamiltonian = ffsim.MolecularHamiltonian(
#         one_body_tensor, two_body_tensor, constant=constant
#     )
#
#     op = ffsim.fermion_operator(mol_hamiltonian)
#
#     with pytest.raises(ValueError,
#                        match="FermionOperator cannot be converted to "
#                              "DiagonalCoulombHamiltonian"):
#         DiagonalCoulombHamiltonian.from_fermion_operator(op)


def test_from_fermion_operator_fermi_hubbard_1d():
    """Test from_fermion_operator method with the fermi_hubbard_1d model."""
    norb = 4
    nelec = (2, 2)
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(dim, seed=rng)

    # open boundary conditions
    op = fermi_hubbard_1d(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=0,
        nearest_neighbor_interaction=0,
    )

    print("op = ", op)

    dc_hamiltonian_from_op = DiagonalCoulombHamiltonian.from_fermion_operator(op)

    new_op = ffsim.fermion_operator(dc_hamiltonian_from_op)

    print("new op = ", new_op)

    print(dc_hamiltonian_from_op)

    actual_linop = ffsim.linear_operator(dc_hamiltonian_from_op, norb, nelec)
    actual_eigs, _ = scipy.sparse.linalg.eigsh(actual_linop, which="SA", k=1)
    print("actual energy = ", actual_eigs[0])

    expected_linop = ffsim.linear_operator(op, norb, nelec)
    eigs, _ = scipy.sparse.linalg.eigsh(expected_linop, which="SA", k=1)
    np.testing.assert_allclose(eigs[0], -2.875942809005)

    print("post test")

    actual = actual_linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)
#
#     # periodic boundary conditions
#     op_periodic = fermi_hubbard_1d(
#                 norb=4,
#                 tunneling=1,
#                 interaction=2,
#                 chemical_potential=3,
#                 nearest_neighbor_interaction=4,
#                 periodic=True,
#     )
#
#     dc_hamiltonian_from_op_periodic = (
#         DiagonalCoulombHamiltonian.from_fermion_operator(op_periodic))
#     actual_linop_periodic = (
#         ffsim.linear_operator(dc_hamiltonian_from_op_periodic, norb, nelec))
#     expected_linop_periodic = ffsim.linear_operator(op_periodic, norb, nelec)
#
#     actual_periodic = actual_linop_periodic @ vec
#     expected_periodic = expected_linop_periodic @ vec
#     np.testing.assert_allclose(actual_periodic, expected_periodic)


# def test_from_fermion_operator_fermi_hubbard_2d():
#     """Test from_fermion_operator method with the fermi_hubbard_2d model."""
#     norb = 4
#     nelec = (2, 2)
#     dim = ffsim.dim(norb, nelec)
#     rng = np.random.default_rng()
#     vec = ffsim.random.random_statevector(dim, seed=rng)
#
#     # open boundary conditions
#     op = fermi_hubbard_2d(
#         norb_x=2,
#         norb_y=2,
#         tunneling=1,
#         interaction=2,
#         chemical_potential=3,
#         nearest_neighbor_interaction=4,
#     )
#
#     dc_hamiltonian_from_op = DiagonalCoulombHamiltonian.from_fermion_operator(op)
#     actual_linop = ffsim.linear_operator(dc_hamiltonian_from_op, norb, nelec)
#     expected_linop = ffsim.linear_operator(op, norb, nelec)
#
#     actual = actual_linop @ vec
#     expected = expected_linop @ vec
#     np.testing.assert_allclose(actual, expected)
#
#     # periodic boundary conditions
#     op_periodic = fermi_hubbard_2d(
#         norb_x=2,
#         norb_y=2,
#         tunneling=1,
#         interaction=2,
#         chemical_potential=3,
#         nearest_neighbor_interaction=4,
#         periodic=True,
#     )
#
#     dc_hamiltonian_from_op_periodic = (
#         DiagonalCoulombHamiltonian.from_fermion_operator(op_periodic))
#     actual_linop_periodic = (
#         ffsim.linear_operator(dc_hamiltonian_from_op_periodic, norb, nelec))
#     expected_linop_periodic = ffsim.linear_operator(op_periodic, norb, nelec)
#
#     actual_periodic = actual_linop_periodic @ vec
#     expected_periodic = expected_linop_periodic @ vec
#     np.testing.assert_allclose(actual_periodic, expected_periodic)


########################################################################################
# OLD
########################################################################################


# @pytest.mark.parametrize(
#     "norb, nelec",
#     [
#         (4, (2, 2)),
#         (4, (1, 2)),
#         (4, (0, 2)),
#         (4, (0, 0)),
#     ],
# )
# def test_linear_operator(norb: int, nelec: tuple[int, int]):
#     """Test linear_operator method."""
#     rng = np.random.default_rng()
#     dim = ffsim.dim(norb, nelec)
#
#     one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
#     diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
#         norb, seed=rng, dtype=float
#     )
#
#     # convert diag_coulomb_mat -> two_body_tensor
#     two_body_tensor = np.zeros((norb, norb, norb, norb), dtype=float)
#     for p, q in itertools.product(range(norb), repeat=2):
#         two_body_tensor[p, p, q, q] = diag_coulomb_mat[p, q]
#
#     constant = rng.standard_normal()
#     dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
#         one_body_tensor, diag_coulomb_mat, constant
#     )
#     mol_hamiltonian = ffsim.MolecularHamiltonian(
#         one_body_tensor, two_body_tensor, constant
#     )
#
#     actual_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)
#     expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)
#
#     vec = ffsim.random.random_statevector(dim, seed=rng)
#     actual = actual_linop @ vec
#     expected = expected_linop @ vec
#     np.testing.assert_allclose(actual, expected)
