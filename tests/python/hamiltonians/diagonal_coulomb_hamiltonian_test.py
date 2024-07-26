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
import pytest

import ffsim
from ffsim import fermi_hubbard_1d, fermi_hubbard_2d
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
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
        norb, seed=rng, dtype=float
    )
    diag_coulomb_mats = np.array([diag_coulomb_mat, diag_coulomb_mat])
    constant = rng.standard_normal()

    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mats, constant
    )
    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian(
        one_body_tensor,
        np.array([diag_coulomb_mat]),
        np.array([np.eye(norb)]),
        constant,
        z_representation=False,
    )

    actual_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)
    expected_linop = ffsim.linear_operator(df_hamiltonian, norb, nelec)

    vec = ffsim.random.random_state_vector(dim, seed=rng)
    actual = actual_linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)

    vec = ffsim.random.random_state_vector(dim, seed=rng, dtype=float)
    actual = actual_linop @ vec
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
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    diag_coulomb_mats = np.empty((2, norb, norb), dtype=float)
    for i in range(2):
        diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
            norb, seed=rng, dtype=float
        )
        diag_coulomb_mats[i] = diag_coulomb_mat
    constant = rng.standard_normal()

    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mats, constant=constant
    )

    op = ffsim.fermion_operator(dc_hamiltonian)
    actual_linop = ffsim.linear_operator(op, norb, nelec)
    expected_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)

    vec = ffsim.random.random_state_vector(dim, seed=rng)
    actual = actual_linop @ vec
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
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    diag_coulomb_mats = np.empty((2, norb, norb), dtype=float)
    for i in range(2):
        diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
            norb, seed=rng, dtype=float
        )
        diag_coulomb_mats[i] = diag_coulomb_mat
    constant = rng.standard_normal()

    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mats, constant=constant
    )

    op = ffsim.fermion_operator(dc_hamiltonian)
    dc_hamiltonian_from_op = DiagonalCoulombHamiltonian.from_fermion_operator(op)
    actual_linop = ffsim.linear_operator(dc_hamiltonian_from_op, norb, nelec)
    expected_linop = ffsim.linear_operator(op, norb, nelec)

    vec = ffsim.random.random_state_vector(dim, seed=rng)
    actual = actual_linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)


def test_from_fermion_operator_failure():
    """Test from_fermion_operator method failure."""
    norb = 4
    rng = np.random.default_rng()

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    constant = rng.standard_normal()

    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant=constant
    )

    op = ffsim.fermion_operator(mol_hamiltonian)

    with pytest.raises(
        ValueError,
        match="FermionOperator cannot be converted to DiagonalCoulombHamiltonian",
    ):
        DiagonalCoulombHamiltonian.from_fermion_operator(op)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
def test_from_fermion_operator_fermi_hubbard_1d(norb: int, nelec: tuple[int, int]):
    """Test from_fermion_operator method with the fermi_hubbard_1d model."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = ffsim.random.random_state_vector(dim, seed=rng)

    # open boundary conditions
    op = fermi_hubbard_1d(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
    )

    dc_hamiltonian_from_op = DiagonalCoulombHamiltonian.from_fermion_operator(op)
    actual_linop = ffsim.linear_operator(dc_hamiltonian_from_op, norb, nelec)
    expected_linop = ffsim.linear_operator(op, norb, nelec)

    actual = actual_linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)

    # periodic boundary conditions
    op_periodic = fermi_hubbard_1d(
        norb=4,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
        periodic=True,
    )

    dc_hamiltonian_from_op_periodic = DiagonalCoulombHamiltonian.from_fermion_operator(
        op_periodic
    )
    actual_linop_periodic = ffsim.linear_operator(
        dc_hamiltonian_from_op_periodic, norb, nelec
    )
    expected_linop_periodic = ffsim.linear_operator(op_periodic, norb, nelec)

    actual_periodic = actual_linop_periodic @ vec
    expected_periodic = expected_linop_periodic @ vec
    np.testing.assert_allclose(actual_periodic, expected_periodic)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
def test_from_fermion_operator_fermi_hubbard_2d(norb: int, nelec: tuple[int, int]):
    """Test from_fermion_operator method with the fermi_hubbard_2d model."""
    dim = ffsim.dim(norb, nelec)
    rng = np.random.default_rng()
    vec = ffsim.random.random_state_vector(dim, seed=rng)

    # open boundary conditions
    op = fermi_hubbard_2d(
        norb_x=2,
        norb_y=2,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
    )

    dc_hamiltonian_from_op = DiagonalCoulombHamiltonian.from_fermion_operator(op)
    actual_linop = ffsim.linear_operator(dc_hamiltonian_from_op, norb, nelec)
    expected_linop = ffsim.linear_operator(op, norb, nelec)

    actual = actual_linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)

    # periodic boundary conditions
    op_periodic = fermi_hubbard_2d(
        norb_x=2,
        norb_y=2,
        tunneling=1,
        interaction=2,
        chemical_potential=3,
        nearest_neighbor_interaction=4,
        periodic=True,
    )

    dc_hamiltonian_from_op_periodic = DiagonalCoulombHamiltonian.from_fermion_operator(
        op_periodic
    )
    actual_linop_periodic = ffsim.linear_operator(
        dc_hamiltonian_from_op_periodic, norb, nelec
    )
    expected_linop_periodic = ffsim.linear_operator(op_periodic, norb, nelec)

    actual_periodic = actual_linop_periodic @ vec
    expected_periodic = expected_linop_periodic @ vec
    np.testing.assert_allclose(actual_periodic, expected_periodic)


def test_diag():
    """Test computing diagonal."""
    rng = np.random.default_rng(2222)
    norb = 5
    nelec = (3, 2)
    # TODO test complex one-body after adding support for it
    one_body_tensor = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb_mat_a = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb_mat_b = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb_mats = np.stack([diag_coulomb_mat_a, diag_coulomb_mat_b])
    constant = rng.standard_normal()
    hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mats, constant=constant
    )
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    hamiltonian_dense = linop @ np.eye(ffsim.dim(norb, nelec))
    np.testing.assert_allclose(
        ffsim.diag(hamiltonian, norb=norb, nelec=nelec), np.diag(hamiltonian_dense)
    )
