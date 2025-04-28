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


@pytest.mark.parametrize("norb", range(1, 5))
def test_from_fermion_operator_random(norb: int):
    """Test initialization from FermionOperator with random Hamiltonian."""
    rng = np.random.default_rng()
    dc_hamiltonian = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=rng)
    op = ffsim.fermion_operator(dc_hamiltonian)
    roundtripped = ffsim.DiagonalCoulombHamiltonian.from_fermion_operator(op)
    assert ffsim.approx_eq(roundtripped, dc_hamiltonian)


def test_from_fermion_operator_errors():
    """Test from_fermion_operator raises errors correctly."""
    norb = 4
    rng = np.random.default_rng()

    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    op = ffsim.fermion_operator(mol_hamiltonian)
    with pytest.raises(ValueError, match="two-body"):
        _ = ffsim.DiagonalCoulombHamiltonian.from_fermion_operator(op)

    dc_hamiltonian = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=rng)
    op = ffsim.fermion_operator(dc_hamiltonian)
    op[()] += 1j
    with pytest.raises(ValueError, match="Constant"):
        _ = ffsim.DiagonalCoulombHamiltonian.from_fermion_operator(op)


@pytest.mark.parametrize("periodic", [False, True])
def test_from_fermion_operator_fermi_hubbard_1d(periodic: bool):
    """Test from_fermion_operator method with the fermi_hubbard_1d model."""
    rng = np.random.default_rng()
    tunneling, interaction, chemical_potential, nearest_neighbor_interaction = (
        rng.standard_normal(4)
    )
    op = ffsim.fermi_hubbard_1d(
        norb=4,
        tunneling=tunneling,
        interaction=interaction,
        chemical_potential=chemical_potential,
        nearest_neighbor_interaction=nearest_neighbor_interaction,
        periodic=periodic,
    )
    op.simplify()
    dc_ham = ffsim.DiagonalCoulombHamiltonian.from_fermion_operator(op)
    roundtripped = ffsim.fermion_operator(dc_ham)
    roundtripped.simplify()
    assert roundtripped.normal_ordered() == op.normal_ordered()


@pytest.mark.parametrize("periodic", [False, True])
def test_from_fermion_operator_fermi_hubbard_2d(periodic: bool):
    """Test from_fermion_operator method with the fermi_hubbard_2d model."""
    rng = np.random.default_rng()
    tunneling, interaction, chemical_potential, nearest_neighbor_interaction = (
        rng.standard_normal(4)
    )
    op = ffsim.fermi_hubbard_2d(
        norb_x=2,
        norb_y=2,
        tunneling=tunneling,
        interaction=interaction,
        chemical_potential=chemical_potential,
        nearest_neighbor_interaction=nearest_neighbor_interaction,
        periodic=periodic,
    )
    op.simplify()
    dc_ham = ffsim.DiagonalCoulombHamiltonian.from_fermion_operator(op)
    roundtripped = ffsim.fermion_operator(dc_ham)
    roundtripped.simplify()
    assert roundtripped.normal_ordered() == op.normal_ordered()


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
