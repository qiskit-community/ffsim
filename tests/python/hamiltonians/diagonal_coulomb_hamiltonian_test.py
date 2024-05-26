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
    diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
        norb, seed=rng, dtype=float
    )

    # convert diag_coulomb_mat -> two_body_tensor
    two_body_tensor = np.zeros((norb, norb, norb, norb), dtype=float)
    for p, q in itertools.product(range(norb), repeat=2):
        two_body_tensor[p, p, q, q] = diag_coulomb_mat[p, q]

    constant = rng.standard_normal()
    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mat, constant
    )
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant
    )

    actual_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    vec = ffsim.random.random_statevector(dim, seed=rng)
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
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
        norb, seed=rng, dtype=float
    )

    constant = rng.standard_normal()
    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mat, constant=constant
    )

    op = ffsim.fermion_operator(dc_hamiltonian)
    actual_linop = ffsim.linear_operator(op, norb, nelec)
    expected_linop = ffsim.linear_operator(dc_hamiltonian, norb, nelec)

    vec = ffsim.random.random_statevector(dim, seed=rng)
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
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
        norb, seed=rng, dtype=float
    )

    constant = rng.standard_normal()
    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mat, constant=constant
    )

    op = ffsim.fermion_operator(dc_hamiltonian)
    dc_hamiltonian_from_op = DiagonalCoulombHamiltonian.from_fermion_operator(op)
    actual_linop = ffsim.linear_operator(dc_hamiltonian_from_op, norb, nelec)
    expected_linop = ffsim.linear_operator(op, norb, nelec)

    vec = ffsim.random.random_statevector(dim, seed=rng)
    actual = actual_linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)
