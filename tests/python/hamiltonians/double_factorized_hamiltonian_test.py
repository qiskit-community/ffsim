# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for double-factorized Hamiltonian."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import ffsim

RNG = np.random.default_rng(73020557092643299885827653110468818597)


@pytest.mark.parametrize("z_representation", [False, True])
def test_linear_operator(z_representation: bool):
    """Test linear operator."""
    norb = 4
    nelec = (2, 2)

    dim = ffsim.dim(norb, nelec)
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(
        norb, seed=RNG, dtype=float
    )

    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian,
        z_representation=z_representation,
    )

    actual_linop = ffsim.linear_operator(df_hamiltonian, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    dim = ffsim.dim(norb, nelec)

    state = ffsim.random.random_state_vector(dim, seed=RNG)
    actual = actual_linop @ state
    expected = expected_linop @ state
    np.testing.assert_allclose(actual, expected)

    state = ffsim.random.random_state_vector(dim, seed=RNG, dtype=float)
    actual = actual_linop @ state
    expected = expected_linop @ state
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("norb", range(5))
def test_z_representation_round_trip(norb: int):
    """Test converting to and from Z-representation"""
    mol_ham = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_ham
    )
    df_hamiltonian_roundtrip = (
        df_hamiltonian.to_z_representation().to_number_representation()
    )
    assert ffsim.approx_eq(df_hamiltonian_roundtrip, df_hamiltonian)


@pytest.mark.parametrize("z_representation", [False, True])
def test_to_molecular_hamiltonian_roundtrip(z_representation: bool):
    """Test converting to molecular Hamiltonian."""
    norb = 5
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    hamiltonian_roundtrip = (
        ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
            hamiltonian, z_representation=z_representation
        ).to_molecular_hamiltonian()
    )
    assert ffsim.approx_eq(hamiltonian_roundtrip, hamiltonian)


@pytest.mark.parametrize("z_representation", [False, True])
def test_to_molecular_hamiltonian_symmetry(z_representation: bool):
    """Test molecular Hamiltonian symmetry."""
    norb = 5
    df_hamiltonian = ffsim.random.random_double_factorized_hamiltonian(
        norb, z_representation=z_representation, seed=RNG
    )
    mol_hamiltonian = df_hamiltonian.to_molecular_hamiltonian()
    two_body_tensor = mol_hamiltonian.two_body_tensor
    for i, j, k, ell in itertools.product(range(norb), repeat=4):
        val = two_body_tensor[i, j, k, ell]
        np.testing.assert_allclose(two_body_tensor[k, ell, i, j], val)
        np.testing.assert_allclose(two_body_tensor[j, i, ell, k], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[ell, k, j, i], val.conjugate())


def test_diag():
    """Test computing diagonal."""
    norb = 5
    nelec = (3, 2)
    # TODO remove dtype=float once complex is supported
    hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    )
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    hamiltonian_dense = linop @ np.eye(ffsim.dim(norb, nelec))
    np.testing.assert_allclose(
        ffsim.diag(hamiltonian, norb=norb, nelec=nelec), np.diag(hamiltonian_dense)
    )


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
    """Test FermionOperator."""
    df_hamiltonian = ffsim.random.random_double_factorized_hamiltonian(
        norb, real=True, seed=RNG
    )
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    op = ffsim.fermion_operator(df_hamiltonian)
    linop = ffsim.linear_operator(op, norb, nelec)
    expected_linop = ffsim.linear_operator(df_hamiltonian, norb, nelec)

    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)
