# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for single-factorized Hamiltonian."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest

import ffsim

RNG = np.random.default_rng(169273111718651392627598708290856329828)


@pytest.mark.parametrize(
    "norb, nelec, cholesky",
    [
        (norb, nelec, z_representation)
        for (norb, nelec), z_representation in itertools.product(
            ffsim.testing.generate_norb_nelec(
                exhaustive=False, include_norb_zero=False
            ),
            [False, True],
        )
    ],
)
def test_linear_operator(norb: int, nelec: tuple[int, int], cholesky: bool):
    """Test linear operator."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(
        norb, seed=RNG, dtype=float
    )
    sf_hamiltonian = ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian, cholesky=cholesky
    )

    actual_linop = ffsim.linear_operator(sf_hamiltonian, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    dim = ffsim.dim(norb, nelec)
    state = ffsim.random.random_state_vector(dim, seed=RNG)

    actual = actual_linop @ state
    expected = expected_linop @ state

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "norb, nelec",
    ffsim.testing.generate_norb_nelec(exhaustive=False, include_norb_zero=False),
)
def test_reduced_matrix_product_states(norb: int, nelec: tuple[int, int]):
    """Test computing reduced matrix on product states."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(
        norb, seed=RNG, dtype=float
    )
    sf_hamiltonian = ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    linop = ffsim.linear_operator(sf_hamiltonian, norb, nelec)

    n_vecs = 3
    occupations = [
        ffsim.testing.random_occupied_orbitals(norb, nelec, seed=RNG)
        for _ in range(n_vecs)
    ]
    orbital_rotations = [
        ffsim.random.random_unitary(norb, seed=RNG) for _ in range(n_vecs)
    ]
    vecs = [
        ffsim.slater_determinant(norb, occ, orbital_rotation=orbital_rotation)
        for occ, orbital_rotation in zip(occupations, orbital_rotations)
    ]
    product_vecs = [
        (
            ffsim.slater_determinant(
                norb, (occ_a, []), orbital_rotation=orbital_rotation
            ),
            ffsim.slater_determinant(
                norb, ([], occ_b), orbital_rotation=orbital_rotation
            ),
        )
        for (occ_a, occ_b), orbital_rotation in zip(occupations, orbital_rotations)
    ]

    actual = sf_hamiltonian.reduced_matrix_product_states(product_vecs, norb, nelec)
    expected = ffsim.linalg.reduced_matrix(linop, vecs)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "norb, nelec",
    ffsim.testing.generate_norb_nelec(exhaustive=False, include_norb_zero=False),
)
def test_expectation_product_state_slater_determinant(
    norb: int, nelec: tuple[int, int]
):
    """Test computing expectation value on Slater determinant product state."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(
        norb, seed=RNG, dtype=float
    )
    sf_hamiltonian = ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    linop = ffsim.linear_operator(sf_hamiltonian, norb, nelec)

    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    for _ in range(5):
        (occ_a, occ_b) = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=RNG)
        vec = ffsim.slater_determinant(
            norb, (occ_a, occ_b), orbital_rotation=orbital_rotation
        )
        vec_a = ffsim.slater_determinant(
            norb, (occ_a, []), orbital_rotation=orbital_rotation
        )
        vec_b = ffsim.slater_determinant(
            norb, ([], occ_b), orbital_rotation=orbital_rotation
        )

        actual = sf_hamiltonian.expectation_product_state((vec_a, vec_b), norb, nelec)
        expected = np.vdot(vec, linop @ vec)

        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "norb, nelec",
    ffsim.testing.generate_norb_nelec(exhaustive=False, include_norb_zero=False),
)
def test_expectation_product_state(norb: int, nelec: tuple[int, int]):
    """Test computing expectation value on product state."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(
        norb, seed=RNG, dtype=float
    )
    sf_hamiltonian = ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    linop = ffsim.linear_operator(sf_hamiltonian, norb, nelec)

    dim_a, dim_b = ffsim.dims(norb, nelec)
    vec_a = ffsim.random.random_state_vector(dim_a, seed=RNG)
    vec_b = ffsim.random.random_state_vector(dim_b, seed=RNG)
    vec = np.kron(vec_a, vec_b)

    actual = sf_hamiltonian.expectation_product_state((vec_a, vec_b), norb, nelec)
    expected = np.vdot(vec, linop @ vec)

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("norb", range(5))
@pytest.mark.parametrize("cholesky", [False, True])
def test_to_molecular_hamiltonian_roundtrip(norb: int, cholesky: bool):
    """Test converting to molecular Hamiltonian."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    hamiltonian_roundtrip = (
        ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
            hamiltonian, cholesky=cholesky
        ).to_molecular_hamiltonian()
    )
    assert ffsim.approx_eq(hamiltonian_roundtrip, hamiltonian)


@pytest.mark.parametrize(
    "norb, nelec",
    ffsim.testing.generate_norb_nelec(exhaustive=False, include_norb_zero=False),
)
@pytest.mark.parametrize("real", [True, False])
def test_to_molecular_hamiltonian_linear_operator(
    norb: int, nelec: tuple[int, int], real: bool
):
    """Test converted molecular Hamiltonian has the same linear operator."""
    sf_hamiltonian = ffsim.random.random_single_factorized_hamiltonian(
        norb, real=real, seed=RNG
    )
    mol_hamiltonian = sf_hamiltonian.to_molecular_hamiltonian()
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    actual = ffsim.linear_operator(mol_hamiltonian, norb, nelec) @ vec
    expected = ffsim.linear_operator(sf_hamiltonian, norb, nelec) @ vec
    np.testing.assert_allclose(actual, expected)


def test_diag():
    """Test computing diagonal."""
    norb = 5
    nelec = (3, 2)
    # TODO remove real=True once complex is supported
    hamiltonian = ffsim.random.random_single_factorized_hamiltonian(
        norb, real=True, seed=RNG
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
@pytest.mark.parametrize("real", [True, False])
def test_fermion_operator(norb: int, nelec: tuple[int, int], real: bool):
    """Test FermionOperator."""
    sf_hamiltonian = ffsim.random.random_single_factorized_hamiltonian(
        norb, real=real, seed=RNG
    )
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    op = ffsim.fermion_operator(sf_hamiltonian)
    linop = ffsim.linear_operator(op, norb, nelec)
    expected_linop = ffsim.linear_operator(sf_hamiltonian, norb, nelec)

    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)


def test_approx_eq():
    """Test approximate equality."""
    norb = 4
    hamiltonian = ffsim.random.random_single_factorized_hamiltonian(norb, seed=RNG)
    perturbed = dataclasses.replace(
        hamiltonian, one_body_tensor=hamiltonian.one_body_tensor + 1e-12
    )
    different_constant = dataclasses.replace(
        hamiltonian, constant=hamiltonian.constant + 1
    )
    fewer_terms = dataclasses.replace(
        hamiltonian, one_body_squares=hamiltonian.one_body_squares[:-1]
    )
    assert ffsim.approx_eq(hamiltonian, hamiltonian)
    assert ffsim.approx_eq(hamiltonian, perturbed)
    assert not ffsim.approx_eq(hamiltonian, different_constant)
    assert not ffsim.approx_eq(hamiltonian, fewer_terms)
    assert not ffsim.approx_eq(hamiltonian, hamiltonian.to_molecular_hamiltonian())
