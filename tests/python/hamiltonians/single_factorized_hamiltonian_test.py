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

import numpy as np
import pytest

import ffsim


@pytest.mark.parametrize(
    "norb, nelec, cholesky",
    [
        (4, (2, 2), False),
        (4, (2, 2), True),
        (4, (2, 1), False),
        (4, (2, 1), True),
        (4, (2, 0), False),
        (4, (2, 0), True),
    ],
)
def test_linear_operator(norb: int, nelec: tuple[int, int], cholesky: bool):
    """Test linear operator."""
    rng = np.random.default_rng(2474)

    dim = ffsim.dim(norb, nelec)
    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    constant = rng.standard_normal()
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant
    )

    sf_hamiltonian = ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian, cholesky=cholesky
    )

    actual_linop = ffsim.linear_operator(sf_hamiltonian, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    dim = ffsim.dim(norb, nelec)
    state = ffsim.random.random_statevector(dim, seed=rng)

    actual = actual_linop @ state
    expected = expected_linop @ state

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (2, 2)),
        (4, (2, 1)),
        (4, (2, 1)),
        (4, (2, 0)),
        (4, (2, 0)),
    ],
)
def test_reduced_matrix_product_states(norb: int, nelec: tuple[int, int]):
    """Test computing reduced matrix on product states."""
    rng = np.random.default_rng(7869)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    constant = rng.standard_normal()
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant
    )
    sf_hamiltonian = ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    linop = ffsim.linear_operator(sf_hamiltonian, norb, nelec)

    n_vecs = 3
    occupations = [
        ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
        for _ in range(n_vecs)
    ]
    orbital_rotations = [
        ffsim.random.random_unitary(norb, seed=rng) for _ in range(n_vecs)
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
    [
        (4, (2, 2)),
        (4, (2, 2)),
        (4, (2, 1)),
        (4, (2, 1)),
        (4, (2, 0)),
        (4, (2, 0)),
    ],
)
def test_expectation_product_state_slater_determinant(
    norb: int, nelec: tuple[int, int]
):
    """Test computing expectation value on Slater determinant product state."""
    rng = np.random.default_rng(3400)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    constant = rng.standard_normal()
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant
    )
    sf_hamiltonian = ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    linop = ffsim.linear_operator(sf_hamiltonian, norb, nelec)

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    for _ in range(5):
        (occ_a, occ_b) = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
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
    [
        (4, (2, 2)),
        (4, (2, 2)),
        (4, (2, 1)),
        (4, (2, 1)),
        (4, (2, 0)),
        (4, (2, 0)),
    ],
)
def test_expectation_product_state(norb: int, nelec: tuple[int, int]):
    """Test computing expectation value on product state."""
    rng = np.random.default_rng(3400)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    constant = rng.standard_normal()
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant
    )
    sf_hamiltonian = ffsim.SingleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    linop = ffsim.linear_operator(sf_hamiltonian, norb, nelec)

    dim_a, dim_b = ffsim.dims(norb, nelec)
    vec_a = ffsim.random.random_statevector(dim_a, seed=rng)
    vec_b = ffsim.random.random_statevector(dim_b, seed=rng)
    vec = np.kron(vec_a, vec_b)

    actual = sf_hamiltonian.expectation_product_state((vec_a, vec_b), norb, nelec)
    expected = np.vdot(vec, linop @ vec)

    np.testing.assert_allclose(actual, expected)
