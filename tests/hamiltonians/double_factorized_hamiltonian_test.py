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

import numpy as np
import pytest

import ffsim


@pytest.mark.parametrize("z_representation", [False, True])
def test_linear_operator(z_representation: bool):
    """Test linear operator."""
    norb = 4
    nelec = (2, 2)
    rng = np.random.default_rng(2474)

    dim = ffsim.dim(norb, nelec)
    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    constant = rng.standard_normal()
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant
    )

    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian,
        z_representation=z_representation,
    )

    actual_linop = ffsim.linear_operator(df_hamiltonian, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    dim = ffsim.dim(norb, nelec)
    state = ffsim.random.random_statevector(dim, seed=rng)

    actual = actual_linop @ state
    expected = expected_linop @ state

    np.testing.assert_allclose(actual, expected)


def test_z_representation_round_trip():
    norb = 4

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=2474)
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=7054, dtype=float)

    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        ffsim.MolecularHamiltonian(one_body_tensor, two_body_tensor)
    )
    df_hamiltonian_num = df_hamiltonian.to_z_representation().to_number_representation()

    np.testing.assert_allclose(
        df_hamiltonian.one_body_tensor, df_hamiltonian_num.one_body_tensor
    )
    np.testing.assert_allclose(
        df_hamiltonian.diag_coulomb_mats, df_hamiltonian_num.diag_coulomb_mats
    )
    np.testing.assert_allclose(
        df_hamiltonian.orbital_rotations, df_hamiltonian_num.orbital_rotations
    )
    np.testing.assert_allclose(df_hamiltonian.constant, df_hamiltonian_num.constant)
    assert df_hamiltonian.z_representation == df_hamiltonian_num.z_representation
