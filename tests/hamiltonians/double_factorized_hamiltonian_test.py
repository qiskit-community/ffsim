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
def test_double_factorized_hamiltonian(z_representation: bool):
    # set parameters
    norb = 4
    nelec = (2, 2)

    # generate random Hamiltonian
    dim = ffsim.dim(norb, nelec)
    # TODO test with complex one-body tensor
    one_body_tensor = np.real(ffsim.random.random_hermitian(norb, seed=2474))
    two_body_tensor = ffsim.random.random_two_body_tensor_real(norb, seed=7054)
    mol_hamiltonian = ffsim.MolecularHamiltonian(one_body_tensor, two_body_tensor)
    hamiltonian = ffsim.linear_operator(
        mol_hamiltonian,
        norb=norb,
        nelec=nelec,
    )

    # perform double factorization
    df_hamiltonian = ffsim.double_factorized_hamiltonian(
        mol_hamiltonian,
        z_representation=z_representation,
    )

    # generate random state
    dim = ffsim.dim(norb, nelec)
    state = ffsim.random.random_statevector(dim, seed=1360)

    # apply Hamiltonian terms
    result = df_hamiltonian.constant * state

    eigs, vecs = np.linalg.eigh(df_hamiltonian.one_body_tensor)
    tmp = ffsim.apply_orbital_rotation(state, vecs.T.conj(), norb=norb, nelec=nelec)
    tmp = ffsim.contract.contract_num_op_sum(tmp, eigs, norb=norb, nelec=nelec)
    tmp = ffsim.apply_orbital_rotation(tmp, vecs, norb=norb, nelec=nelec)
    result += tmp

    for diag_coulomb_mat, orbital_rotation in zip(
        df_hamiltonian.diag_coulomb_mats, df_hamiltonian.orbital_rotations
    ):
        tmp = ffsim.apply_orbital_rotation(
            state, orbital_rotation.T.conj(), norb=norb, nelec=nelec
        )
        tmp = ffsim.contract.contract_diag_coulomb(
            tmp,
            diag_coulomb_mat,
            norb=norb,
            nelec=nelec,
            z_representation=z_representation,
        )
        tmp = ffsim.apply_orbital_rotation(
            tmp, orbital_rotation, norb=norb, nelec=nelec
        )
        result += tmp

    # apply Hamiltonian directly
    expected = hamiltonian @ state

    # check agreement
    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_z_representation_round_trip():
    norb = 4

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=2474)
    two_body_tensor = ffsim.random.random_two_body_tensor_real(norb, seed=7054)

    df_hamiltonian = ffsim.double_factorized_hamiltonian(
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
