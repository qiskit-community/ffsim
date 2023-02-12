# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for double factorization utils."""


import numpy as np

from fqcsim.double_factorized import double_factorized_decomposition
from fqcsim.fci import (
    contract_core_tensor,
    contract_num_op_sum,
    get_dimension,
    get_hamiltonian_linop,
)

from fqcsim.gates import (
    apply_orbital_rotation,
)
from fqcsim.random_utils import (
    random_hermitian,
    random_statevector,
    random_two_body_tensor_real,
)


def test_double_factorized_decomposition():
    # set parameters
    n_orbitals = 4
    n_electrons = (2, 2)

    # generate random Hamiltonian
    dim = get_dimension(n_orbitals, n_electrons)
    # TODO test with complex one-body tensor
    one_body_tensor = np.real(np.array(random_hermitian(n_orbitals, seed=2474)))
    two_body_tensor = random_two_body_tensor_real(n_orbitals, seed=7054)
    hamiltonian = get_hamiltonian_linop(one_body_tensor, two_body_tensor, n_electrons)

    # perform double factorization
    df_hamiltonian = double_factorized_decomposition(one_body_tensor, two_body_tensor)

    # generate random state
    dim = get_dimension(n_orbitals, n_electrons)
    state = np.array(random_statevector(dim, seed=1360))

    # apply Hamiltonian terms
    result = np.zeros_like(state)

    eigs, vecs = np.linalg.eigh(df_hamiltonian.one_body_tensor)
    tmp = apply_orbital_rotation(
        vecs.T.conj(), state, n_orbitals=n_orbitals, n_electrons=n_electrons
    )
    tmp = contract_num_op_sum(eigs, tmp, n_electrons)
    tmp = apply_orbital_rotation(
        vecs, tmp, n_orbitals=n_orbitals, n_electrons=n_electrons
    )
    result += tmp

    for core_tensor, leaf_tensor in zip(
        df_hamiltonian.core_tensors, df_hamiltonian.leaf_tensors
    ):
        tmp = apply_orbital_rotation(
            leaf_tensor.T.conj(), state, n_orbitals=n_orbitals, n_electrons=n_electrons
        )
        tmp = contract_core_tensor(core_tensor, tmp, n_electrons)
        tmp = apply_orbital_rotation(
            leaf_tensor, tmp, n_orbitals=n_orbitals, n_electrons=n_electrons
        )
        result += tmp

    # apply Hamiltonian directly
    expected = hamiltonian @ state

    # check agreement
    np.testing.assert_allclose(result, expected, atol=1e-8)
