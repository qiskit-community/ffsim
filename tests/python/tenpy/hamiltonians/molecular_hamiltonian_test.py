# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the TeNPy molecular Hamiltonian."""

import itertools

import numpy as np
import pytest

import ffsim
from ffsim.tenpy.hamiltonians.molecular_hamiltonian import MolecularHamiltonianMPOModel
from ffsim.tenpy.util import statevector_to_mps


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (2, 2)),
        (2, (2, 1)),
        (2, (1, 2)),
        (2, (1, 1)),
        (2, (0, 2)),
        (2, (0, 0)),
        (3, (2, 2)),
    ],
)
def test_from_molecular_hamiltonian(norb: int, nelec: tuple[int, int]):
    """Test conversion from MolecularHamiltonian to MolecularHamiltonianMPOModel."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    mol_hamiltonian_mpo_model = MolecularHamiltonianMPOModel.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    mol_hamiltonian_mpo = mol_hamiltonian_mpo_model.H_MPO

    dim = ffsim.dim(norb, nelec)
    for i, j in itertools.product(range(dim), repeat=2):
        # generate product states
        product_state_i = ffsim.linalg.one_hot(dim, i)
        product_state_j = ffsim.linalg.one_hot(dim, j)

        # convert product states to MPS
        product_state_mps_i = statevector_to_mps(product_state_i, norb, nelec)
        product_state_mps_j = statevector_to_mps(product_state_j, norb, nelec)

        # test expectation is preserved
        original_expectation = np.vdot(product_state_i, hamiltonian @ product_state_j)
        mol_hamiltonian_mpo.apply_naively(product_state_mps_j)
        mpo_expectation = product_state_mps_i.overlap(product_state_mps_j)
        np.testing.assert_allclose(original_expectation, mpo_expectation)
