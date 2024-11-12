# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for molecular Hamiltonian TeNPy methods."""

import numpy as np
import pytest

import ffsim
from ffsim.tenpy.hamiltonians.molecular_hamiltonian import MolecularHamiltonianMPOModel
from ffsim.tenpy.util import product_state_as_mps


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
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

    # generate a random product state
    dim = ffsim.dim(norb, nelec)
    idx = rng.integers(0, high=dim)
    product_state = np.zeros(dim)
    product_state[idx] = 1

    # convert product state to MPS
    strings_a, strings_b = ffsim.addresses_to_strings(
        range(dim),
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.STRING,
        concatenate=False,
    )
    product_state_mps = product_state_as_mps((strings_a[idx], strings_b[idx]))

    # test expectation is preserved
    original_expectation = np.vdot(product_state, hamiltonian @ product_state)
    mpo_expectation = mol_hamiltonian_mpo.expectation_value_finite(product_state_mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
