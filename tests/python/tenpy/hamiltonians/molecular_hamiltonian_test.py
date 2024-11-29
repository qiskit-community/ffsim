# (C) Copyright IBM 2024.
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
from ffsim.tenpy.util import bitstring_to_mps


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (2, (2, 2)),
        (2, (1, 2)),
        (2, (0, 2)),
        (2, (0, 0)),
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
    for idx1, idx2 in itertools.product(range(dim), repeat=2):
        # generate product states
        product_state_1 = ffsim.linalg.one_hot(dim, idx1)
        product_state_2 = ffsim.linalg.one_hot(dim, idx2)

        # convert product states to MPS
        strings_a_1, strings_b_1 = ffsim.addresses_to_strings(
            [idx1],
            norb=norb,
            nelec=nelec,
            bitstring_type=ffsim.BitstringType.STRING,
            concatenate=False,
        )
        product_state_mps_1 = bitstring_to_mps(
            (int(strings_a_1[0], 2), int(strings_b_1[0], 2)), norb
        )
        strings_a_2, strings_b_2 = ffsim.addresses_to_strings(
            [idx2],
            norb=norb,
            nelec=nelec,
            bitstring_type=ffsim.BitstringType.STRING,
            concatenate=False,
        )
        product_state_mps_2 = bitstring_to_mps(
            (int(strings_a_2[0], 2), int(strings_b_2[0], 2)), norb
        )

        # test expectation is preserved
        original_expectation = np.vdot(product_state_1, hamiltonian @ product_state_2)
        mol_hamiltonian_mpo.apply_naively(product_state_mps_2)
        mpo_expectation = product_state_mps_1.overlap(product_state_mps_2)
        np.testing.assert_allclose(
            abs(original_expectation.real),
            abs(mpo_expectation.real),
            rtol=1e-05,
            atol=1e-08,
        )
        np.testing.assert_allclose(
            abs(original_expectation.imag),
            abs(mpo_expectation.imag),
            rtol=1e-05,
            atol=1e-08,
        )
