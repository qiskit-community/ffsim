# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from copy import deepcopy

import numpy as np
import pytest
from tenpy.models.molecular import MolecularModel

import ffsim
from ffsim.tenpy.random.random import random_mps, random_mps_product_state
from ffsim.tenpy.util import mps_to_statevector, statevector_to_mps


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
def test_mps_to_statevector_product_state(norb: int, nelec: tuple[int, int]):
    """Test converting an MPS to a statevector using a product state."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    model_params = dict(
        one_body_tensor=mol_hamiltonian.one_body_tensor,
        two_body_tensor=mol_hamiltonian.two_body_tensor,
        constant=mol_hamiltonian.constant,
    )
    mol_hamiltonian_mpo_model = MolecularModel(model_params)
    mol_hamiltonian_mpo = mol_hamiltonian_mpo_model.H_MPO

    # generate a random MPS (product state)
    mps = random_mps_product_state(norb, nelec)

    # convert MPS to state vector (product state)
    statevector = mps_to_statevector(mps)

    # test expectation is preserved
    original_expectation = np.vdot(statevector, hamiltonian @ statevector)
    mps_original = deepcopy(mps)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = mps_original.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)


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
def test_mps_to_statevector(norb: int, nelec: tuple[int, int]):
    """Test converting an MPS to a state vector."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    model_params = dict(
        one_body_tensor=mol_hamiltonian.one_body_tensor,
        two_body_tensor=mol_hamiltonian.two_body_tensor,
        constant=mol_hamiltonian.constant,
    )
    mol_hamiltonian_mpo_model = MolecularModel(model_params)
    mol_hamiltonian_mpo = mol_hamiltonian_mpo_model.H_MPO

    # generate a random MPS
    mps = random_mps(norb, nelec)

    # convert MPS to state vector
    statevector = mps_to_statevector(mps)

    # test expectation is preserved
    original_expectation = np.vdot(statevector, hamiltonian @ statevector)
    mps_original = deepcopy(mps)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = mps_original.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)


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
def test_statevector_to_mps_product_state(norb: int, nelec: tuple[int, int]):
    """Test converting a state vector to an MPS using a product state."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    model_params = dict(
        one_body_tensor=mol_hamiltonian.one_body_tensor,
        two_body_tensor=mol_hamiltonian.two_body_tensor,
        constant=mol_hamiltonian.constant,
    )
    mol_hamiltonian_mpo_model = MolecularModel(model_params)
    mol_hamiltonian_mpo = mol_hamiltonian_mpo_model.H_MPO

    # generate a random state vector (product state)
    dim = ffsim.dim(norb, nelec)
    idx = rng.integers(0, high=dim)
    statevector = ffsim.linalg.one_hot(dim, idx)

    # convert state vector to MPS (product state)
    mps = statevector_to_mps(statevector, norb, nelec)

    # test expectation is preserved
    original_expectation = np.vdot(statevector, hamiltonian @ statevector)
    mps_original = deepcopy(mps)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = mps_original.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)


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
def test_statevector_to_mps(norb: int, nelec: tuple[int, int]):
    """Test converting a state vector to an MPS."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    model_params = dict(
        one_body_tensor=mol_hamiltonian.one_body_tensor,
        two_body_tensor=mol_hamiltonian.two_body_tensor,
        constant=mol_hamiltonian.constant,
    )
    mol_hamiltonian_mpo_model = MolecularModel(model_params)
    mol_hamiltonian_mpo = mol_hamiltonian_mpo_model.H_MPO

    # generate a random state vector
    dim = ffsim.dim(norb, nelec)
    statevector = ffsim.random.random_state_vector(dim, seed=rng)

    # convert state vector to MPS
    mps = statevector_to_mps(statevector, norb, nelec)

    # test expectation is preserved
    original_expectation = np.vdot(statevector, hamiltonian @ statevector)
    mps_original = deepcopy(mps)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = mps_original.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
