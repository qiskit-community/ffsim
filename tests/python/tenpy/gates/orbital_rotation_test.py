# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the TeNPy orbital rotation gate."""

from copy import deepcopy

import numpy as np
import pytest
from tenpy.algorithms.tebd import TEBDEngine

import ffsim
from ffsim.tenpy.hamiltonians.molecular_hamiltonian import MolecularHamiltonianMPOModel
from ffsim.tenpy.util import statevector_to_mps


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
def test_apply_orbital_rotation(
    norb: int,
    nelec: tuple[int, int],
):
    """Test applying an orbital rotation gate to an MPS."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    mol_hamiltonian_mpo_model = MolecularHamiltonianMPOModel.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    mol_hamiltonian_mpo = mol_hamiltonian_mpo_model.H_MPO

    # generate a random state vector
    dim = ffsim.dim(norb, nelec)
    original_vec = ffsim.random.random_state_vector(dim, seed=rng)

    # convert random state vector to MPS
    mps = statevector_to_mps(original_vec, norb, nelec)
    original_mps = deepcopy(mps)

    # generate a random orbital rotation
    mat = ffsim.random.random_unitary(norb, seed=rng)

    # apply random orbital rotation to state vector
    vec = ffsim.apply_orbital_rotation(original_vec, mat, norb, nelec)

    # apply random orbital rotation to MPS
    eng = TEBDEngine(mps, None, {})
    ffsim.tenpy.apply_orbital_rotation(eng, mat)

    # test expectation is preserved
    original_expectation = np.vdot(original_vec, hamiltonian @ vec)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = original_mps.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
