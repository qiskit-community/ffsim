# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the TeNPy diagonal Coulomb evolution gate."""

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
def test_apply_diag_coulomb_evolution(norb: int, nelec: tuple[int, int]):
    """Test applying a diagonal Coulomb evolution gate to an MPS."""
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

    # generate random diagonal Coulomb evolution parameters
    mat_aa = np.diag(rng.standard_normal(norb - 1), k=-1)
    mat_aa += mat_aa.T
    mat_ab = np.diag(rng.standard_normal(norb))
    diag_coulomb_mats = np.array([mat_aa, mat_ab, mat_aa])
    time = rng.random()

    # apply random diagonal Coulomb evolution to state vector
    vec = ffsim.apply_diag_coulomb_evolution(
        original_vec, diag_coulomb_mats, time, norb, nelec
    )

    # apply random diagonal Coulomb evolution to MPS
    eng = TEBDEngine(mps, None, {})
    ffsim.tenpy.apply_diag_coulomb_evolution(eng, diag_coulomb_mats[:2], time)

    # test expectation is preserved
    original_expectation = np.vdot(original_vec, hamiltonian @ vec)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = original_mps.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
