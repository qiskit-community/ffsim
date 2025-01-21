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
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.networks.site import FermionSite, SpinHalfFermionSite

import ffsim
from ffsim.tenpy.hamiltonians.molecular_hamiltonian import MolecularHamiltonianMPOModel
from ffsim.tenpy.random.random import random_mps
from ffsim.tenpy.util import bitstring_to_mps, mps_to_statevector, statevector_to_mps


@pytest.mark.parametrize(
    "bitstring, norb, product_state",
    [
        ((0, 0), 2, [0, 0]),
        ((2, 0), 2, [0, 1]),
        ((0, 2), 2, [0, 2]),
        ((2, 2), 2, [0, 3]),
        ((1, 0), 2, [1, 0]),
        ((3, 0), 2, [1, 1]),
        ((1, 2), 2, [1, 2]),
        ((3, 2), 2, [1, 3]),
        ((0, 1), 2, [2, 0]),
        ((2, 1), 2, [2, 1]),
        ((0, 3), 2, [2, 2]),
        ((2, 3), 2, [2, 3]),
        ((1, 1), 2, [3, 0]),
        ((3, 1), 2, [3, 1]),
        ((1, 3), 2, [3, 2]),
        ((3, 3), 2, [3, 3]),
        ((5, 6), 3, [1, 2, 3]),
    ],
)
def test_bitstring_to_mps(bitstring: tuple[int, int], norb: int, product_state: list):
    """Test converting a bitstring to an MPS."""

    # convert bitstring to MPS
    mps = bitstring_to_mps(bitstring, norb)

    # construct expected MPS
    shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    expected_mps = MPS.from_product_state([shfs] * norb, product_state)

    # map from TeNPy to ffsim ordering
    fs = FermionSite(conserve="N")
    alpha_sector = mps.expectation_value("Nu")
    beta_sector = mps.expectation_value("Nd")
    product_state_fs_tenpy = [
        int(val) for pair in zip(alpha_sector, beta_sector) for val in pair
    ]
    mps_fs = MPS.from_product_state([fs] * 2 * norb, product_state_fs_tenpy)

    tenpy_ordering = list(range(2 * norb))
    midpoint = len(tenpy_ordering) // 2
    mask1 = tenpy_ordering[:midpoint][::-1]
    mask2 = tenpy_ordering[midpoint:][::-1]
    ffsim_ordering = [int(val) for pair in zip(mask1, mask2) for val in pair]

    mps_ref = deepcopy(mps_fs)
    mps_ref.permute_sites(ffsim_ordering, swap_op=None)
    mps_fs.permute_sites(ffsim_ordering, swap_op="auto")
    swap_factor = mps_fs.overlap(mps_ref)

    if swap_factor == -1:
        minus_identity_npc = npc.Array.from_ndarray(
            -shfs.get_op("Id").to_ndarray(),
            [shfs.leg, shfs.leg.conj()],
            labels=["p", "p*"],
        )
        expected_mps.apply_local_op(0, minus_identity_npc)

    # test overlap is one
    overlap = mps.overlap(expected_mps)
    np.testing.assert_equal(overlap, 1)


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
    """Test converting an MPS to a statevector."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    mol_hamiltonian_mpo_model = MolecularHamiltonianMPOModel.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    mol_hamiltonian_mpo = mol_hamiltonian_mpo_model.H_MPO

    # generate a random MPS
    mps = random_mps(norb, nelec)

    # convert MPS to statevector
    statevector = mps_to_statevector(mps, mol_hamiltonian_mpo_model)

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
    """Test converting a statevector to an MPS."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    mol_hamiltonian_mpo_model = MolecularHamiltonianMPOModel.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    mol_hamiltonian_mpo = mol_hamiltonian_mpo_model.H_MPO

    # generate a random statevector
    dim = ffsim.dim(norb, nelec)
    statevector = ffsim.random.random_state_vector(dim, seed=rng)

    # convert statevector to MPS
    mps = statevector_to_mps(statevector, mol_hamiltonian_mpo_model, norb, nelec)

    # test expectation is preserved
    original_expectation = np.vdot(statevector, hamiltonian @ statevector)
    mps_original = deepcopy(mps)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = mps_original.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
