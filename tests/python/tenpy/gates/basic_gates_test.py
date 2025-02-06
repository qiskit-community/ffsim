# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the TeNPy basic gates."""

from copy import deepcopy

import numpy as np
import pytest
from tenpy.algorithms.tebd import TEBDEngine

import ffsim
from ffsim.spin import Spin
from ffsim.tenpy.gates.basic_gates import (
    givens_rotation,
    num_interaction,
    num_num_interaction,
    on_site_interaction,
)
from ffsim.tenpy.hamiltonians.molecular_hamiltonian import MolecularHamiltonianMPOModel
from ffsim.tenpy.util import statevector_to_mps


@pytest.mark.parametrize(
    "norb, nelec, spin",
    [
        (4, (0, 0), Spin.ALPHA),
        (4, (0, 1), Spin.ALPHA),
        (4, (1, 2), Spin.ALPHA),
        (4, (2, 2), Spin.ALPHA),
        (4, (0, 0), Spin.BETA),
        (4, (0, 1), Spin.BETA),
        (4, (1, 2), Spin.BETA),
        (4, (2, 2), Spin.BETA),
        (4, (0, 0), Spin.ALPHA_AND_BETA),
        (4, (0, 1), Spin.ALPHA_AND_BETA),
        (4, (1, 2), Spin.ALPHA_AND_BETA),
        (4, (2, 2), Spin.ALPHA_AND_BETA),
    ],
)
def test_givens_rotation(norb: int, nelec: tuple[int, int], spin: Spin):
    """Test applying a Givens rotation gate to an MPS."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    mpo_model = MolecularHamiltonianMPOModel.from_molecular_hamiltonian(mol_hamiltonian)
    mpo = mpo_model.H_MPO

    # generate a random state vector
    dim = ffsim.dim(norb, nelec)
    original_vec = ffsim.random.random_state_vector(dim, seed=rng)

    # convert random state vector to MPS
    mps = statevector_to_mps(original_vec, mpo_model, norb, nelec)
    original_mps = deepcopy(mps)

    # generate random Givens rotation parameters
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, 2 * np.pi)
    p = rng.integers(norb - 1)

    # apply random Givens rotation to state vector
    vec = ffsim.apply_givens_rotation(
        original_vec, theta, (p, p + 1), norb, nelec, spin, phi=phi
    )

    # apply random orbital rotation to MPS
    eng = TEBDEngine(mps, None, {})
    ffsim.tenpy.apply_two_site(eng, givens_rotation(theta, spin, phi=phi), (p, p + 1))

    # test expectation is preserved
    original_expectation = np.vdot(original_vec, linop @ vec)
    mpo.apply_naively(mps)
    mpo_expectation = original_mps.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)


@pytest.mark.parametrize(
    "norb, nelec, spin",
    [
        (4, (2, 2), Spin.ALPHA),
        (4, (1, 2), Spin.ALPHA),
        (4, (0, 2), Spin.ALPHA),
        (4, (0, 0), Spin.ALPHA),
        (4, (2, 2), Spin.BETA),
        (4, (1, 2), Spin.BETA),
        (4, (0, 2), Spin.BETA),
        (4, (0, 0), Spin.BETA),
        (4, (2, 2), Spin.ALPHA_AND_BETA),
        (4, (1, 2), Spin.ALPHA_AND_BETA),
        (4, (0, 2), Spin.ALPHA_AND_BETA),
        (4, (0, 0), Spin.ALPHA_AND_BETA),
    ],
)
def test_num_interaction(norb: int, nelec: tuple[int, int], spin: Spin):
    """Test applying a number interaction gate to an MPS."""
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
    mps = statevector_to_mps(original_vec, mol_hamiltonian_mpo_model, norb, nelec)
    original_mps = deepcopy(mps)

    # generate random number interaction parameters
    theta = 2 * np.pi * rng.random()
    p = rng.integers(0, norb)

    # apply random number interaction to state vector
    vec = ffsim.apply_num_interaction(original_vec, theta, p, norb, nelec, spin)

    # apply random number interaction to MPS
    eng = TEBDEngine(mps, None, {})
    ffsim.tenpy.apply_single_site(eng, num_interaction(theta, spin), p)

    # test expectation is preserved
    original_expectation = np.vdot(original_vec, hamiltonian @ vec)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = original_mps.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
def test_on_site_interaction(
    norb: int,
    nelec: tuple[int, int],
):
    """Test applying an on-site interaction gate to an MPS."""
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
    mps = statevector_to_mps(original_vec, mol_hamiltonian_mpo_model, norb, nelec)
    original_mps = deepcopy(mps)

    # generate random on-site interaction parameters
    theta = 2 * np.pi * rng.random()
    p = rng.integers(0, norb)

    # apply random on-site interaction to state vector
    vec = ffsim.apply_on_site_interaction(original_vec, theta, p, norb, nelec)

    # apply random on-site interaction to MPS
    eng = TEBDEngine(mps, None, {})
    ffsim.tenpy.apply_single_site(eng, on_site_interaction(theta), p)

    # test expectation is preserved
    original_expectation = np.vdot(original_vec, hamiltonian @ vec)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = original_mps.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)


@pytest.mark.parametrize(
    "norb, nelec, spin",
    [
        (4, (2, 2), Spin.ALPHA),
        (4, (1, 2), Spin.ALPHA),
        (4, (0, 2), Spin.ALPHA),
        (4, (0, 0), Spin.ALPHA),
        (4, (2, 2), Spin.BETA),
        (4, (1, 2), Spin.BETA),
        (4, (0, 2), Spin.BETA),
        (4, (0, 0), Spin.BETA),
        (4, (2, 2), Spin.ALPHA_AND_BETA),
        (4, (1, 2), Spin.ALPHA_AND_BETA),
        (4, (0, 2), Spin.ALPHA_AND_BETA),
        (4, (0, 0), Spin.ALPHA_AND_BETA),
    ],
)
def test_num_num_interaction(norb: int, nelec: tuple[int, int], spin: Spin):
    """Test applying a number-number interaction gate to an MPS."""
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
    mps = statevector_to_mps(original_vec, mol_hamiltonian_mpo_model, norb, nelec)
    original_mps = deepcopy(mps)

    # generate random number-number interaction parameters
    theta = 2 * np.pi * rng.random()
    p = rng.integers(0, norb - 1)

    # apply random number-number interaction to state vector
    vec = ffsim.apply_num_num_interaction(
        original_vec, theta, (p, p + 1), norb, nelec, spin
    )

    # apply random number-number interaction to MPS
    eng = TEBDEngine(mps, None, {})
    ffsim.tenpy.apply_two_site(eng, num_num_interaction(theta, spin), (p, p + 1))

    # test expectation is preserved
    original_expectation = np.vdot(original_vec, hamiltonian @ vec)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_expectation = original_mps.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
