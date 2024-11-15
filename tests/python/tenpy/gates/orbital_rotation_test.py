# (C) Copyright IBM 2024.
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
from ffsim.tenpy.util import bitstring_to_mps


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

    # generate a random product state
    dim = ffsim.dim(norb, nelec)
    idx = rng.integers(0, high=dim)
    original_vec = np.zeros(dim, dtype=complex)
    original_vec[idx] = 1

    # convert random product state to MPS
    strings_a, strings_b = ffsim.addresses_to_strings(
        range(dim),
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.STRING,
        concatenate=False,
    )
    mps = bitstring_to_mps((strings_a[idx], strings_b[idx]))
    original_mps = deepcopy(mps)

    # generate a random orbital rotation
    mat = ffsim.random.random_unitary(norb, seed=rng)

    # apply random orbital rotation to state vector
    vec = ffsim.apply_orbital_rotation(original_vec, mat, norb, nelec)

    # apply random orbital rotation to MPS
    options = {"trunc_params": {"chi_max": 16, "svd_min": 1e-6}}
    eng = TEBDEngine(mps, None, options)
    ffsim.tenpy.apply_orbital_rotation(eng, mat)

    # test matrix element is preserved
    original_matrix_element = np.vdot(original_vec, hamiltonian @ vec)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_matrix_element = mps.overlap(original_mps)
    np.testing.assert_allclose(original_matrix_element, mpo_matrix_element)
