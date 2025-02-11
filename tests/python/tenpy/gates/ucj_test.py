# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the TeNPy unitary cluster Jastrow gate."""

import numpy as np
import pytest
from tenpy.algorithms.tebd import TEBDEngine

import ffsim
from ffsim.tenpy.gates.ucj import apply_ucj_op_spin_balanced
from ffsim.tenpy.hamiltonians.molecular_hamiltonian import MolecularHamiltonianMPOModel
from ffsim.tenpy.util import statevector_to_mps
from ffsim.variational.util import interaction_pairs_spin_balanced


@pytest.mark.parametrize(
    "norb, nelec, n_reps, connectivity",
    [
        (4, (2, 2), 1, "square"),
        (4, (1, 2), 1, "square"),
        (4, (0, 2), 1, "square"),
        (4, (0, 0), 1, "square"),
        (4, (2, 2), 1, "hex"),
        (4, (1, 2), 1, "hex"),
        (4, (0, 2), 1, "hex"),
        (4, (0, 0), 1, "hex"),
        (4, (2, 2), 1, "heavy-hex"),
        (4, (1, 2), 1, "heavy-hex"),
        (4, (0, 2), 1, "heavy-hex"),
        (4, (0, 0), 1, "heavy-hex"),
        (4, (2, 2), 2, "square"),
        (4, (1, 2), 2, "square"),
        (4, (0, 2), 2, "square"),
        (4, (0, 0), 2, "square"),
        (4, (2, 2), 2, "hex"),
        (4, (1, 2), 2, "hex"),
        (4, (0, 2), 2, "hex"),
        (4, (0, 0), 2, "hex"),
        (4, (2, 2), 2, "heavy-hex"),
        (4, (1, 2), 2, "heavy-hex"),
        (4, (0, 2), 2, "heavy-hex"),
        (4, (0, 0), 2, "heavy-hex"),
    ],
)
def test_apply_ucj_op_spin_balanced(
    norb: int, nelec: tuple[int, int], n_reps: int, connectivity: str
):
    """Test applying a spin-balanced unitary cluster Jastrow gate to an MPS."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    mol_hamiltonian_mpo_model = MolecularHamiltonianMPOModel.from_molecular_hamiltonian(
        mol_hamiltonian
    )
    mol_hamiltonian_mpo = mol_hamiltonian_mpo_model.H_MPO

    # generate a random LUCJ ansatz
    lucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs_spin_balanced(
            connectivity=connectivity, norb=norb
        ),
        with_final_orbital_rotation=True,
        seed=rng,
    )

    # generate the corresponding LUCJ circuit statevector
    hf_state = ffsim.hartree_fock_state(norb, nelec)
    lucj_state = ffsim.apply_unitary(hf_state, lucj_op, norb, nelec)

    # generate the corresponding LUCJ circuit MPS
    dim = ffsim.dim(norb, nelec)
    wavefunction_mps = statevector_to_mps(np.array([1] + [0] * (dim - 1)), norb, nelec)
    options = {"trunc_params": {"chi_max": 16, "svd_min": 1e-6}}
    eng = TEBDEngine(wavefunction_mps, None, options)
    apply_ucj_op_spin_balanced(eng, lucj_op)

    # test expectation is preserved
    original_expectation = np.vdot(lucj_state, hamiltonian @ lucj_state).real
    mpo_expectation = mol_hamiltonian_mpo.expectation_value_finite(wavefunction_mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
