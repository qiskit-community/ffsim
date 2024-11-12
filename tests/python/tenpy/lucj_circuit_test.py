# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for LUCJ circuit TeNPy methods."""

from copy import deepcopy

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit, QuantumRegister
from tenpy.algorithms.tebd import TEBDEngine

import ffsim
from ffsim.tenpy.circuits.lucj_circuit import lucj_circuit_as_mps
from ffsim.tenpy.hamiltonians.molecular_hamiltonian import MolecularHamiltonianMPOModel
from ffsim.tenpy.util import product_state_as_mps


def _interaction_pairs_spin_balanced_(
    connectivity: str, norb: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Returns alpha-alpha and alpha-beta diagonal Coulomb interaction pairs."""
    if connectivity == "square":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb)]
    elif connectivity == "hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 2 == 0]
    elif connectivity == "heavy-hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 4 == 0]
    else:
        raise ValueError(f"Invalid connectivity: {connectivity}")
    return pairs_aa, pairs_ab


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
def test_lucj_circuit_as_mps(
    norb: int, nelec: tuple[int, int], n_reps: int, connectivity: str
):
    """Test LUCJ circuit MPS construction."""
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
    n_params = ffsim.UCJOpSpinBalanced.n_params(
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=_interaction_pairs_spin_balanced_(
            connectivity=connectivity, norb=norb
        ),
        with_final_orbital_rotation=True,
    )
    params = rng.uniform(-10, 10, size=n_params)
    lucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
        params,
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=_interaction_pairs_spin_balanced_(
            connectivity=connectivity, norb=norb
        ),
        with_final_orbital_rotation=True,
    )

    # generate the corresponding LUCJ circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(lucj_op), qubits)
    lucj_state = ffsim.qiskit.final_state_vector(circuit).vec

    # convert LUCJ ansatz to MPS
    options = {"trunc_params": {"chi_max": 16, "svd_min": 1e-6}}
    wavefunction_mps, _ = lucj_circuit_as_mps(norb, nelec, lucj_op, options)

    # test expectation is preserved
    original_expectation = np.vdot(lucj_state, hamiltonian @ lucj_state).real
    mpo_expectation = mol_hamiltonian_mpo.expectation_value_finite(wavefunction_mps)
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
def test_apply_orbital_rotation(
    norb: int,
    nelec: tuple[int, int],
):
    """Test applying orbital rotation to MPS."""
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
    mps = product_state_as_mps(norb, nelec, idx)
    original_mps = deepcopy(mps)

    # generate a random orbital rotation
    mat = ffsim.random.random_unitary(norb, seed=rng)

    # apply random orbital rotation to state vector
    vec = ffsim.apply_orbital_rotation(original_vec, mat, norb, nelec)

    # apply random orbital rotation to MPS
    chi_list: list[int] = []
    options = {"trunc_params": {"chi_max": 16, "svd_min": 1e-6}}
    eng = TEBDEngine(mps, None, options)
    ffsim.tenpy.apply_orbital_rotation(mps, mat, eng=eng, chi_list=chi_list)

    # test matrix element is preserved
    original_matrix_element = np.vdot(original_vec, hamiltonian @ vec)
    mol_hamiltonian_mpo.apply_naively(mps)
    mpo_matrix_element = mps.overlap(original_mps)
    np.testing.assert_allclose(original_matrix_element, mpo_matrix_element)
