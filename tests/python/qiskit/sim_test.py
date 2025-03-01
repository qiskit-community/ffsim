# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CPhaseGate, PhaseGate, XGate, XXPlusYYGate
from qiskit.quantum_info import Statevector

import ffsim


def _brickwork(norb: int, n_layers: int):
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_random_gates_spinful(norb: int, nelec: tuple[int, int]):
    """Test sampler with random gates."""
    rng = np.random.default_rng(12285)

    # Initialize test objects
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    ucj_op_balanced = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=rng
    )
    ucj_op_unbalanced = ffsim.random.random_ucj_op_spin_unbalanced(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=rng
    )
    df_hamiltonian_num_rep = ffsim.random.random_double_factorized_hamiltonian(
        norb, rank=3, z_representation=False, seed=rng
    )
    df_hamiltonian_z_rep = ffsim.random.random_double_factorized_hamiltonian(
        norb, rank=3, z_representation=True, seed=rng
    )
    interaction_pairs = list(_brickwork(norb, norb))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phis = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phase_angles = rng.uniform(-np.pi, np.pi, size=norb)
    givens_ansatz_op = ffsim.GivensAnsatzOp(
        norb, interaction_pairs, thetas, phis=phis, phase_angles=phase_angles
    )

    # Construct circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation), qubits)
    circuit.append(
        ffsim.qiskit.DiagCoulombEvolutionJW(norb, diag_coulomb_mat, time=1.0), qubits
    )
    circuit.append(ffsim.qiskit.GivensAnsatzOpJW(givens_ansatz_op), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op_balanced), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op_unbalanced), qubits)
    circuit.append(
        ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(
            df_hamiltonian_num_rep, time=1.0
        ),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(df_hamiltonian_z_rep, time=1.0),
        qubits,
    )

    # Compute state vector using ffsim
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit)

    # Compute state vector using Qiskit
    qiskit_vec = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        Statevector(circuit).data, norb=norb, nelec=nelec
    )

    # Check that the state vectors match
    np.testing.assert_allclose(ffsim_vec, qiskit_vec)


@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(1, 5)))
def test_random_gates_spinless(norb: int, nocc: int):
    """Test sampler with random spinless gates."""
    rng = np.random.default_rng(52622)

    # Initialize test objects
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    interaction_pairs = list(_brickwork(norb, norb))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phis = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phase_angles = rng.uniform(-np.pi, np.pi, size=norb)
    givens_ansatz_op = ffsim.GivensAnsatzOp(
        norb, interaction_pairs, thetas, phis=phis, phase_angles=phase_angles
    )
    ucj_op = ffsim.random.random_ucj_op_spinless(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=rng
    )

    # Construct circuit
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockSpinlessJW(norb, nocc), qubits)
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(norb, orbital_rotation), qubits
    )
    circuit.append(ffsim.qiskit.GivensAnsatzOpSpinlessJW(givens_ansatz_op), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinlessJW(ucj_op), qubits)

    # Compute state vector using ffsim
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit)

    # Compute state vector using Qiskit
    qiskit_vec = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        Statevector(circuit).data, norb=norb, nelec=nocc
    )

    # Check that the state vectors match
    np.testing.assert_allclose(ffsim_vec, qiskit_vec)


# @pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
# def test_qiskit_gates(norb: int, nelec: tuple[int, int]):
# TODO replace with above commented-out lines
def test_qiskit_gates_spinful():
    norb = 4
    nelec = (2, 2)
    """Test sampler with Qiskit gates."""
    rng = np.random.default_rng(12285)

    # Construct circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    for i in range(norb // 2):
        circuit.append(XGate(), [qubits[i]])
        circuit.append(XGate(), [qubits[norb + i]])
    for i, j in _brickwork(norb, norb):
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[norb + j], qubits[norb + i]],
        )
    for i, j in _brickwork(norb, norb):
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[norb + j], qubits[norb + i]],
        )
    for q in qubits:
        circuit.append(PhaseGate(rng.uniform(-10, 10)), [q])
    # for i, j in _brickwork(2 * norb, norb):
    #     circuit.append(CPhaseGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
    # for q in qubits:
    #     circuit.append(RZGate(rng.uniform(-10, 10)), [q])
    # for i, j in _brickwork(2 * norb, norb):
    #     circuit.append(RZZGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
    # for i, j in _brickwork(norb, norb):
    #     circuit.append(
    #         XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
    #         [qubits[i], qubits[j]],
    #     )
    #     circuit.append(
    #         XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
    #         [qubits[norb + i], qubits[norb + j]],
    #     )
    # for i, j in _brickwork(norb, norb):
    #     circuit.append(
    #         XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
    #         [qubits[i], qubits[j]],
    #     )
    #     circuit.append(
    #         XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
    #         [qubits[norb + i], qubits[norb + j]],
    #     )

    # Compute state vector using ffsim
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=norb, nelec=nelec)

    # Compute state vector using Qiskit
    qiskit_vec = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        Statevector(circuit).data, norb=norb, nelec=nelec
    )

    # Check that the state vectors match
    np.testing.assert_allclose(ffsim_vec, qiskit_vec)


# TODO parameterize
def test_qiskit_gates_spinless():
    norb = 4
    nelec = 2
    """Test sampler with Qiskit gates."""
    rng = np.random.default_rng(12285)

    # Construct circuit
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    for i in range(norb // 2):
        circuit.append(XGate(), [qubits[i]])
    for i, j in _brickwork(norb, norb):
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
    for i, j in _brickwork(norb, norb):
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
    for q in qubits:
        circuit.append(PhaseGate(rng.uniform(-10, 10)), [q])
    # for i, j in _brickwork(2 * norb, norb):
    #     circuit.append(CPhaseGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
    # for q in qubits:
    #     circuit.append(RZGate(rng.uniform(-10, 10)), [q])
    # for i, j in _brickwork(2 * norb, norb):
    #     circuit.append(RZZGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
    # for i, j in _brickwork(norb, norb):
    #     circuit.append(
    #         XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
    #         [qubits[i], qubits[j]],
    #     )
    #     circuit.append(
    #         XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
    #         [qubits[norb + i], qubits[norb + j]],
    #     )
    # for i, j in _brickwork(norb, norb):
    #     circuit.append(
    #         XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
    #         [qubits[i], qubits[j]],
    #     )
    #     circuit.append(
    #         XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
    #         [qubits[norb + i], qubits[norb + j]],
    #     )

    # Compute state vector using ffsim
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=norb, nelec=nelec)

    # Compute state vector using Qiskit
    qiskit_vec = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        Statevector(circuit).data, norb=norb, nelec=nelec
    )

    # Check that the state vectors match
    np.testing.assert_allclose(ffsim_vec, qiskit_vec)
