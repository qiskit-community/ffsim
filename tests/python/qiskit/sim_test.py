# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for state vector simulation of Qiskit circuits."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    CPhaseGate,
    CZGate,
    GlobalPhaseGate,
    PhaseGate,
    RZGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    TdgGate,
    TGate,
    XGate,
    XXPlusYYGate,
    ZGate,
    iSwapGate,
)
from qiskit.quantum_info import Statevector

import ffsim


def _brickwork(norb: int, n_layers: int):
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_random_gates_spinful(norb: int, nelec: tuple[int, int]):
    """Test with random gates."""
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
    """Test with random spinless gates."""
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


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (norb, nelec)
        for norb, nelec in ffsim.testing.generate_norb_nelec(range(1, 5))
        if nelec != (0, 0)
    ],
)
def test_qiskit_gates_spinful(norb: int, nelec: tuple[int, int]):
    """Test with Qiskit gates, spinful."""
    rng = np.random.default_rng(12285)
    prng = random.Random(11832)
    pairs = list(itertools.combinations(range(norb), 2))
    prng.shuffle(pairs)
    big_pairs = list(itertools.combinations(range(2 * norb), 2))
    prng.shuffle(big_pairs)

    # Construct circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    n_alpha, n_beta = nelec
    for i in range(n_alpha):
        circuit.append(XGate(), [qubits[i]])
    for i in range(n_beta):
        circuit.append(XGate(), [qubits[norb + i]])
    for i, j in pairs:
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
    for i, j in big_pairs:
        circuit.append(CPhaseGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
        circuit.append(CZGate(), [qubits[i], qubits[j]])
    for i, j in pairs:
        circuit.append(iSwapGate(), [qubits[i], qubits[j]])
        circuit.append(iSwapGate(), [qubits[norb + i], qubits[norb + j]])
    for q in qubits:
        circuit.append(RZGate(rng.uniform(-10, 10)), [q])
    for i, j in big_pairs:
        circuit.append(RZZGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
    for i, j in pairs:
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[norb + i], qubits[norb + j]],
        )
        circuit.append(SwapGate(), [qubits[i], qubits[j]])
        circuit.append(SwapGate(), [qubits[norb + i], qubits[norb + j]])
    circuit.append(GlobalPhaseGate(rng.uniform(-10, 10)))

    # Compute state vector using ffsim
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=norb, nelec=nelec)

    # Compute state vector using Qiskit
    qiskit_vec = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        Statevector(circuit).data, norb=norb, nelec=nelec
    )

    # Check that the state vectors match
    np.testing.assert_allclose(ffsim_vec, qiskit_vec)


@pytest.mark.parametrize(
    "norb, nocc",
    [
        (norb, nocc)
        for norb, nocc in ffsim.testing.generate_norb_nocc(range(1, 5))
        if nocc
    ],
)
def test_qiskit_gates_spinless(norb: int, nocc: int):
    """Test with Qiskit gates, spinless."""
    rng = np.random.default_rng(12285)
    prng = random.Random(11832)
    pairs = list(itertools.combinations(range(norb), 2))
    prng.shuffle(pairs)

    # Construct circuit
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    for i in range(nocc):
        circuit.append(XGate(), [qubits[i]])
    for i, j in pairs:
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
    for q in qubits:
        circuit.append(PhaseGate(rng.uniform(-10, 10)), [q])
    for i, j in pairs:
        circuit.append(CPhaseGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
        circuit.append(CZGate(), [qubits[i], qubits[j]])
        circuit.append(iSwapGate(), [qubits[i], qubits[j]])
    for q in qubits:
        circuit.append(RZGate(rng.uniform(-10, 10)), [q])
    for i, j in pairs:
        circuit.append(RZZGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
    for i, j in pairs:
        circuit.append(SwapGate(), [qubits[i], qubits[j]])
    circuit.append(GlobalPhaseGate(rng.uniform(-10, 10)))

    # Compute state vector using ffsim
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=norb, nelec=nocc)

    # Compute state vector using Qiskit
    qiskit_vec = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        Statevector(circuit).data, norb=norb, nelec=nocc
    )

    # Check that the state vectors match
    np.testing.assert_allclose(ffsim_vec, qiskit_vec)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (norb, nelec)
        for norb, nelec in ffsim.testing.generate_norb_nelec(range(1, 4))
        if nelec != (0, 0)
    ],
)
def test_z_s_t_gates_spinful(norb: int, nelec: tuple[int, int]):
    """Test Z, S, and T gates, spinful."""
    rng = np.random.default_rng(12285)
    qubits = QuantumRegister(2 * norb)
    n_alpha, n_beta = nelec
    for gate in [ZGate, SGate, SdgGate, TGate, TdgGate]:
        circuit = QuantumCircuit(qubits)
        for i in range(n_alpha):
            circuit.append(XGate(), [qubits[i]])
        for i in range(n_beta):
            circuit.append(XGate(), [qubits[norb + i]])
        for i, j in _brickwork(norb, 2):
            circuit.append(
                XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
                [qubits[i], qubits[j]],
            )
            circuit.append(
                XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
                [qubits[norb + i], qubits[norb + j]],
            )
        for q in qubits:
            circuit.append(gate(), [q])
        ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=norb, nelec=nelec)
        qiskit_vec = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            Statevector(circuit).data, norb=norb, nelec=nelec
        )
        np.testing.assert_allclose(ffsim_vec, qiskit_vec)


@pytest.mark.parametrize(
    "norb, nocc",
    [
        (norb, nocc)
        for norb, nocc in ffsim.testing.generate_norb_nocc(range(1, 4))
        if nocc
    ],
)
def test_z_s_t_gates_spinless(norb: int, nocc: int):
    """Test Z, S, and T gates, spinless."""
    rng = np.random.default_rng(12285)
    qubits = QuantumRegister(norb)
    for gate in [ZGate, SGate, SdgGate, TGate, TdgGate]:
        circuit = QuantumCircuit(qubits)
        for i in range(nocc):
            circuit.append(XGate(), [qubits[i]])
        for i, j in _brickwork(norb, 2):
            circuit.append(
                XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
                [qubits[i], qubits[j]],
            )
        for q in qubits:
            circuit.append(gate(), [q])
        ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=norb, nelec=nocc)
        qiskit_vec = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            Statevector(circuit).data, norb=norb, nelec=nocc
        )
        np.testing.assert_allclose(ffsim_vec, qiskit_vec)


def test_qiskit_gates_norb_nelec():
    """Test Qiskit gates passing different values for norb and nelec."""
    norb = 4
    nelec = (2, 2)

    rng = np.random.default_rng(12285)

    # Construct circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    n_alpha, n_beta = nelec
    for i in range(n_alpha):
        circuit.append(XGate(), [qubits[i]])
    for i in range(n_beta):
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

    # Compute state vector using Qiskit
    qiskit_vec_spinful = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        Statevector(circuit).data, norb=norb, nelec=nelec
    )
    qiskit_vec_spinless = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        Statevector(circuit).data, norb=2 * norb, nelec=sum(nelec)
    )

    # Not passing norb and nelec should give spinless result
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit)
    np.testing.assert_allclose(ffsim_vec, qiskit_vec_spinless)

    # Passing norb and nelec should give spinful result
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=norb, nelec=nelec)
    np.testing.assert_allclose(ffsim_vec, qiskit_vec_spinful)

    # Pass only norb
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=2 * norb)
    np.testing.assert_allclose(ffsim_vec, qiskit_vec_spinless)

    # Pass only nelec
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit, nelec=nelec)
    np.testing.assert_allclose(ffsim_vec, qiskit_vec_spinful)
    ffsim_vec = ffsim.qiskit.final_state_vector(circuit, nelec=sum(nelec))
    np.testing.assert_allclose(ffsim_vec, qiskit_vec_spinless)

    # Pass wrong norb
    with pytest.raises(ValueError, match="norb"):
        ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=2 * norb, nelec=nelec)

    # Pass wrong nelec
    with pytest.raises(ValueError, match="nelec"):
        ffsim_vec = ffsim.qiskit.final_state_vector(circuit, norb=norb, nelec=(2, 1))
