# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for FfsimSampler."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler

import ffsim
import ffsim.random.random


def _fidelity(probs1: dict, probs2: dict) -> float:
    result = 0.0
    for bitstring in probs1.keys() | probs2.keys():
        prob1 = probs1.get(bitstring, 0)
        prob2 = probs2.get(bitstring, 0)
        result += math.sqrt(prob1 * prob2)
    return result**2


def _brickwork(norb: int, n_layers: int):
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)


# TODO remove after removing UCJOperatorJW
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_random_gates_spinful(norb: int, nelec: tuple[int, int]):
    """Test sampler with random gates."""
    rng = np.random.default_rng(12285)

    qubits = QuantumRegister(2 * norb)

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    ucj_op = ffsim.random.random_ucj_operator(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=rng
    )
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
    givens_ansatz_op = ffsim.GivensAnsatzOperator(norb, interaction_pairs, thetas)

    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation), qubits)
    circuit.append(
        ffsim.qiskit.DiagCoulombEvolutionJW(norb, diag_coulomb_mat, time=1.0), qubits
    )
    circuit.append(ffsim.qiskit.GivensAnsatzOperatorJW(givens_ansatz_op), qubits)
    circuit.append(ffsim.qiskit.UCJOperatorJW(ucj_op), qubits)
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
    circuit.measure_all()

    shots = 5000

    sampler = StatevectorSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    samples = pub_result.data.meas.get_counts()

    vec = ffsim.qiskit.final_state_vector(
        circuit.remove_final_measurements(inplace=False)
    )
    exact_probs = np.abs(vec) ** 2
    strings, counts = zip(*samples.items())
    addresses = ffsim.strings_to_addresses(strings, norb, nelec)
    assert sum(counts) == shots
    empirical_probs = np.zeros(ffsim.dim(norb, nelec), dtype=float)
    empirical_probs[addresses] = np.array(counts) / shots
    assert np.sum(np.sqrt(exact_probs * empirical_probs)) > 0.999


# TODO remove after removing UCJOperatorJW
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(1, 5)))
def test_random_gates_spinless(norb: int, nocc: int):
    """Test sampler with random spinless gates."""
    rng = np.random.default_rng(52622)

    qubits = QuantumRegister(norb)

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    interaction_pairs = list(_brickwork(norb, norb))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    givens_ansatz_op = ffsim.GivensAnsatzOperator(norb, interaction_pairs, thetas)
    ucj_op = ffsim.random.random_ucj_op_spinless(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=rng
    )

    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockSpinlessJW(norb, nocc), qubits)
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(norb, orbital_rotation), qubits
    )
    circuit.append(
        ffsim.qiskit.GivensAnsatzOperatorSpinlessJW(givens_ansatz_op), qubits
    )
    circuit.append(ffsim.qiskit.UCJOpSpinlessJW(ucj_op), qubits)
    circuit.measure_all()

    shots = 1000

    sampler = StatevectorSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    samples = pub_result.data.meas.get_counts()

    vec = ffsim.qiskit.final_state_vector(
        circuit.remove_final_measurements(inplace=False)
    )
    exact_probs = np.abs(vec) ** 2
    strings, counts = zip(*samples.items())
    addresses = ffsim.strings_to_addresses(strings, norb, nocc)
    assert sum(counts) == shots
    empirical_probs = np.zeros(ffsim.dim(norb, nocc), dtype=float)
    empirical_probs[addresses] = np.array(counts) / shots
    assert np.sum(np.sqrt(exact_probs * empirical_probs)) > 0.999


# TODO remove after removing UCJOperatorJW
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_measure_subset_spinful(norb: int, nelec: tuple[int, int]):
    """Test measuring a subset of qubits."""
    rng = np.random.default_rng(5332)

    qubits = QuantumRegister(2 * norb, name="q")
    clbits = ClassicalRegister(norb, name="meas")
    measured_qubits = list(rng.choice(qubits, size=len(clbits), replace=False))
    measured_clbits = list(rng.choice(clbits, size=len(clbits), replace=False))

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)

    circuit = QuantumCircuit(qubits, clbits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        ),
        qubits,
    )
    circuit.measure(measured_qubits, measured_clbits)

    shots = 3000

    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    ffsim_probs = {bitstring: count / shots for bitstring, count in counts.items()}
    np.testing.assert_allclose(sum(ffsim_probs.values()), 1)

    sampler = StatevectorSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    qiskit_probs = {bitstring: count / shots for bitstring, count in counts.items()}
    np.testing.assert_allclose(sum(qiskit_probs.values()), 1)

    assert _fidelity(ffsim_probs, qiskit_probs) > 0.99


# TODO remove after removing UCJOperatorJW
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(1, 5)))
def test_measure_subset_spinless(norb: int, nocc: int):
    """Test measuring a subset of qubits, spinless."""
    rng = np.random.default_rng(5332)

    qubits = QuantumRegister(norb, name="q")
    clbits = ClassicalRegister(norb, name="meas")
    measured_qubits = list(rng.choice(qubits, size=len(clbits), replace=False))
    measured_clbits = list(rng.choice(clbits, size=len(clbits), replace=False))

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(
        norb, (nocc, 0), seed=rng
    )[0]

    circuit = QuantumCircuit(qubits, clbits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        ),
        qubits,
    )
    circuit.measure(measured_qubits, measured_clbits)

    shots = 3000

    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    ffsim_probs = {bitstring: count / shots for bitstring, count in counts.items()}
    np.testing.assert_allclose(sum(ffsim_probs.values()), 1)

    sampler = StatevectorSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    qiskit_probs = {bitstring: count / shots for bitstring, count in counts.items()}
    np.testing.assert_allclose(sum(qiskit_probs.values()), 1)

    assert _fidelity(ffsim_probs, qiskit_probs) > 0.99


# TODO remove after removing UCJOperatorJW
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_reproducible_with_seed():
    """Test sampler with random gates."""
    rng = np.random.default_rng(14062)

    norb = 4
    nelec = (2, 2)

    qubits = QuantumRegister(2 * norb, name="q")

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)

    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        ),
        qubits,
    )
    circuit.measure_all()

    shots = 3000

    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=12345)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts_1 = pub_result.data.meas.get_counts()

    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=12345)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts_2 = pub_result.data.meas.get_counts()

    assert counts_1 == counts_2
