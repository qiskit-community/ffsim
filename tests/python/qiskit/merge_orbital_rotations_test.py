# (C) Copyright IBM 2024.
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
from qiskit.quantum_info import Operator, Statevector

import ffsim


@pytest.mark.parametrize("norb", range(1, 5))
def test_yields_equivalent_circuit_spinful(norb: int):
    """Test merging orbital rotations results in an equivalent circuit."""
    rng = np.random.default_rng()
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb, ffsim.random.random_unitary(norb, seed=rng)
            ),
            qubits,
        )
    for _ in range(4):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb,
                (
                    ffsim.random.random_unitary(norb, seed=rng),
                    ffsim.random.random_unitary(norb, seed=rng),
                ),
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert circuit.count_ops()["orb_rot_jw"] == 7
    assert transpiled.count_ops()["orb_rot_jw"] == 1
    np.testing.assert_allclose(
        np.array(Operator(circuit)), np.array(Operator(transpiled))
    )


@pytest.mark.parametrize("norb", range(1, 5))
def test_yields_equivalent_circuit_spinless(norb: int):
    """Test merging orbital rotations results in an equivalent circuit, spinless."""
    rng = np.random.default_rng()
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationSpinlessJW(
                norb, ffsim.random.random_unitary(norb, seed=rng)
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert circuit.count_ops()["orb_rot_spinless_jw"] == 3
    assert transpiled.count_ops()["orb_rot_spinless_jw"] == 1
    np.testing.assert_allclose(
        np.array(Operator(circuit)), np.array(Operator(transpiled))
    )


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_merge_slater_spinful(norb: int, nelec: tuple[int, int]):
    """Test merging orbital rotations into Slater determinant preparation."""
    rng = np.random.default_rng()

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb,
            ffsim.testing.random_occupied_orbitals(norb, nelec),
            ffsim.random.random_unitary(norb, seed=rng),
        ),
        qubits,
    )
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb, ffsim.random.random_unitary(norb, seed=rng)
            ),
            qubits,
        )
    for _ in range(4):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb,
                (
                    ffsim.random.random_unitary(norb, seed=rng),
                    ffsim.random.random_unitary(norb, seed=rng),
                ),
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert circuit.count_ops() == {"slater_jw": 1, "orb_rot_jw": 7}
    assert transpiled.count_ops() == {"slater_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(1, 5)))
def test_merge_slater_spinless(norb: int, nocc: int):
    """Test merging orbital rotations into Slater determinant preparation, spinless."""
    rng = np.random.default_rng()

    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
            norb,
            ffsim.testing.random_occupied_orbitals(norb, (nocc, 0))[0],
            ffsim.random.random_unitary(norb, seed=rng),
        ),
        qubits,
    )
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationSpinlessJW(
                norb, ffsim.random.random_unitary(norb, seed=rng)
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert circuit.count_ops() == {"slater_spinless_jw": 1, "orb_rot_spinless_jw": 3}
    assert transpiled.count_ops() == {"slater_spinless_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_merge_hartree_fock_spinful(norb: int, nelec: tuple[int, int]):
    """Test merging orbital rotations into Hartree-Fock state preparation."""
    rng = np.random.default_rng()

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),
        qubits,
    )
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb, ffsim.random.random_unitary(norb, seed=rng)
            ),
            qubits,
        )
    for _ in range(4):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb,
                (
                    ffsim.random.random_unitary(norb, seed=rng),
                    ffsim.random.random_unitary(norb, seed=rng),
                ),
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert circuit.count_ops() == {"hartree_fock_jw": 1, "orb_rot_jw": 7}
    assert transpiled.count_ops() == {"slater_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(1, 5)))
def test_merge_hartree_fock_spinless(norb: int, nocc: int):
    """Test merging orbital rotations into Hartree-Fock state preparation."""
    rng = np.random.default_rng()

    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareHartreeFockSpinlessJW(norb, nocc),
        qubits,
    )
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationSpinlessJW(
                norb, ffsim.random.random_unitary(norb, seed=rng)
            ),
            qubits,
        )

    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert circuit.count_ops() == {
        "hartree_fock_spinless_jw": 1,
        "orb_rot_spinless_jw": 3,
    }
    assert transpiled.count_ops() == {"slater_spinless_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_merge_ucj(norb: int, nelec: tuple[int, int]):
    """Test merging orbital rotations in UCJ operator."""
    rng = np.random.default_rng()
    qubits = QuantumRegister(2 * norb)
    n_reps = 3

    with pytest.deprecated_call():
        circuit = QuantumCircuit(qubits)
        circuit.append(
            ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),
            qubits,
        )
        ucj_op = ffsim.random.random_ucj_operator(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )
        circuit.append(ffsim.qiskit.UCJOperatorJW(ucj_op), qubits)
        transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
        assert circuit.count_ops() == {"hartree_fock_jw": 1, "ucj_jw": 1}
        assert transpiled.count_ops()["slater_jw"] == 1
        assert transpiled.count_ops()["orb_rot_jw"] == n_reps
        ffsim.testing.assert_allclose_up_to_global_phase(
            np.array(Statevector(circuit)), np.array(Statevector(transpiled))
        )

    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),
        qubits,
    )
    ucj_op_unbalanced = ffsim.random.random_ucj_op_spin_unbalanced(
        norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
    )
    circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op_unbalanced), qubits)
    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert circuit.count_ops() == {"hartree_fock_jw": 1, "ucj_unbalanced_jw": 1}
    assert transpiled.count_ops()["slater_jw"] == 1
    assert transpiled.count_ops()["orb_rot_jw"] == n_reps
    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )

    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),
        qubits,
    )
    ucj_op_balanced = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
    )
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op_balanced), qubits)
    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert circuit.count_ops() == {"hartree_fock_jw": 1, "ucj_balanced_jw": 1}
    assert transpiled.count_ops()["slater_jw"] == 1
    assert transpiled.count_ops()["orb_rot_jw"] == n_reps
    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )
