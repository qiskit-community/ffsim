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
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Decompose

import ffsim


# TODO handle norb = 0
@pytest.mark.parametrize("norb", range(1, 5))
def test_yields_equivalent_circuit(norb: int):
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
    pass_manager = PassManager([ffsim.qiskit.MergeOrbitalRotations()])
    transpiled = pass_manager.run(circuit)
    assert circuit.count_ops()["orb_rot_jw"] == 7
    assert transpiled.count_ops()["orb_rot_jw"] == 1
    np.testing.assert_allclose(
        np.array(Operator(circuit)), np.array(Operator(transpiled))
    )


# TODO handle norb = 0
@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_merge_slater(norb: int, nelec: tuple[int, int]):
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
    pass_manager = PassManager([ffsim.qiskit.MergeOrbitalRotations()])
    transpiled = pass_manager.run(circuit)
    assert circuit.count_ops() == {"slater_jw": 1, "orb_rot_jw": 7}
    assert transpiled.count_ops() == {"slater_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


# TODO handle norb = 0
@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_merge_hartree_fock(norb: int, nelec: tuple[int, int]):
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
    pass_manager = PassManager(
        [Decompose(["hartree_fock_jw", "ucj_jw"]), ffsim.qiskit.MergeOrbitalRotations()]
    )
    transpiled = pass_manager.run(circuit)
    assert circuit.count_ops() == {"hartree_fock_jw": 1, "orb_rot_jw": 7}
    assert transpiled.count_ops() == {"slater_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


# TODO handle norb = 0
@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_merge_ucj(norb: int, nelec: tuple[int, int]):
    """Test merging orbital rotations in UCJ operator."""
    rng = np.random.default_rng()

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),
        qubits,
    )

    n_reps = 3
    ucj_op = ffsim.UCJOperator(
        diag_coulomb_mats_alpha_alpha=np.stack(
            [
                ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
                for _ in range(n_reps)
            ]
        ),
        diag_coulomb_mats_alpha_beta=np.stack(
            [
                ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
                for _ in range(n_reps)
            ]
        ),
        orbital_rotations=np.stack(
            [ffsim.random.random_unitary(norb, seed=rng) for _ in range(n_reps)]
        ),
        final_orbital_rotation=ffsim.random.random_unitary(norb, seed=rng),
    )
    circuit.append(ffsim.qiskit.UCJOperatorJW(ucj_op), qubits)
    pass_manager = PassManager(
        [Decompose(["hartree_fock_jw", "ucj_jw"]), ffsim.qiskit.MergeOrbitalRotations()]
    )
    transpiled = pass_manager.run(circuit)
    assert circuit.count_ops() == {"hartree_fock_jw": 1, "ucj_jw": 1}
    assert transpiled.count_ops()["slater_jw"] == 1

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )
