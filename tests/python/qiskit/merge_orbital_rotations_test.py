# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager

import ffsim


def test_yields_equivalent_circuit():
    """Test merging orbital rotations results in an equivalent circuit."""
    rng = np.random.default_rng()
    norb = 3
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb, ffsim.random.random_unitary(3, seed=rng)
            ),
            qubits,
        )
    for _ in range(4):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb,
                (
                    ffsim.random.random_unitary(3, seed=rng),
                    ffsim.random.random_unitary(3, seed=rng),
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
