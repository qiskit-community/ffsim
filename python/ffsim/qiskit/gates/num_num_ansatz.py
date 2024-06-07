# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Number-number interaction ansatz gate."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import CPhaseGate, PhaseGate

from ffsim.variational.num_num import NumNumAnsatzOpSpinBalanced


class NumNumAnsatzOpSpinBalancedJW(Gate):
    r"""Spin-balanced number-number ansatz under the Jordan-Wigner transformation.

    See :class:`NumNumAnsatzOpSpinBalanced` for a description of this gate's unitary.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.
    """

    def __init__(
        self,
        num_num_ansatz_op: NumNumAnsatzOpSpinBalanced,
        *,
        label: str | None = None,
    ):
        """Create a new number-number ansatz operator gate.

        Args:
            num_num_ansatz_op: The number-number ansatz operator.
            label: The label of the gate.
        """
        self.num_num_ansatz_op = num_num_ansatz_op
        super().__init__(
            "num_num_ansatz_balanced_jw", 2 * num_num_ansatz_op.norb, [], label=label
        )

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        for instruction in _num_num_ansatz_spin_balanced_jw(
            qubits, self.num_num_ansatz_op
        ):
            circuit.append(instruction)
        self.definition = circuit


def _num_num_ansatz_spin_balanced_jw(
    qubits: Sequence[Qubit], num_num_ansatz_operator: NumNumAnsatzOpSpinBalanced
) -> Iterator[CircuitInstruction]:
    norb = num_num_ansatz_operator.norb
    pairs_aa, pairs_ab = num_num_ansatz_operator.interaction_pairs
    thetas_aa, thetas_ab = num_num_ansatz_operator.thetas

    # gates that involve a single spin sector
    for sigma in range(2):
        for (i, j), theta in zip(pairs_aa, thetas_aa):
            if i == j:
                yield CircuitInstruction(
                    PhaseGate(0.5 * theta), (qubits[i + sigma * norb],)
                )
            else:
                yield CircuitInstruction(
                    CPhaseGate(theta),
                    (qubits[i + sigma * norb], qubits[j + sigma * norb]),
                )

    # gates that involve both spin sectors
    for (i, j), theta in zip(pairs_ab, thetas_ab):
        angle = 0.5 * theta if i == j else theta
        yield CircuitInstruction(CPhaseGate(angle), (qubits[i], qubits[j + norb]))
        yield CircuitInstruction(CPhaseGate(angle), (qubits[j], qubits[i + norb]))
