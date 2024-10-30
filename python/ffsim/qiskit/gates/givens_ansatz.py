# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Givens rotation ansatz gate."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import PhaseGate, XXPlusYYGate

from ffsim.variational import GivensAnsatzOp


class GivensAnsatzOpJW(Gate):
    """Givens rotation ansatz operator under the Jordan-Wigner transformation.

    See :class:`ffsim.GivensAnsatzOp` for a description of this gate's unitary.
    """

    def __init__(self, givens_ansatz_op: GivensAnsatzOp, *, label: str | None = None):
        """Create a new Givens ansatz operator gate.

        Args:
            givens_ansatz_op: The Givens rotation ansatz operator.
            label: The label of the gate.
        """
        self.givens_ansatz_op = givens_ansatz_op
        super().__init__("givens_ansatz_jw", 2 * givens_ansatz_op.norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        norb = len(qubits) // 2
        alpha_qubits = qubits[:norb]
        beta_qubits = qubits[norb:]
        for instruction in _givens_ansatz_jw(alpha_qubits, self.givens_ansatz_op):
            circuit.append(instruction)
        for instruction in _givens_ansatz_jw(beta_qubits, self.givens_ansatz_op):
            circuit.append(instruction)
        self.definition = circuit


class GivensAnsatzOpSpinlessJW(Gate):
    """Spinless Givens rotation ansatz operator under the Jordan-Wigner transformation.

    Like :class:`GivensAnsatzOpJW` but only acts on a single spin species.
    """

    def __init__(self, givens_ansatz_op: GivensAnsatzOp, *, label: str | None = None):
        """Create a new Givens ansatz operator gate.

        Args:
            givens_ansatz_op: The Givens rotation ansatz operator.
            label: The label of the gate.
        """
        self.givens_ansatz_op = givens_ansatz_op
        super().__init__(
            "givens_ansatz_spinless_jw", givens_ansatz_op.norb, [], label=label
        )

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        for instruction in _givens_ansatz_jw(qubits, self.givens_ansatz_op):
            circuit.append(instruction)
        self.definition = circuit


def _givens_ansatz_jw(
    qubits: Sequence[Qubit], givens_ansatz_op: GivensAnsatzOp
) -> Iterator[CircuitInstruction]:
    phis = givens_ansatz_op.phis
    if phis is None:
        phis = np.zeros(len(givens_ansatz_op.interaction_pairs))
    for (i, j), theta, phi in zip(
        givens_ansatz_op.interaction_pairs, givens_ansatz_op.thetas, phis
    ):
        yield CircuitInstruction(
            XXPlusYYGate(2 * theta, phi - 0.5 * math.pi), (qubits[i], qubits[j])
        )
    if givens_ansatz_op.phase_angles is not None:
        for i, phase_angle in enumerate(givens_ansatz_op.phase_angles):
            yield CircuitInstruction(PhaseGate(phase_angle), (qubits[i],))
