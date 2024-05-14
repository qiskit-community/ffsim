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

import itertools
import math
from collections.abc import Iterator, Sequence

from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import XXPlusYYGate

from ffsim.variational import GivensAnsatzOperator


class GivensAnsatzOperatorJW(Gate):
    """Givens rotation ansatz operator under the Jordan-Wigner transformation.

    See :class:`ffsim.GivensAnsatzOperator` for a description of this gate's unitary.
    """

    def __init__(
        self, givens_ansatz_operator: GivensAnsatzOperator, *, label: str | None = None
    ):
        """Create a new Givens ansatz operator gate.

        Args:
            givens_ansatz_operator: The Givens rotation ansatz operator.
            label: The label of the gate.
        """
        self.givens_ansatz_operator = givens_ansatz_operator
        super().__init__(
            "givens_ansatz_jw", 2 * givens_ansatz_operator.norb, [], label=label
        )

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        norb = len(qubits) // 2
        alpha_qubits = qubits[:norb]
        beta_qubits = qubits[norb:]
        for instruction in _givens_ansatz_jw(alpha_qubits, self.givens_ansatz_operator):
            circuit.append(instruction)
        for instruction in _givens_ansatz_jw(beta_qubits, self.givens_ansatz_operator):
            circuit.append(instruction)
        self.definition = circuit


class GivensAnsatzOperatorSpinlessJW(Gate):
    """Givens rotation ansatz operator under the Jordan-Wigner transformation, spinless.

    Like :class:`GivensAnsatzOperatorJW` but only acts on a single spin species.
    """

    def __init__(
        self, givens_ansatz_operator: GivensAnsatzOperator, *, label: str | None = None
    ):
        """Create a new Givens ansatz operator gate.

        Args:
            givens_ansatz_operator: The Givens rotation ansatz operator.
            label: The label of the gate.
        """
        self.givens_ansatz_operator = givens_ansatz_operator
        super().__init__(
            "givens_ansatz_jw", givens_ansatz_operator.norb, [], label=label
        )

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _givens_ansatz_jw(qubits, self.givens_ansatz_operator),
            qubits=qubits,
            name=self.name,
        )


def _givens_ansatz_jw(
    qubits: Sequence[Qubit], givens_ansatz_operator: GivensAnsatzOperator
) -> Iterator[CircuitInstruction]:
    for (i, j), theta in zip(
        itertools.cycle(givens_ansatz_operator.interaction_pairs),
        givens_ansatz_operator.thetas,
    ):
        yield CircuitInstruction(
            XXPlusYYGate(2 * theta, -0.5 * math.pi), (qubits[i], qubits[j])
        )
