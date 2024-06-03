# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""(Local) unitary cluster Jastrow ansatz gate."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)

from ffsim import variational
from ffsim.qiskit.gates.diag_coulomb import (
    DiagCoulombEvolutionJW,
    DiagCoulombEvolutionSpinlessJW,
)
from ffsim.qiskit.gates.orbital_rotation import (
    OrbitalRotationJW,
    OrbitalRotationSpinlessJW,
)


class UCJOpSpinBalancedJW(Gate):
    """Spin-balanced UCJ operator under the Jordan-Wigner transformation.

    See :class:`ffsim.UCJOpSpinBalanced` for a description of this gate's unitary.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.
    """

    def __init__(
        self, ucj_op: variational.UCJOpSpinBalanced, *, label: str | None = None
    ):
        """Create a new spin-balanced unitary cluster Jastrow (UCJ) gate.

        Args:
            ucj_op: The UCJ operator.
            label: The label of the gate.
        """
        self.ucj_op = ucj_op
        super().__init__("ucj_balanced_jw", 2 * ucj_op.norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _ucj_op_spin_balanced_jw(qubits, self.ucj_op),
            qubits=qubits,
            name=self.name,
        )


def _ucj_op_spin_balanced_jw(
    qubits: Sequence[Qubit], ucj_op: variational.UCJOpSpinBalanced
) -> Iterator[CircuitInstruction]:
    for (diag_coulomb_mat_aa, diag_coulomb_mat_ab), orbital_rotation in zip(
        ucj_op.diag_coulomb_mats, ucj_op.orbital_rotations
    ):
        yield CircuitInstruction(
            OrbitalRotationJW(ucj_op.norb, orbital_rotation.T.conj()),
            qubits,
        )
        yield CircuitInstruction(
            DiagCoulombEvolutionJW(
                ucj_op.norb,
                (diag_coulomb_mat_aa, diag_coulomb_mat_ab, diag_coulomb_mat_aa),
                -1.0,
            ),
            qubits,
        )
        yield CircuitInstruction(
            OrbitalRotationJW(ucj_op.norb, orbital_rotation), qubits
        )
    if ucj_op.final_orbital_rotation is not None:
        yield CircuitInstruction(
            OrbitalRotationJW(ucj_op.norb, ucj_op.final_orbital_rotation), qubits
        )


class UCJOpSpinUnbalancedJW(Gate):
    """Spin-unbalanced UCJ operator under the Jordan-Wigner transformation.

    See :class:`ffsim.UCJOpSpinUnbalanced` for a description of this gate's unitary.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.
    """

    def __init__(
        self, ucj_op: variational.UCJOpSpinUnbalanced, *, label: str | None = None
    ):
        """Create a new spin-unbalanced unitary cluster Jastrow (UCJ) gate.

        Args:
            ucj_op: The UCJ operator.
            label: The label of the gate.
        """
        self.ucj_op = ucj_op
        super().__init__("ucj_unbalanced_jw", 2 * ucj_op.norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _ucj_op_spin_unbalanced_jw(qubits, self.ucj_op),
            qubits=qubits,
            name=self.name,
        )


def _ucj_op_spin_unbalanced_jw(
    qubits: Sequence[Qubit], ucj_op: variational.UCJOpSpinUnbalanced
) -> Iterator[CircuitInstruction]:
    for diag_colomb_mat, orbital_rotation in zip(
        ucj_op.diag_coulomb_mats, ucj_op.orbital_rotations
    ):
        yield CircuitInstruction(
            OrbitalRotationJW(ucj_op.norb, orbital_rotation.transpose(0, 2, 1).conj()),
            qubits,
        )
        yield CircuitInstruction(
            DiagCoulombEvolutionJW(ucj_op.norb, diag_colomb_mat, -1.0),
            qubits,
        )
        yield CircuitInstruction(
            OrbitalRotationJW(ucj_op.norb, orbital_rotation), qubits
        )
    if ucj_op.final_orbital_rotation is not None:
        yield CircuitInstruction(
            OrbitalRotationJW(ucj_op.norb, ucj_op.final_orbital_rotation), qubits
        )


class UCJOpSpinlessJW(Gate):
    """Spinless UCJ operator under the Jordan-Wigner transformation.

    See :class:`ffsim.UCJOpSpinless` for a description of this gate's unitary.
    """

    def __init__(self, ucj_op: variational.UCJOpSpinless, *, label: str | None = None):
        """Create a new spinless unitary cluster Jastrow (UCJ) gate.

        Args:
            ucj_op: The UCJ operator.
            label: The label of the gate.
        """
        self.ucj_op = ucj_op
        super().__init__("ucj_spinless_jw", ucj_op.norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _ucj_op_spinless_jw(qubits, self.ucj_op),
            qubits=qubits,
            name=self.name,
        )


def _ucj_op_spinless_jw(
    qubits: Sequence[Qubit], ucj_op: variational.UCJOpSpinless
) -> Iterator[CircuitInstruction]:
    for diag_coulomb_mat, orbital_rotation in zip(
        ucj_op.diag_coulomb_mats, ucj_op.orbital_rotations
    ):
        yield CircuitInstruction(
            OrbitalRotationSpinlessJW(ucj_op.norb, orbital_rotation.T.conj()),
            qubits,
        )
        yield CircuitInstruction(
            DiagCoulombEvolutionSpinlessJW(ucj_op.norb, diag_coulomb_mat, -1.0),
            qubits,
        )
        yield CircuitInstruction(
            OrbitalRotationSpinlessJW(ucj_op.norb, orbital_rotation), qubits
        )
    if ucj_op.final_orbital_rotation is not None:
        yield CircuitInstruction(
            OrbitalRotationSpinlessJW(ucj_op.norb, ucj_op.final_orbital_rotation),
            qubits,
        )
