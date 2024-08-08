# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Diagonal Coulomb Trotter evolution gate."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence

import numpy as np
import scipy.linalg
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import GlobalPhaseGate

from ffsim.hamiltonians import DiagonalCoulombHamiltonian
from ffsim.qiskit.gates.diag_coulomb import DiagCoulombEvolutionJW
from ffsim.qiskit.gates.num_op_sum import NumOpSumEvolutionJW
from ffsim.qiskit.gates.orbital_rotation import OrbitalRotationJW
from ffsim.trotter._util import simulate_trotter_step_iterator


class SimulateTrotterDiagCoulombSplitOpJW(Gate):
    r"""Split operator Trotter evolution of diagonal Coulomb Hamiltonian, Jordan-Wigner.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.
    """

    def __init__(
        self,
        hamiltonian: DiagonalCoulombHamiltonian,
        time: float,
        *,
        n_steps: int = 1,
        order: int = 0,
        label: str | None = None,
    ):
        r"""Create diagonal Coulomb split-operator Trotter evolution gate.

        Args:
            norb: The number of spatial orbitals.
            hamiltonian: The Hamiltonian.
            time: The evolution time.
            n_steps: The number of Trotter steps.
            order: The order of the Trotter decomposition.
            label: The label of the gate.
        """
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}.")
        if n_steps < 0:
            raise ValueError(f"n_steps must be non-negative, got {n_steps}.")
        self.hamiltonian = hamiltonian
        self.time = time
        self.n_steps = n_steps
        self.order = order
        super().__init__(
            "dc_trotter_split_op_jw", 2 * self.hamiltonian.norb, [], label=label
        )

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _simulate_trotter_diag_coulomb_split_op(
                qubits,
                self.hamiltonian,
                time=self.time,
                n_steps=self.n_steps,
                order=self.order,
            ),
            qubits=qubits,
        )


def _simulate_trotter_diag_coulomb_split_op(
    qubits: Sequence[Qubit],
    hamiltonian: DiagonalCoulombHamiltonian,
    time: float,
    n_steps: int = 1,
    order: int = 0,
) -> Iterator[CircuitInstruction]:
    if n_steps == 0:
        return

    one_body_energies, one_body_basis_change = scipy.linalg.eigh(
        hamiltonian.one_body_tensor
    )
    step_time = time / n_steps

    current_basis = np.eye(hamiltonian.norb, dtype=complex)
    for _ in range(n_steps):
        current_basis = yield from _simulate_trotter_step_diag_coulomb_split_op(
            qubits,
            current_basis,
            one_body_energies,
            one_body_basis_change,
            hamiltonian.diag_coulomb_mats,
            step_time,
            norb=hamiltonian.norb,
            order=order,
        )
    yield CircuitInstruction(OrbitalRotationJW(hamiltonian.norb, current_basis), qubits)
    yield CircuitInstruction(GlobalPhaseGate(-time * hamiltonian.constant), [])


def _simulate_trotter_step_diag_coulomb_split_op(
    qubits: Sequence[Qubit],
    current_basis: np.ndarray,
    one_body_energies: np.ndarray,
    one_body_basis_change: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    time: float,
    norb: int,
    order: int,
) -> Generator[CircuitInstruction, None, np.ndarray]:
    diag_coulomb_aa, diag_coulomb_ab = diag_coulomb_mats
    eye = np.eye(norb)
    for term_index, time in simulate_trotter_step_iterator(2, time, order):
        if term_index == 0:
            yield CircuitInstruction(
                OrbitalRotationJW(norb, one_body_basis_change.T.conj() @ current_basis),
                qubits,
            )
            yield CircuitInstruction(
                NumOpSumEvolutionJW(norb, coeffs=one_body_energies, time=time), qubits
            )
            current_basis = one_body_basis_change
        else:
            yield CircuitInstruction(OrbitalRotationJW(norb, current_basis), qubits)
            yield CircuitInstruction(
                DiagCoulombEvolutionJW(
                    norb,
                    (diag_coulomb_aa, diag_coulomb_ab, diag_coulomb_aa),
                    time,
                ),
                qubits,
            )
            current_basis = eye
    return current_basis
