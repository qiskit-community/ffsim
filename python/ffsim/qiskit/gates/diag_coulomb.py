# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Diagonal Coulomb evolution gate."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence

import numpy as np
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import CPhaseGate, PhaseGate, RZZGate


class DiagCoulombEvolutionJW(Gate):
    r"""Diagonal Coulomb evolution under the Jordan-Wigner transformation.

    The diagonal Coulomb evolution gate has the unitary

    .. math::

        \exp\left(-i t \sum_{\sigma, \tau, i, j}
        Z^{(\sigma \tau)}_{ij} n_{\sigma, i} n_{\tau, j} / 2\right)

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, :math:`Z^{(\sigma \tau)}` is a real-valued matrix.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.
    """

    def __init__(
        self,
        norb: int,
        mat: np.ndarray
        | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
        time: float,
        *,
        z_representation: bool = False,
        label: str | None = None,
    ):
        r"""Create new diagonal Coulomb evolution gate.

        Args:
            norb: The number of spatial orbitals.
            mat: The diagonal Coulomb matrix :math:`Z`.
                You can pass either a single Numpy array specifying the coefficients
                to use for all spin interactions, or you can pass a tuple of three Numpy
                arrays specifying independent coefficients for alpha-alpha, alpha-beta,
                and beta-beta interactions (in that order). If passing a tuple, you can
                set a tuple element to ``None`` to indicate the absence of interactions
                of that type. The alpha-alpha and beta-beta matrices are assumed to be
                symmetric, and only their upper triangular entries are used.
            time: The evolution time.
            z_representation: Whether the input matrices are in the "Z" representation.
            label: The label of the gate.
        """
        self.norb = norb
        self.mat = mat
        self.time = time
        self.z_representation = z_representation
        super().__init__("diag_coulomb_jw", 2 * norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        generate_instructions = (
            _diag_coulomb_evo_z_rep_jw
            if self.z_representation
            else _diag_coulomb_evo_num_rep_jw
        )
        self.definition = QuantumCircuit.from_instructions(
            generate_instructions(qubits, mat=self.mat, time=self.time, norb=self.norb),
            qubits=qubits,
        )

    def inverse(self):
        """Inverse gate."""
        return DiagCoulombEvolutionJW(
            self.norb, self.mat, -self.time, z_representation=self.z_representation
        )


class DiagCoulombEvolutionSpinlessJW(Gate):
    r"""Spinless diagonal Coulomb evolution under the Jordan-Wigner transformation.

    The spinless diagonal Coulomb evolution gate has the unitary

    .. math::

        \exp\left(-i t \sum_{i, j}
        Z^{ij} n_i n_j / 2\right)

    where :math:`n_i` denotes the number operator on orbital :math:`i` and
    :math:`Z` is a real symmetric matrix.
    """

    def __init__(
        self,
        norb: int,
        mat: np.ndarray,
        time: float,
        *,
        label: str | None = None,
    ):
        r"""Create new diagonal Coulomb evolution gate.

        Args:
            norb: The number of spatial orbitals.
            mat: The diagonal Coulomb matrix :math:`Z`.
                It is assumed to be symmetric, and only its upper triangular entries
                are used.
            time: The evolution time.
            label: The label of the gate.
        """
        self.norb = norb
        self.mat = mat
        self.time = time
        super().__init__("diag_coulomb_spinless_jw", norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _diag_coulomb_evo_num_rep_spinless_jw(
                qubits, mat=self.mat, time=self.time, norb=self.norb
            ),
            qubits=qubits,
        )

    def inverse(self):
        """Inverse gate."""
        return DiagCoulombEvolutionSpinlessJW(
            self.norb, self.mat, -self.time, z_representation=self.z_representation
        )


def _diag_coulomb_evo_num_rep_spinless_jw(
    qubits: Sequence[Qubit], mat: np.ndarray, time: float, norb: int
) -> Iterator[CircuitInstruction]:
    assert len(qubits) == norb
    for i in range(norb):
        if mat[i, i]:
            yield CircuitInstruction(PhaseGate(-0.5 * mat[i, i] * time), (qubits[i],))
    for i, j in itertools.combinations(range(norb), 2):
        if mat[i, j]:
            yield CircuitInstruction(
                CPhaseGate(-mat[i, j] * time), (qubits[i], qubits[j])
            )


def _diag_coulomb_evo_num_rep_jw(
    qubits: Sequence[Qubit],
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
) -> Iterator[CircuitInstruction]:
    assert len(qubits) == 2 * norb
    mat_aa: np.ndarray | None
    mat_ab: np.ndarray | None
    mat_bb: np.ndarray | None
    if isinstance(mat, np.ndarray) and mat.ndim == 2:
        mat_aa, mat_ab, mat_bb = mat, mat, mat
    else:
        mat_aa, mat_ab, mat_bb = mat

    # gates that involve a single spin sector
    for sigma, this_mat in enumerate([mat_aa, mat_bb]):
        if this_mat is not None:
            for i in range(norb):
                if this_mat[i, i]:
                    yield CircuitInstruction(
                        PhaseGate(-0.5 * this_mat[i, i] * time),
                        (qubits[i + sigma * norb],),
                    )
            for i, j in itertools.combinations(range(norb), 2):
                if this_mat[i, j]:
                    yield CircuitInstruction(
                        CPhaseGate(-this_mat[i, j] * time),
                        (qubits[i + sigma * norb], qubits[j + sigma * norb]),
                    )

    # gates that involve both spin sectors
    if mat_ab is not None:
        for i in range(norb):
            if mat_ab[i, i]:
                yield CircuitInstruction(
                    CPhaseGate(-mat_ab[i, i] * time),
                    (qubits[i], qubits[i + norb]),
                )
        for i, j in itertools.combinations(range(norb), 2):
            if mat_ab[i, j]:
                yield CircuitInstruction(
                    CPhaseGate(-mat_ab[i, j] * time),
                    (qubits[i], qubits[j + norb]),
                )
            if mat_ab[j, i]:
                yield CircuitInstruction(
                    CPhaseGate(-mat_ab[j, i] * time),
                    (qubits[j], qubits[i + norb]),
                )


def _diag_coulomb_evo_z_rep_jw(
    qubits: Sequence[Qubit],
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
) -> Iterator[CircuitInstruction]:
    assert len(qubits) == 2 * norb
    mat_aa: np.ndarray | None
    mat_ab: np.ndarray | None
    mat_bb: np.ndarray | None
    if isinstance(mat, np.ndarray) and mat.ndim == 2:
        mat_aa, mat_ab, mat_bb = mat, mat, mat
    else:
        mat_aa, mat_ab, mat_bb = mat
    for i, j in itertools.combinations(range(2 * norb), 2):
        if (i < norb) and (j < norb):
            this_mat = mat_aa
        elif (i >= norb) and (j >= norb):
            this_mat = mat_bb
        else:
            this_mat = mat_ab
        if this_mat is not None and this_mat[i % norb, j % norb]:
            yield CircuitInstruction(
                RZZGate(0.5 * this_mat[i % norb, j % norb] * time),
                (qubits[i], qubits[j]),
            )
