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

from ffsim import linalg


class DiagCoulombEvolutionJW(Gate):
    r"""Diagonal Coulomb evolution under the Jordan-Wigner transformation.

    The diagonal Coulomb evolution gate has the unitary

    .. math::

        \exp\left(-i t \sum_{\sigma, \tau, i, j}
        Z_{ij} n_{\sigma, i} n_{\tau, j} / 2\right)

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, :math:`Z` is a real symmetric matrix.
    If `mat_alpha_beta` is also given, then it is used in place of :math:`Z`
    for the terms in the sum where the spins differ (:math:`\sigma \neq \tau`).

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.
    """

    def __init__(
        self,
        mat: np.ndarray,
        time: float,
        *,
        mat_alpha_beta: np.ndarray | None = None,
        z_representation: bool = False,
        label: str | None = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        r"""Create new diagonal Coulomb evolution gate.

        Args:
            mat: The real symmetric matrix :math:`Z`.
            time: The evolution time.
            mat_alpha_beta: A matrix of coefficients to use for interactions between
                orbitals with differing spin.
            z_representation: Whether the input matrices are in the "Z" representation.
            label: The label of the gate.
            validate: Whether to check that the input matrix is real symmetric and
                raise an error if it isn't.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.

        Raises:
            ValueError: The input matrix is not real symmetric.
        """
        if validate:
            if not linalg.is_real_symmetric(mat, rtol=rtol, atol=atol):
                raise ValueError("The input matrix is not real symmetric.")
            if mat_alpha_beta is not None and not linalg.is_real_symmetric(
                mat_alpha_beta, rtol=rtol, atol=atol
            ):
                raise ValueError("The input alpha-beta matrix is not real symmetric.")
        self.mat = mat
        self.time = time
        self.mat_alpha_beta = mat_alpha_beta
        self.z_representation = z_representation
        norb, _ = mat.shape
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
            generate_instructions(
                qubits,
                mat=self.mat,
                time=self.time,
                mat_alpha_beta=self.mat_alpha_beta
                if self.mat_alpha_beta is not None
                else self.mat,
            ),
            qubits=qubits,
        )

    def inverse(self):
        """Inverse gate."""
        return DiagCoulombEvolutionJW(
            self.mat,
            -self.time,
            mat_alpha_beta=self.mat_alpha_beta,
            z_representation=self.z_representation,
            validate=False,
        )


def _diag_coulomb_evo_num_rep_jw(
    qubits: Sequence[Qubit],
    mat: np.ndarray,
    time: float,
    mat_alpha_beta: np.ndarray,
) -> Iterator[CircuitInstruction]:
    norb, _ = mat.shape
    assert norb == len(qubits) // 2

    # gates that involve a single spin sector
    for sigma in range(2):
        for i in range(norb):
            yield CircuitInstruction(
                PhaseGate(-0.5 * mat[i % norb, i % norb] * time),
                (qubits[i + sigma * norb],),
            )
        for i, j in itertools.combinations(range(norb), 2):
            yield CircuitInstruction(
                CPhaseGate(-mat[i % norb, j % norb] * time),
                (qubits[i + sigma * norb], qubits[j + sigma * norb]),
            )

    # gates that involve both spin sectors
    for i in range(norb):
        yield CircuitInstruction(
            CPhaseGate(-mat_alpha_beta[i % norb, i % norb] * time),
            (qubits[i], qubits[i + norb]),
        )
    for i, j in itertools.combinations(range(norb), 2):
        yield CircuitInstruction(
            CPhaseGate(-mat_alpha_beta[i % norb, j % norb] * time),
            (qubits[i], qubits[j + norb]),
        )
        yield CircuitInstruction(
            CPhaseGate(-mat_alpha_beta[i % norb, j % norb] * time),
            (qubits[i + norb], qubits[j]),
        )


def _diag_coulomb_evo_z_rep_jw(
    qubits: Sequence[Qubit],
    mat: np.ndarray,
    time: float,
    mat_alpha_beta: np.ndarray,
) -> Iterator[CircuitInstruction]:
    norb, _ = mat.shape
    assert norb == len(qubits) // 2
    for i, j in itertools.combinations(range(len(qubits)), 2):
        this_mat = mat if (i < norb) == (j < norb) else mat_alpha_beta
        yield CircuitInstruction(
            RZZGate(0.5 * this_mat[i % norb, j % norb] * time),
            (qubits[i], qubits[j]),
        )
