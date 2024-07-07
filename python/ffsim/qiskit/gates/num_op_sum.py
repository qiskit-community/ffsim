# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Number operator sum evolution gate."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import PhaseGate


class NumOpSumEvolutionJW(Gate):
    r"""Number operator sum evolution under the Jordan-Wigner transformation.

    The number operator sum evolution gate has the unitary

    .. math::

        \exp\left(-i t \sum_{\sigma, i} \lambda^{(\sigma)}_i n_{\sigma, i}\right)

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma` and the :math:`\lambda_i` are real numbers.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.
    """

    def __init__(
        self,
        norb: int,
        coeffs: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
        time: float,
        *,
        label: str | None = None,
    ):
        r"""Create new number operator sum evolution gate.

        Args:
            norb: The number of spatial orbitals.
            coeffs: The coefficients of the linear combination.
                You can pass either a single Numpy array specifying the coefficients
                to apply to both spin sectors, or you can pass a pair of Numpy arrays
                specifying independent coefficients for spin alpha and spin beta.
                If passing a pair, you can use ``None`` for one of the
                values in the pair to indicate that no operation should be applied to
                that spin sector.
            time: The evolution time.
            label: The label of the gate.
        """
        self.norb = norb
        self.coeffs = coeffs
        self.time = time
        super().__init__("num_op_sum_jw", 2 * norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _num_op_sum_evo_jw(
                qubits, coeffs=self.coeffs, time=self.time, norb=self.norb
            ),
            qubits=qubits,
        )

    def inverse(self):
        """Inverse gate."""
        return NumOpSumEvolutionJW(self.norb, self.coeffs, -self.time)


class NumOpSumEvolutionSpinlessJW(Gate):
    r"""Spinless number operator sum evolution under the Jordan-Wigner transformation.

    The spinless number operator sum evolution gate has the unitary

    .. math::

        \exp\left(-i t \sum_{i} \lambda_i n_{i}\right)

    where :math:`n_i` denotes the number operator on orbital :math:`i` and the
    :math:`\lambda_i` are real numbers.
    """

    def __init__(
        self,
        norb: int,
        coeffs: np.ndarray,
        time: float,
        *,
        label: str | None = None,
    ):
        r"""Create new number operator sum evolution gate.

        Args:
            norb: The number of spatial orbitals.
            coeffs: The coefficients of the linear combination.
            time: The evolution time.
            label: The label of the gate.
        """
        self.norb = norb
        self.coeffs = coeffs
        self.time = time
        super().__init__("num_op_sum_spinless_jw", norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        self.definition = QuantumCircuit.from_instructions(
            _num_op_sum_evo_spinless_jw(
                qubits, coeffs=self.coeffs, time=self.time, norb=self.norb
            ),
            qubits=qubits,
        )

    def inverse(self):
        """Inverse gate."""
        return NumOpSumEvolutionSpinlessJW(self.norb, self.coeffs, -self.time)


def _num_op_sum_evo_spinless_jw(
    qubits: Sequence[Qubit], coeffs: np.ndarray, time: float, norb: int
) -> Iterator[CircuitInstruction]:
    assert len(qubits) == norb
    for i in range(norb):
        if coeffs[i]:
            yield CircuitInstruction(PhaseGate(-coeffs[i] * time), (qubits[i],))


def _num_op_sum_evo_jw(
    qubits: Sequence[Qubit],
    coeffs: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
) -> Iterator[CircuitInstruction]:
    assert len(qubits) == 2 * norb
    coeffs_a: np.ndarray | None
    coeffs_b: np.ndarray | None
    if isinstance(coeffs, np.ndarray) and coeffs.ndim == 1:
        coeffs_a, coeffs_b = coeffs, coeffs
    else:
        coeffs_a, coeffs_b = coeffs

    # gates that involve a single spin sector
    for sigma, these_coeffs in enumerate([coeffs_a, coeffs_b]):
        if these_coeffs is not None:
            for i in range(norb):
                if these_coeffs[i]:
                    yield CircuitInstruction(
                        PhaseGate(-these_coeffs[i] * time), (qubits[i + sigma * norb],)
                    )
