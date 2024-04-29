# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Slater determinant preparation gate."""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterator, Sequence

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitInstruction, Gate, Qubit
from qiskit.circuit.library import XGate, XXPlusYYGate
from scipy.linalg.lapack import zrot

from ffsim.linalg.givens import GivensRotation, zrotg
from ffsim.spin import Spin


class PrepareSlaterDeterminantJW(Gate):
    r"""Gate that prepares a Slater determinant (under JWT) from the all zeros state.

    A Slater determinant is a state of the form

    .. math::

        \prod_\sigma b^\dagger_{\sigma, 1} \cdots b^\dagger_{\sigma, N_f}
        \lvert \text{vac} \rangle,

    where

    .. math::

        b^\dagger_{\sigma, i} = \sum_{k = 1}^N Q_{ji} a^\dagger_{\sigma, j}.

    - :math:`Q` is an :math:`N \times N_f` matrix with orthonormal columns.
    - :math:`\lvert \text{vac} \rangle` is the vacuum state.

    This gate assumes the Jordan-Wigner transformation (JWT).

    This gate is meant to be applied to the all zeros state. Its behavior when applied
    to any other state is not guaranteed. The global phase of the prepared state may
    differ from the given equation.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.

    Reference: `arXiv:1711.05395`_

    .. _arXiv:1711.05395: https://arxiv.org/abs/1711.05395
    """

    def __init__(
        self,
        orbital_coeffs: np.ndarray,
        spin: Spin = Spin.ALPHA_AND_BETA,
        label: str | None = None,
        validate: bool = True,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        """Create new Slater determinant preparation gate.

        Args:
            orbital_coeffs: The matrix :math:`Q` that specifies the coefficients of the
                new creation operators in terms of the original creation operators.
                The columns of the matrix must be orthonormal.
            spin: Choice of spin sector(s) to act on.

                - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
                - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
                - To act on both spin alpha and spin beta, pass
                  :const:`ffsim.Spin.ALPHA_AND_BETA` (this is the default value).

            label: The label of the gate.
            validate: Whether to validate the inputs.
            rtol: Relative numerical tolerance for input validation.
            atol: Absolute numerical tolerance for input validation.

        Raises:
            ValueError: orbital_coeffs must be a 2-dimensional array.
            ValueError: orbital_coeffs must have orthonormal columns.
        """
        if validate and not _columns_are_orthonormal(
            orbital_coeffs, rtol=rtol, atol=atol
        ):
            raise ValueError(
                "The input orbital_coeffs did not have orthonormal columns."
            )
        self.orbital_coeffs = orbital_coeffs
        self.spin = spin
        norb, _ = orbital_coeffs.shape
        if spin is Spin.ALPHA:
            name = "slater_jw_a"
        elif spin is Spin.BETA:
            name = "slater_jw_b"
        else:
            name = "slater_jw"
        super().__init__(name, 2 * norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        norb = len(qubits) // 2
        alpha_qubits = qubits[:norb]
        beta_qubits = qubits[norb:]
        if self.spin & Spin.ALPHA:
            for instruction in _prepare_slater_determinant_jw(
                alpha_qubits, self.orbital_coeffs.T
            ):
                circuit.append(instruction)
        if self.spin & Spin.BETA:
            for instruction in _prepare_slater_determinant_jw(
                beta_qubits, self.orbital_coeffs.T
            ):
                circuit.append(instruction)
        self.definition = circuit


def _columns_are_orthonormal(
    mat: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    _, n = mat.shape
    return np.allclose(mat.T.conj() @ mat, np.eye(n), rtol=rtol, atol=atol)


def _prepare_slater_determinant_jw(
    qubits: Sequence[Qubit], orbital_coeffs: np.ndarray
) -> Iterator[CircuitInstruction]:
    m, n = orbital_coeffs.shape

    # set the first n_particles qubits to 1
    for i in range(m):
        yield CircuitInstruction(XGate(), (qubits[i],))

    # if all orbitals are filled, no further operations are needed
    if m == n:
        return

    # yield Givens rotations
    givens_rotations = _givens_decomposition_slater(orbital_coeffs)
    for c, s, i, j in givens_rotations:
        yield CircuitInstruction(
            XXPlusYYGate(2 * math.acos(c), cmath.phase(s) - 0.5 * math.pi),
            (qubits[i], qubits[j]),
        )


def _givens_decomposition_slater(orbital_coeffs: np.ndarray) -> list[GivensRotation]:
    m, n = orbital_coeffs.shape

    current_matrix = orbital_coeffs.copy()

    # zero out top right corner by rotating rows; this is a no-op
    for j in reversed(range(n - m + 1, n)):
        # Zero out entries in column j
        for i in range(m - n + j):
            # Zero out entry in row i if needed
            if not cmath.isclose(current_matrix[i, j], 0):
                c, s = zrotg(current_matrix[i + 1, j], current_matrix[i, j])
                (
                    current_matrix[i + 1],
                    current_matrix[i],
                ) = zrot(
                    current_matrix[i + 1],
                    current_matrix[i],
                    c,
                    s,
                )

    # decompose matrix into Givens rotations
    rotations = []
    for i in range(m):
        # zero out the columns in row i
        for j in range(n - m + i, i, -1):
            if not cmath.isclose(current_matrix[i, j], 0):
                # zero out element j of row i
                c, s = zrotg(current_matrix[i, j - 1], current_matrix[i, j])
                rotations.append(GivensRotation(c, s, j, j - 1))
                current_matrix = current_matrix.T.copy()
                (
                    current_matrix[j - 1],
                    current_matrix[j],
                ) = zrot(
                    current_matrix[j - 1],
                    current_matrix[j],
                    c,
                    s,
                )
                current_matrix = current_matrix.T

    return rotations[::-1]
