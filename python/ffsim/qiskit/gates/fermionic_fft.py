# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import cmath
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

from ffsim import linalg


class FermionicFFTJW(Gate):
    r"""Fermionic fast Fourier transform under the Jordan-Wigner transformation.

    Performs the discrete Fourier transform (DFT) on creation operators:

    .. math::
        a^\dagger_k \mapsto \frac{1}{\sqrt{N}}
        \sum_{n=0}^{N-1} e^{-i 2\pi k n / N} a^\dagger_n

    Implemented using a Cooley-Tukey quantum decomposition that minimizes the
    number of distinct gate parameters.

    Assumes qubits are ordered with the first ``norb`` qubits for spin alpha
    and the next ``norb`` qubits for spin beta.
    """

    def __init__(
        self,
        norb: int,
        *,
        label: str | None = None,
    ):
        """Create new fermionic FFT gate.

        Args:
            norb: The number of spatial orbitals.
            label: Optional label for the gate.
        """
        self.norb = norb
        super().__init__("ffft_jw", 2 * norb, [], label=label)

    def _define(self):
        """Gate decomposition."""
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        norb = len(qubits) // 2
        alpha_qubits = qubits[:norb]
        beta_qubits = qubits[norb:]

        for instruction in _fermionic_fft_jw(alpha_qubits):
            circuit.append(instruction)
        for instruction in _fermionic_fft_jw(beta_qubits):
            circuit.append(instruction)

        self.definition = circuit

    def inverse(self):
        """Inverse gate (Inverse Fermionic FFT)."""
        return FermionicFFTJW(self.norb).conjugate()


class FermionicFFTSpinlessJW(Gate):
    r"""Spinless fermionic fast Fourier transform gate acting on `norb` qubits."""

    def __init__(self, norb: int, *, label: str | None = None):
        self.norb = norb
        super().__init__("ffft_spinless_jw", norb, [], label=label)

    def _define(self):
        qubits = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(qubits, name=self.name)
        for instruction in _fermionic_fft_jw(qubits):
            circuit.append(instruction)
        self.definition = circuit

    def inverse(self):
        return FermionicFFTSpinlessJW(self.norb).conjugate()


def _fermionic_fft_jw(qubits: Sequence[Qubit]) -> Iterator[CircuitInstruction]:
    """Yield instructions executing a Cooley-Tukey fermionic FFT on the given qubits.

    For powers of 2, uses an efficient Cooley-Tukey decomposition requiring only
    log2(N) distinct angle classes for twiddle factors / Givens rotations.
    For arbitrary N, falls back to standard Givens decomposition of the Fourier matrix.
    """
    n = len(qubits)
    if n <= 1:
        return

    # Check if power of 2
    if n & (n - 1) == 0:
        yield from _cooley_tukey_ffft_jw(qubits)
    else:
        # Fallback to dense Fourier matrix Givens decomposition
        fourier_mat = _dft_matrix(n)
        givens_rotations, phase_shifts = linalg.givens_decomposition(fourier_mat)

        for c, s, i, j in givens_rotations:
            c_angle = math.acos(c)
            if c_angle:
                yield CircuitInstruction(
                    XXPlusYYGate(2 * c_angle, cmath.phase(s) - 0.5 * math.pi),
                    (qubits[i], qubits[j]),
                )
        for i, phase_shift in enumerate(phase_shifts):
            phase = cmath.phase(phase_shift)
            if phase:
                yield CircuitInstruction(PhaseGate(phase), (qubits[i]))


def _cooley_tukey_ffft_jw(qubits: Sequence[Qubit]) -> Iterator[CircuitInstruction]:
    """Cooley-Tukey recursive algorithm for N = 2^k qubits under Jordan-Wigner."""
    n = len(qubits)
    if n == 1:
        return

    half = n // 2
    even_qubits = qubits[::2]
    odd_qubits = qubits[1::2]

    # Recursive sub-FFTs
    yield from _cooley_tukey_ffft_jw(even_qubits)
    yield from _cooley_tukey_ffft_jw(odd_qubits)

    # Twiddle factors and 2-point fermionic Fourier.
    # A 2-point fermionic FFT butterfly between mode k (even) and
    # mode k (odd) with phase theta:
    # U = 1/sqrt(2) * [[1, e^{-i theta}], [1, -e^{-i theta}]]
    for k in range(half):
        theta = 2 * math.pi * k / n
        q_even = even_qubits[k]
        q_odd = odd_qubits[k]

        # Apply phase shift (twiddle factor) to odd mode
        if theta != 0:
            yield CircuitInstruction(PhaseGate(-theta), (q_odd,))

        # 2-point Fourier transform gate on (even, odd) pair
        # Represented via XXPlusYYGate with angle pi/2 and phase shift
        yield CircuitInstruction(
            XXPlusYYGate(math.pi / 2, -math.pi / 2),
            (q_even, q_odd),
        )


def _dft_matrix(n: int) -> np.ndarray:
    """Generate the unitary n x n DFT matrix.

    This is the equivalent of scipy.linalg.dft(n, scale="sqrtn).
    The matrix elements are defined as:

    .. math::
        F_{k, j} = \\frac{1}{\\sqrt{n}} e^{-i 2\\pi k j / n}

    Example:
        >>> _fourier_matrix(2)
        array([[ 1/sqrt(2),  1/sqrt(2)],
               [ 1/sqrt(2), -1/sqrt(2)]])

    Args:
        n: The dimension of the Fourier transform matrix.

    Returns:
        n x n unitary DFT matrix.
    """
    mat = np.zeros((n, n), dtype=complex)
    omega = np.exp(-2j * np.pi / n)
    for k in range(n):
        for j in range(n):
            mat[k, j] = (omega ** (k * j)) / np.sqrt(n)
    return mat
