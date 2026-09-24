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

import math
from collections.abc import Iterator, Sequence

import numpy as np
import scipy
from qiskit.circuit import (
    CircuitInstruction,
    Gate,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.library import PhaseGate

from ffsim.qiskit.gates.orbital_rotation import OrbitalRotationSpinlessJW


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


def _fermionic_fft_jw(qubits: Sequence[Qubit]) -> Iterator[CircuitInstruction]:
    """Yield instructions executing a Cooley-Tukey fermionic FFT on the given qubits."""
    n = len(qubits)
    if n <= 1:
        return
    yield from _cooley_tukey_ffft_jw(qubits, _prime_factors(n))


def _prime_factors(n: int) -> list[int]:
    """Return the prime factors of n, with multiplicity, using trial division."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def _cooley_tukey_ffft_jw(
    qubits: Sequence[Qubit],
    factors: Sequence[int],
) -> Iterator[CircuitInstruction]:
    r"""Cooley-Tukey fermionic FFT under Jordan-Wigner.

    Recursively applies the general Cooley-Tukey factorization N = N_1 N_2, with
    the smallest prime factor N_2 as the radix at each stage (mixed-radix,
    decimation in frequency (DIF)).

    Follows the general factorization described in the "Variations"
    section of https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm.

    Composite size :math:`N = N_1 N_2`, where N_2 = factors[0] is prime, with
    input index :math:`n = N_1 n_2 + n_1`
    and output index :math:`k = N_2 k_1 + k_2`:

    .. math::
        X_{N_2 k_1 + k_2} =
            \sum_{n_1=0}^{N_1-1}
              \left[ e^{-\frac{2\pi i}{N_1N_2} n_1 k_2 } \right]
              \left( \sum_{n_2=0}^{N_2-1} x_{N_1 n_2 + n_1}
                      e^{-\frac{2\pi i}{N_2} n_2 k_2 } \right)
              e^{-\frac{2\pi i}{N_1} n_1 k_1 }

    where each inner sum is a DFT of size :math:`N_2`, each outer sum is a DFT of
    size :math:`N_1`. That is:

    1. Perform :math:`N_1` DFTs of size :math:`N_2`, emitted as OrbitalRotation
       gates.
    2. Multiply by the twiddle factors
       :math:`e^{-\frac{2\pi i}{N_1N_2} n_1 k_2 }`.
    3. Perform :math:`N_2` DFTs of size :math:`N_1`, computed recursively.

    Since the DFTs act on strided sets of modes, fermionic mode permutations are
    applied so that each DFT acts on contiguous qubits.
    """
    N = len(qubits)  # noqa: N806
    N2 = factors[0]  # noqa: N806

    if len(factors) == 1:
        # Base case: DFT of prime size
        yield CircuitInstruction(
            OrbitalRotationSpinlessJW(N, scipy.linalg.dft(N, scale="sqrtn")),
            tuple(qubits),
        )
        return

    N1 = N // N2  # noqa: N806

    # Move mode N1 * n2 + n1 to position n1 * N2 + n2, so that for each n1 the
    # modes n2 = 0, ..., N2 - 1 are contiguous. This transposes the input viewed
    # as an N1 x N2 matrix in column-major order
    perm = np.arange(N).reshape(N2, N1).T.ravel()
    yield from _permute_modes_jw(qubits, perm)

    # Step 1: N1 DFTs of size N2, yielding index n1 * N2 + k2
    dft_mat = scipy.linalg.dft(N2, scale="sqrtn")
    for n1 in range(N1):
        yield CircuitInstruction(
            OrbitalRotationSpinlessJW(N2, dft_mat),
            tuple(qubits[n1 * N2 : (n1 + 1) * N2]),
        )

    # Step 2: Twiddle factors exp(-2 pi i n1 k2 / N)
    for n1 in range(1, N1):
        for k2 in range(1, N2):
            yield CircuitInstruction(
                PhaseGate(-2 * math.pi * n1 * k2 / N), (qubits[n1 * N2 + k2],)
            )

    # Move index n1 * N2 + k2 to position k2 * N1 + n1, so that for each k2 the
    # indices n1 = 0, ..., N1 - 1 are contiguous
    transpose = np.arange(N).reshape(N1, N2).T.ravel()
    yield from _permute_modes_jw(qubits, transpose)

    # Step 3: N2 DFTs of size N1, yielding index k2 * N1 + k1
    for k2 in range(N2):
        yield from _cooley_tukey_ffft_jw(qubits[k2 * N1 : (k2 + 1) * N1], factors[1:])

    # Move index k2 * N1 + k1 to output position N2 * k1 + k2
    output_perm = np.arange(N).reshape(N2, N1).T.ravel()
    yield from _permute_modes_jw(qubits, output_perm)


def _permute_modes_jw(
    qubits: Sequence[Qubit],
    perm: np.ndarray,
) -> Iterator[CircuitInstruction]:
    """Move fermionic mode perm[i] to position i."""
    n = len(qubits)
    if np.array_equal(perm, np.arange(n)):
        return
    mat = np.zeros((n, n))
    mat[range(n), perm] = 1
    yield CircuitInstruction(OrbitalRotationSpinlessJW(n, mat), tuple(qubits))
