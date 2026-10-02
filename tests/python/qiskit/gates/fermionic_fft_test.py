# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for fermionic fast Fourier transform gates."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from qiskit.quantum_info import Statevector

import ffsim
from ffsim.qiskit.gates.fermionic_fft import (
    FermionicFFTJW,
    FermionicFFTSpinlessJW,
)

RNG = np.random.default_rng(167752397076783491645090817541429291935)


def assert_implements_dft(
    gate: FermionicFFTJW | FermionicFFTSpinlessJW,
    norb: int,
    nelec: tuple[int, int],
    n_trials: int = 3,
):
    """Assert that the gate applies the DFT orbital rotation to random states."""
    dim = ffsim.dim(norb, nelec)
    mat = scipy.linalg.dft(norb, scale="sqrtn")
    for _ in range(n_trials):
        small_vec = ffsim.random.random_state_vector(dim, seed=RNG)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_orbital_rotation(small_vec, mat, norb=norb, nelec=nelec)
        np.testing.assert_allclose(result, expected, atol=1e-12)


def assert_inverse(
    gate: FermionicFFTJW | FermionicFFTSpinlessJW,
    norb: int,
    nelec: tuple[int, int],
    n_trials: int = 3,
):
    """Assert that the gate followed by its inverse is the identity."""
    dim = ffsim.dim(norb, nelec)
    for _ in range(n_trials):
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=RNG), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate).evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec", tuple(ffsim.testing.generate_norb_nelec(exhaustive=False))
)
def test_fermionic_fft_spinful(norb: int, nelec: tuple[int, int]):
    """Test spinful fermionic FFT circuit gives correct output state."""
    assert_implements_dft(FermionicFFTJW(norb), norb, nelec)


@pytest.mark.parametrize(
    "norb, nocc", tuple(ffsim.testing.generate_norb_nocc(exhaustive=False))
)
def test_fermionic_fft_spinless(norb: int, nocc: int):
    """Test spinless fermionic FFT circuit gives correct output state."""
    assert_implements_dft(FermionicFFTSpinlessJW(norb), norb, (nocc, 0))


@pytest.mark.parametrize(
    "norb, nelec", tuple(ffsim.testing.generate_norb_nelec(exhaustive=False))
)
def test_inverse_spinful(norb: int, nelec: tuple[int, int]):
    """Test inverse for spinful fermionic FFT."""
    assert_inverse(FermionicFFTJW(norb), norb, nelec)


@pytest.mark.parametrize(
    "norb, nocc", tuple(ffsim.testing.generate_norb_nocc(exhaustive=False))
)
def test_inverse_spinless(norb: int, nocc: int):
    """Test inverse for spinless fermionic FFT."""
    assert_inverse(FermionicFFTSpinlessJW(norb), norb, (nocc, 0))


# Each size exercises a distinct path of the mixed-radix recursion.
# N = N1 * N2, where N2 is the smallest prime factor.
#   6 = 2 * 3:     mixed prime factors, so the size-N2 and size-N1 DFTs differ
#   8 = 2 * 2 * 2: recursion three levels deep
#   9 = 3 * 3:     repeated odd prime, with non-trivial twiddles for N2 > 2
@pytest.mark.parametrize("norb", [6, 8, 9])
def test_fermionic_fft_spinless_composite(norb: int):
    """Test spinless fermionic FFT for composite sizes."""
    gate = FermionicFFTSpinlessJW(norb)
    for nocc in [1, norb // 2]:
        assert_implements_dft(gate, norb, (nocc, 0), n_trials=1)


@pytest.mark.parametrize("norb", [6, 8, 9])
def test_fermionic_fft_spinful_composite(norb: int):
    """Test spinful fermionic FFT for composite sizes."""
    gate = FermionicFFTJW(norb)
    for nelec in [(1, 2), (norb // 2, 1)]:
        assert_implements_dft(gate, norb, nelec, n_trials=1)


def test_equality():
    """Test equality comparison."""
    norb = 5
    gate = ffsim.qiskit.FermionicFFTJW(norb)
    assert gate == ffsim.qiskit.FermionicFFTJW(norb)
    assert gate != ffsim.qiskit.FermionicFFTJW(norb + 1)
    assert gate != ffsim.qiskit.FermionicFFTSpinlessJW(2 * norb)
    assert gate != "gate"


def test_equality_spinless():
    """Test equality comparison, spinless."""
    norb = 5
    gate = ffsim.qiskit.FermionicFFTSpinlessJW(norb)
    assert gate == ffsim.qiskit.FermionicFFTSpinlessJW(norb)
    assert gate != ffsim.qiskit.FermionicFFTSpinlessJW(norb + 1)
    assert gate != "gate"
