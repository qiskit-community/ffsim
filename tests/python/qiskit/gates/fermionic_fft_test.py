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


@pytest.mark.parametrize(
    "norb, nelec", tuple(ffsim.testing.generate_norb_nelec(exhaustive=False))
)
def test_fermionic_fft_spinful(norb: int, nelec: tuple[int, int]):
    """Test spinful fermionic FFT circuit gives correct output state."""
    dim = ffsim.dim(norb, nelec)
    mat = scipy.linalg.dft(norb, scale="sqrtn")
    gate = FermionicFFTJW(norb)

    for _ in range(3):
        small_vec = ffsim.random.random_state_vector(dim, seed=RNG)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_orbital_rotation(small_vec, mat, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nocc", tuple(ffsim.testing.generate_norb_nocc(exhaustive=False))
)
def test_fermionic_fft_spinless(norb: int, nocc: int):
    """Test spinless fermionic FFT circuit gives correct output state."""
    nelec = (nocc, 0)
    dim = ffsim.dim(norb, nelec)
    mat = scipy.linalg.dft(norb, scale="sqrtn")
    gate = FermionicFFTSpinlessJW(norb)

    for _ in range(3):
        small_vec = ffsim.random.random_state_vector(dim, seed=RNG)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_orbital_rotation(small_vec, mat, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec", tuple(ffsim.testing.generate_norb_nelec(exhaustive=False))
)
def test_inverse_spinful(norb: int, nelec: tuple[int, int]):
    """Test inverse for spinful fermionic FFT."""
    dim = ffsim.dim(norb, nelec)
    gate = FermionicFFTJW(norb)

    for _ in range(3):
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=RNG), norb=norb, nelec=nelec
        )

        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())

        np.testing.assert_allclose(np.array(statevec), vec)


@pytest.mark.parametrize(
    "norb, nocc", tuple(ffsim.testing.generate_norb_nocc(exhaustive=False))
)
def test_inverse_spinless(norb: int, nocc: int):
    """Test inverse for spinless fermionic FFT."""
    nelec = (nocc, 0)
    dim = ffsim.dim(norb, nelec)
    gate = FermionicFFTSpinlessJW(norb)

    for _ in range(3):
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=RNG), norb=norb, nelec=nelec
        )

        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())

        np.testing.assert_allclose(np.array(statevec), vec)
