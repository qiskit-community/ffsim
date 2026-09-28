# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Givens rotation ansatz gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim

RNG = np.random.default_rng(47209049524507342724926741129799062179)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_random_spinful(norb: int, nelec: tuple[int, int]):
    """Test random Givens rotation ansatz gives correct output state."""
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        givens_ansatz_op = ffsim.random.random_givens_ansatz_op(norb, seed=RNG)
        gate = ffsim.qiskit.GivensAnsatzOpJW(givens_ansatz_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=RNG)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(
            small_vec, givens_ansatz_op, norb=norb, nelec=nelec
        )

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_random_spinless(norb: int, nelec: int):
    """Test random spinless Givens rotation ansatz gives correct output state."""
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        givens_ansatz_op = ffsim.random.random_givens_ansatz_op(norb, seed=RNG)
        gate = ffsim.qiskit.GivensAnsatzOpSpinlessJW(givens_ansatz_op)
        assert gate.num_qubits == norb

        small_vec = ffsim.random.random_state_vector(dim, seed=RNG)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(
            small_vec, givens_ansatz_op, norb=norb, nelec=nelec
        )

        np.testing.assert_allclose(result, expected)


def test_equality_spinful():
    """Test equality comparison."""
    norb = 5
    givens_ansatz_op = ffsim.random.random_givens_ansatz_op(norb, seed=RNG)
    other_givens_ansatz_op = ffsim.random.random_givens_ansatz_op(norb, seed=RNG)

    gate = ffsim.qiskit.GivensAnsatzOpJW(givens_ansatz_op)
    assert gate == ffsim.qiskit.GivensAnsatzOpJW(givens_ansatz_op)
    assert gate != ffsim.qiskit.GivensAnsatzOpJW(other_givens_ansatz_op)
    assert gate != ffsim.qiskit.GivensAnsatzOpSpinlessJW(givens_ansatz_op)
    assert gate != "gate"


def test_equality_spinless():
    """Test equality comparison, spinless."""
    norb = 5
    givens_ansatz_op = ffsim.random.random_givens_ansatz_op(norb, seed=RNG)
    other_givens_ansatz_op = ffsim.random.random_givens_ansatz_op(norb, seed=RNG)

    gate = ffsim.qiskit.GivensAnsatzOpSpinlessJW(givens_ansatz_op)
    assert gate == ffsim.qiskit.GivensAnsatzOpSpinlessJW(givens_ansatz_op)
    assert gate != ffsim.qiskit.GivensAnsatzOpSpinlessJW(other_givens_ansatz_op)
    assert gate != "gate"
