# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for unitary cluster Jastrow gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_random_ucj_op_spin_unbalanced(norb: int, nelec: tuple[int, int]):
    """Test random spin-unbalanced UCJ gate gives correct output state."""
    rng = np.random.default_rng()
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )
        gate = ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(small_vec, ucj_op, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_random_ucj_op_spin_balanced(norb: int, nelec: tuple[int, int]):
    """Test random spin-balanced UCJ gate gives correct output state."""
    rng = np.random.default_rng()
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.random.random_ucj_op_spin_balanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )
        gate = ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(small_vec, ucj_op, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(5)))
def test_random_ucj_op_spinless(norb: int, nelec: int):
    """Test random spin-balanced UCJ gate gives correct output state."""
    rng = np.random.default_rng()
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.random.random_ucj_op_spinless(
            norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=rng
        )
        gate = ffsim.qiskit.UCJOpSpinlessJW(ucj_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(small_vec, ucj_op, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)
