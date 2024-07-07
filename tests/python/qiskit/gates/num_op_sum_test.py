# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for number operator sum evolution gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_random_spinful(norb: int, nelec: tuple[int, int]):
    """Test random gate gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        coeffs_a = rng.standard_normal(norb)
        coeffs_b = rng.standard_normal(norb)
        time = rng.uniform(-10, 10)

        # coeffs
        gate = ffsim.qiskit.NumOpSumEvolutionJW(norb, coeffs_a, time)
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_num_op_sum_evolution(
            small_vec, coeffs_a, time, norb=norb, nelec=nelec
        )
        np.testing.assert_allclose(result, expected)

        # (coeffs_a, coeffs_b)
        gate = ffsim.qiskit.NumOpSumEvolutionJW(norb, (coeffs_a, coeffs_b), time)
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_num_op_sum_evolution(
            small_vec, (coeffs_a, coeffs_b), time, norb=norb, nelec=nelec
        )
        np.testing.assert_allclose(result, expected)

        # (coeffs_a, None)
        gate = ffsim.qiskit.NumOpSumEvolutionJW(norb, (coeffs_a, None), time)
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_num_op_sum_evolution(
            small_vec, (coeffs_a, None), time, norb=norb, nelec=nelec
        )
        np.testing.assert_allclose(result, expected)

        # Numpy array input
        gate = ffsim.qiskit.NumOpSumEvolutionJW(
            norb, np.stack((coeffs_a, coeffs_b)), time
        )
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_num_op_sum_evolution(
            small_vec, (coeffs_a, coeffs_b), time, norb=norb, nelec=nelec
        )
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(5)))
def test_random_spinless(norb: int, nelec: int):
    """Test random spinless gate gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        coeffs = rng.standard_normal(norb)
        time = rng.uniform(-10, 10)

        gate = ffsim.qiskit.NumOpSumEvolutionSpinlessJW(norb, coeffs, time)
        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )
        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.apply_num_op_sum_evolution(
            small_vec, coeffs, time, norb=norb, nelec=nelec
        )
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_inverse_spinful(norb: int, nelec: tuple[int, int]):
    """Test inverse."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        coeffs_a = rng.standard_normal(norb)
        coeffs_b = rng.standard_normal(norb)
        time = rng.uniform(-10, 10)

        # coeffs
        gate = ffsim.qiskit.NumOpSumEvolutionJW(norb, coeffs_a, time)
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=rng), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec)

        # (coeffs_a, coeffs_b)
        gate = ffsim.qiskit.NumOpSumEvolutionJW(norb, (coeffs_a, coeffs_b), time)
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=rng), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec)

        # (coeffs_a, None)
        gate = ffsim.qiskit.NumOpSumEvolutionJW(norb, (coeffs_a, None), time)
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=rng), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec)

        # (None, coeffs_b)
        gate = ffsim.qiskit.NumOpSumEvolutionJW(norb, (None, coeffs_b), time)
        vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            ffsim.random.random_state_vector(dim, seed=rng), norb=norb, nelec=nelec
        )
        statevec = Statevector(vec).evolve(gate)
        statevec = statevec.evolve(gate.inverse())
        np.testing.assert_allclose(np.array(statevec), vec)
