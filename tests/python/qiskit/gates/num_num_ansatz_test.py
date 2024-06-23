# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for number-number interaction ansatz gate."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_random_num_num_ansatz(norb: int, nelec: tuple[int, int]):
    """Test random number-number interaction ansatz gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        pairs_aa = list(itertools.combinations_with_replacement(range(norb), 2))
        pairs_ab = list(itertools.combinations_with_replacement(range(norb), 2))
        thetas_aa = rng.uniform(-np.pi, np.pi, size=len(pairs_aa))
        thetas_ab = rng.uniform(-np.pi, np.pi, size=len(pairs_ab))
        num_num_ansatz_op = ffsim.NumNumAnsatzOpSpinBalanced(
            norb, interaction_pairs=(pairs_aa, pairs_ab), thetas=(thetas_aa, thetas_ab)
        )
        gate = ffsim.qiskit.NumNumAnsatzOpSpinBalancedJW(num_num_ansatz_op)

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(
            small_vec, num_num_ansatz_op, norb=norb, nelec=nelec
        )

        np.testing.assert_allclose(result, expected)
