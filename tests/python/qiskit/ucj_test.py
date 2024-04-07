# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for unitary cluster Jastrow circuit."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_random_ucj_operator(norb: int, nelec: tuple[int, int]):
    """Test random orbital rotation circuit gives correct output state."""
    rng = np.random.default_rng()
    n_reps = 3
    dim = ffsim.dim(norb, nelec)
    for _ in range(3):
        ucj_op = ffsim.UCJOperator(
            diag_coulomb_mats_alpha_alpha=np.stack(
                [
                    ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
                    for _ in range(n_reps)
                ]
            ),
            diag_coulomb_mats_alpha_beta=np.stack(
                [
                    ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
                    for _ in range(n_reps)
                ]
            ),
            orbital_rotations=np.stack(
                [ffsim.random.random_unitary(norb, seed=rng) for _ in range(n_reps)]
            ),
            final_orbital_rotation=ffsim.random.random_unitary(norb, seed=rng),
        )
        gate = ffsim.qiskit.UCJOperatorJW(ucj_op)

        small_vec = ffsim.random.random_statevector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.apply_unitary(small_vec, ucj_op, norb=norb, nelec=nelec)

        np.testing.assert_allclose(result, expected)
