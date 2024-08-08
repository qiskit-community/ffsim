# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for diagonal Coulomb Trotter evolution gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize(
    "norb, nelec, time, n_steps, order",
    [
        (4, (2, 2), 0.1, 1, 0),
        (4, (2, 2), 0.1, 2, 0),
        (4, (2, 2), 0.1, 1, 1),
        (4, (2, 2), 0.1, 1, 2),
    ],
)
def test_random(
    norb: int,
    nelec: tuple[int, int],
    time: float,
    n_steps: int,
    order: int,
):
    """Test random gate gives correct output state."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)
    time = 1.0
    for _ in range(3):
        hamiltonian = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=rng)
        gate = ffsim.qiskit.SimulateTrotterDiagCoulombSplitOpJW(
            hamiltonian, time, n_steps=n_steps, order=order
        )

        small_vec = ffsim.random.random_state_vector(dim, seed=rng)
        big_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(
            small_vec, norb=norb, nelec=nelec
        )

        statevec = Statevector(big_vec).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.simulate_trotter_diag_coulomb_split_op(
            small_vec,
            hamiltonian,
            time=time,
            norb=norb,
            nelec=nelec,
            n_steps=n_steps,
            order=order,
        )

        np.testing.assert_allclose(result, expected)
