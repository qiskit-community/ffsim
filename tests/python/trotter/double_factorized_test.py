# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for double-factorized Trotter simulation."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg

import ffsim


@pytest.mark.parametrize(
    "norb, nelec, time, n_steps, order, z_representation, target_fidelity",
    [
        (3, (1, 1), 0.1, 10, 0, False, 0.999),
        (3, (1, 1), 0.1, 2, 1, True, 0.999),
        (4, (2, 1), 0.1, 1, 2, False, 0.999),
        (4, (1, 2), 0.1, 3, 2, True, 0.999),
        (4, (2, 2), 0.1, 4, 1, False, 0.999),
        (5, (3, 2), 0.1, 5, 1, True, 0.999),
    ],
)
def test_random(
    norb: int,
    nelec: tuple[int, int],
    time: float,
    n_steps: int,
    order: int,
    z_representation: bool,
    target_fidelity: float,
):
    """Test random Hamiltonian."""
    rng = np.random.default_rng(2488)

    # generate random Hamiltonian
    dim = ffsim.dim(norb, nelec)
    hamiltonian = ffsim.random.random_double_factorized_hamiltonian(
        norb, rank=norb, z_representation=z_representation, seed=rng
    )
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)

    # generate initial state
    dim = ffsim.dim(norb, nelec)
    initial_state = ffsim.random.random_state_vector(dim, seed=rng)
    original_state = initial_state.copy()

    # compute exact state
    exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * time * linop,
        initial_state,
        traceA=1.0,
    )

    # make sure time is not too small
    assert abs(np.vdot(exact_state, initial_state)) < 0.95

    # simulate
    final_state = ffsim.simulate_trotter_double_factorized(
        initial_state,
        hamiltonian,
        time,
        norb=norb,
        nelec=nelec,
        n_steps=n_steps,
        order=order,
    )

    # check that initial state was not modified
    np.testing.assert_allclose(initial_state, original_state)

    # check agreement
    np.testing.assert_allclose(np.linalg.norm(final_state), 1.0)
    fidelity = np.abs(np.vdot(final_state, exact_state))
    assert fidelity >= target_fidelity
