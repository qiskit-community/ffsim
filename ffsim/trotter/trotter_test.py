# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Trotter simulation."""

from __future__ import annotations

import numpy as np
import pytest

from ffsim.double_factorized import double_factorized_decomposition
from ffsim.fci import get_dimension, get_hamiltonian_linop
from ffsim.linalg import expm_multiply_taylor
from ffsim.random import (
    random_hermitian,
    random_statevector,
    random_two_body_tensor_real,
)
from ffsim.trotter import simulate_trotter_double_factorized


@pytest.mark.parametrize(
    "norb, nelec, time, n_steps, order, z_representation, target_fidelity",
    [
        (3, (1, 1), 0.1, 10, 0, False, 0.99),
        (3, (1, 1), 0.1, 3, 1, True, 0.99),
        (4, (2, 1), 0.1, 3, 2, False, 0.99),
        (4, (1, 2), 0.1, 3, 2, True, 0.99),
        (4, (2, 2), 0.1, 8, 1, False, 0.99),
    ],
)
def test_simulate_trotter_double_factorized_random(
    norb: int,
    nelec: tuple[int, int],
    time: float,
    n_steps: int,
    order: int,
    z_representation: bool,
    target_fidelity: float,
):
    # generate random Hamiltonian
    dim = get_dimension(norb, nelec)
    # TODO test with complex one-body tensor after fixing get_hamiltonian_linop
    one_body_tensor = np.real(np.array(random_hermitian(norb, seed=2474)))
    two_body_tensor = random_two_body_tensor_real(norb, seed=7054)
    hamiltonian = get_hamiltonian_linop(
        one_body_tensor, two_body_tensor, norb=norb, nelec=nelec
    )

    # perform double factorization
    df_hamiltonian = double_factorized_decomposition(
        one_body_tensor, two_body_tensor, z_representation=z_representation
    )

    # generate initial state
    dim = get_dimension(norb, nelec)
    initial_state = random_statevector(dim, seed=1360)
    original_state = initial_state.copy()

    # compute exact state
    exact_state = expm_multiply_taylor(initial_state, -1j * time * hamiltonian)

    # make sure time is not too small
    assert abs(np.vdot(exact_state, initial_state)) < 0.98

    # simulate
    final_state = simulate_trotter_double_factorized(
        initial_state,
        df_hamiltonian,
        time,
        norb=norb,
        nelec=nelec,
        n_steps=n_steps,
        order=order,
    )

    # check that initial state was not modified
    np.testing.assert_allclose(initial_state, original_state)

    # check agreement
    np.testing.assert_allclose(np.linalg.norm(final_state), 1.0, atol=1e-8)
    fidelity = np.abs(np.vdot(final_state, exact_state))
    assert fidelity >= target_fidelity
