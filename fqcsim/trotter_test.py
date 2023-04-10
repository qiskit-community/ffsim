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

from fqcsim.double_factorized import double_factorized_decomposition
from fqcsim.fci import get_dimension, get_hamiltonian_linop, get_trace
from fqcsim.linalg import expm_multiply_taylor
from fqcsim.random_utils import (
    random_hermitian,
    random_statevector,
    random_two_body_tensor_real,
)
from fqcsim.trotter import simulate_trotter_suzuki_double_factorized


@pytest.mark.parametrize(
    "norb, nelec, time, n_steps, order, target_fidelity",
    [
        (3, (1, 1), 0.1, 10, 0, 0.99),
        (3, (1, 1), 0.1, 3, 1, 0.99),
        (4, (2, 1), 0.1, 3, 2, 0.99),
        (4, (1, 2), 0.1, 3, 2, 0.99),
        (4, (2, 2), 0.1, 8, 1, 0.99),
    ],
)
def test_simulate_trotter_suzuki_double_factorized_random(
    norb: int,
    nelec: tuple[int, int],
    time: float,
    n_steps: int,
    order: int,
    target_fidelity: float,
):
    # generate random Hamiltonian
    dim = get_dimension(norb, nelec)
    one_body_tensor = np.real(np.array(random_hermitian(norb, seed=2474)))
    two_body_tensor = random_two_body_tensor_real(norb, seed=7054)
    hamiltonian = get_hamiltonian_linop(one_body_tensor, two_body_tensor, nelec)
    trace = get_trace(one_body_tensor, two_body_tensor, nelec)

    # perform double factorization
    df_hamiltonian = double_factorized_decomposition(one_body_tensor, two_body_tensor)

    # generate initial state
    dim = get_dimension(norb, nelec)
    initial_state = np.array(random_statevector(dim, seed=1360))
    original_state = initial_state.copy()

    # compute exact state
    exact_state = expm_multiply_taylor(-1j * time * hamiltonian, initial_state)

    # make sure time is not too small
    assert abs(np.vdot(exact_state, initial_state)) < 0.98

    # simulate
    final_state = simulate_trotter_suzuki_double_factorized(
        df_hamiltonian.one_body_tensor,
        df_hamiltonian.core_tensors,
        df_hamiltonian.leaf_tensors,
        time,
        initial_state,
        nelec,
        n_steps=n_steps,
        order=order,
    )

    # check that initial state was not modified
    np.testing.assert_allclose(initial_state, original_state)

    # check agreement
    np.testing.assert_allclose(np.linalg.norm(final_state), 1.0, atol=1e-8)
    fidelity = np.abs(np.vdot(final_state, exact_state))
    assert fidelity >= target_fidelity
