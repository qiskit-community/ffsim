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
import scipy.sparse.linalg

import ffsim


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
    dim = ffsim.dim(norb, nelec)
    # TODO test with complex one-body tensor fails due to the following issue
    # https://github.com/qiskit-community/ffsim/issues/14
    one_body_tensor = np.real(ffsim.random.random_hermitian(norb, seed=2474))
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=7054, dtype=float)
    mol_hamiltonian = ffsim.MolecularHamiltonian(one_body_tensor, two_body_tensor)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)

    # perform double factorization
    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian, z_representation=z_representation
    )

    # generate initial state
    dim = ffsim.dim(norb, nelec)
    initial_state = ffsim.random.random_statevector(dim, seed=1360)
    original_state = initial_state.copy()

    # compute exact state
    exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * time * hamiltonian,
        initial_state,
        traceA=ffsim.trace(mol_hamiltonian, norb=norb, nelec=nelec),
    )

    # make sure time is not too small
    assert abs(np.vdot(exact_state, initial_state)) < 0.98

    # simulate
    final_state = ffsim.simulate_trotter_double_factorized(
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
    np.testing.assert_allclose(np.linalg.norm(final_state), 1.0)
    fidelity = np.abs(np.vdot(final_state, exact_state))
    assert fidelity >= target_fidelity
