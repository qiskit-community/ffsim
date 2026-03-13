# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for diagonal Coulomb Trotter simulation."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg

import ffsim


@pytest.mark.parametrize(
    "norb, nelec, time, n_steps, order, atol",
    [
        (3, (1, 1), 0.1, 20, 0, 1e-2),
        (4, (2, 1), 0.1, 10, 2, 1e-3),
        (4, (2, 2), 0.1, 10, 1, 1e-3),
    ],
)
def test_random(
    norb: int,
    nelec: tuple[int, int],
    time: float,
    n_steps: int,
    order: int,
    atol: float,
):
    rng = np.random.default_rng(2488)

    # generate random Hamiltonian
    dim = ffsim.dim(norb, nelec)
    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    diag_coulomb_mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb_mat_ab = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb_mats = np.stack([diag_coulomb_mat_aa, diag_coulomb_mat_ab])
    constant = rng.uniform(-10, 10)
    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mats, constant=constant
    )
    linop = ffsim.linear_operator(dc_hamiltonian, norb=norb, nelec=nelec)

    # generate initial state
    dim = ffsim.dim(norb, nelec)
    initial_state = ffsim.random.random_state_vector(dim, seed=rng)
    original_state = initial_state.copy()

    # compute exact state
    exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * time * linop,
        initial_state,
        traceA=-1j * time * np.sum(np.abs(diag_coulomb_mats)),
    )

    # make sure time is not too small
    assert abs(np.vdot(exact_state, initial_state)) < 0.98

    # simulate
    final_state = ffsim.simulate_trotter_diag_coulomb_split_op(
        initial_state,
        dc_hamiltonian,
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
    np.testing.assert_allclose(final_state, exact_state, atol=atol)


def test_hubbard():
    rng = np.random.default_rng(2488)

    hubbard_model = ffsim.fermi_hubbard_2d(
        norb_x=3,
        norb_y=3,
        tunneling=1.0,
        interaction=4.0,
        chemical_potential=0.5,
        nearest_neighbor_interaction=2.0,
        periodic=False,
    )
    dc_hamiltonian = ffsim.DiagonalCoulombHamiltonian.from_fermion_operator(
        hubbard_model
    )
    norb = dc_hamiltonian.norb
    nelec = (norb // 2, norb // 2)
    time = 0.1
    n_steps = 1
    order = 1
    atol = 1e-3

    dim = ffsim.dim(norb, nelec)
    linop = ffsim.linear_operator(dc_hamiltonian, norb=norb, nelec=nelec)

    # generate initial state
    dim = ffsim.dim(norb, nelec)
    initial_state = ffsim.random.random_state_vector(dim, seed=rng)
    original_state = initial_state.copy()

    # compute exact state
    exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * time * linop,
        initial_state,
        traceA=-1j * time * np.sum(np.abs(dc_hamiltonian.diag_coulomb_mats)),
    )

    # make sure time is not too small
    assert abs(np.vdot(exact_state, initial_state)) < 0.98

    # simulate
    final_state = ffsim.simulate_trotter_diag_coulomb_split_op(
        initial_state,
        dc_hamiltonian,
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
    np.testing.assert_allclose(final_state, exact_state, atol=atol)
