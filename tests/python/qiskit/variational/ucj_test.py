# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for UCJ ansatzes in Qiskit."""

import numpy as np

import ffsim

RNG = np.random.default_rng(473284857536346)


def test_ucj_spin_balanced_parameters_from_t2():
    norb = 6
    nocc = 3
    nelec = (nocc, nocc)

    n_reps = 1
    pairs_aa = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 0)]
    pairs_ab = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    # interaction_pairs = interaction_pairs_spin_balanced("square", norb)
    interaction_pairs = (pairs_aa, pairs_ab)

    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG)
    t1 = ffsim.random.random_hermitian(norb, seed=RNG)

    circuit = ffsim.qiskit.ucj_spin_balanced_ansatz(
        norb, nelec, n_reps=n_reps, interaction_pairs=interaction_pairs
    )

    params = ffsim.qiskit.ucj_spin_balanced_parameters_from_t_amplitudes(
        t2, t1=t1, interaction_pairs=interaction_pairs
    )

    assigned = circuit.assign_parameters(params)

    final_state = ffsim.qiskit.final_state_vector(assigned, norb=norb, nelec=nelec)

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2, t1=t1, n_reps=n_reps, interaction_pairs=interaction_pairs
    )
    expected = ffsim.apply_unitary(
        ffsim.hartree_fock_state(norb, nelec), ucj_op, norb=norb, nelec=nelec
    )

    np.testing.assert_allclose(final_state, expected)
