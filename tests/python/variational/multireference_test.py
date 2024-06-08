# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for multireference states."""

import numpy as np

import ffsim
import ffsim.random.random


def test_multireference_state_prod():
    """Test multireference state for product operator."""
    rng = np.random.default_rng(30314)

    norb = 8
    nelec = (4, 4)

    def brickwork(norb: int, n_layers: int):
        for i in range(n_layers):
            for j in range(i % 2, norb - 1, 2):
                yield (j, j + 1)

    n_layers = norb
    interaction_pairs = list(brickwork(norb, n_layers))

    for _ in range(5):
        thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
        operator = ffsim.HopGateAnsatzOperator(norb, interaction_pairs, thetas)

        mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(
            norb, seed=rng, dtype=float
        )
        reference_occupations = [
            ((0, 1, 2, 3), (1, 2, 3, 4)),
            ((0, 1, 2, 4), (2, 3, 4, 6)),
            ((1, 2, 4, 5), (2, 3, 4, 7)),
        ]

        energy, prod_state_sum = ffsim.multireference_state_prod(
            mol_hamiltonian,
            (operator, operator),
            reference_occupations,
            norb=norb,
            nelec=nelec,
        )
        reconstructed_state = np.tensordot(
            prod_state_sum.coeffs,
            [np.kron(vec_a, vec_b) for vec_a, vec_b in prod_state_sum.states],
            axes=1,
        )
        expected_energy, state = ffsim.multireference_state(
            mol_hamiltonian,
            operator,
            reference_occupations,
            norb=norb,
            nelec=nelec,
        )

        np.testing.assert_allclose(energy, expected_energy)
        ffsim.testing.assert_allclose_up_to_global_phase(reconstructed_state, state)
