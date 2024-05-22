# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for hop gate ansatz."""

import itertools
import math

import numpy as np

import ffsim


def test_parameters_roundtrip():
    """Test converting to and back from parameters gives consistent results."""
    norb = 5
    n_reps = 2
    rng = np.random.default_rng()

    def ncycles(iterable, n):
        "Returns the sequence elements n times"
        return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))

    pairs = itertools.combinations(range(norb), 2)
    interaction_pairs = list(ncycles(pairs, n_reps))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    final_orbital_rotation = ffsim.random.random_unitary(norb)

    operator = ffsim.HopGateAnsatzOperator(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=thetas,
        final_orbital_rotation=final_orbital_rotation,
    )
    assert len(operator.to_parameters()) == n_reps * math.comb(norb, 2) + norb**2
    roundtripped = ffsim.HopGateAnsatzOperator.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=True,
    )

    np.testing.assert_allclose(roundtripped.thetas, operator.thetas)
    np.testing.assert_allclose(
        roundtripped.final_orbital_rotation, operator.final_orbital_rotation
    )
