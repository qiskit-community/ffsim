# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Givens rotation ansatz."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

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

    operator = ffsim.GivensAnsatzOperator(
        norb=norb, interaction_pairs=interaction_pairs, thetas=thetas
    )
    assert len(operator.to_parameters()) == n_reps * math.comb(norb, 2)
    roundtripped = ffsim.GivensAnsatzOperator.from_parameters(
        operator.to_parameters(), norb=norb, interaction_pairs=interaction_pairs
    )

    np.testing.assert_allclose(roundtripped.thetas, operator.thetas)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_orbital_rotation(norb: int, nelec: tuple[int, int]):
    n_reps = 2
    rng = np.random.default_rng()

    def ncycles(iterable, n):
        "Returns the sequence elements n times"
        return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))

    pairs = itertools.combinations(range(norb), 2)
    interaction_pairs = list(ncycles(pairs, n_reps))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))

    operator = ffsim.GivensAnsatzOperator(
        norb=norb, interaction_pairs=interaction_pairs, thetas=thetas
    )

    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)
    actual = ffsim.apply_orbital_rotation(
        vec, operator.orbital_rotation, norb=norb, nelec=nelec
    )
    expected = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)

    np.testing.assert_allclose(actual, expected)
