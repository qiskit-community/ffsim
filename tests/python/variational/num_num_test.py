# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for number-number interaction ansatz."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import ffsim


def test_parameters_roundtrip():
    """Test converting to and back from parameters gives consistent results."""
    norb = 5
    rng = np.random.default_rng()

    pairs_aa = list(itertools.combinations(range(norb), 2))[:norb]
    pairs_ab = list(itertools.combinations(range(norb), 2))[-norb:]
    thetas_aa = rng.uniform(-np.pi, np.pi, size=len(pairs_aa))
    thetas_ab = rng.uniform(-np.pi, np.pi, size=len(pairs_ab))

    operator = ffsim.NumNumAnsatzOpSpinBalanced(
        norb=norb,
        interaction_pairs=(pairs_aa, pairs_ab),
        thetas=(thetas_aa, thetas_ab),
    )
    assert (
        operator.n_params(interaction_pairs=(pairs_aa, pairs_ab))
        == len(operator.to_parameters())
        == 2 * norb
    )
    roundtripped = ffsim.NumNumAnsatzOpSpinBalanced.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=(pairs_aa, pairs_ab),
    )
    assert ffsim.approx_eq(roundtripped, operator)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_diag_coulomb_mats(norb: int, nelec: tuple[int, int]):
    """Test initialization from diagonal Coulomb matrices."""
    rng = np.random.default_rng()
    mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    mat_ab = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    operator = ffsim.NumNumAnsatzOpSpinBalanced.from_diag_coulomb_mats((mat_aa, mat_ab))
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_diag_coulomb_evolution(
        vec, (mat_aa, mat_ab, mat_aa), time=-1.0, norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(actual, expected)


def test_incorrect_num_params():
    """Test that passing incorrect number of parameters throws an error."""
    norb = 5
    pairs_aa = [(0, 1), (2, 3)]
    pairs_ab = [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match="number"):
        _ = ffsim.NumNumAnsatzOpSpinBalanced(
            norb=norb,
            interaction_pairs=(pairs_aa, pairs_ab),
            thetas=(np.zeros(len(pairs_aa) + 1), np.zeros(len(pairs_ab))),
        )
    with pytest.raises(ValueError, match="number"):
        _ = ffsim.NumNumAnsatzOpSpinBalanced(
            norb=norb,
            interaction_pairs=(pairs_aa, pairs_ab),
            thetas=(np.zeros(len(pairs_aa)), np.zeros(len(pairs_ab) - 1)),
        )


def test_invalid_pairs():
    """Test that passing invalid pairs throws an error."""
    norb = 5
    good_pairs = [(0, 1), (2, 3)]
    bad_pairs = [(1, 0), (2, 3)]
    with pytest.raises(ValueError, match="triangular"):
        _ = ffsim.NumNumAnsatzOpSpinBalanced(
            norb=norb,
            interaction_pairs=(bad_pairs, good_pairs),
            thetas=(np.zeros(len(bad_pairs)), np.zeros(len(good_pairs))),
        )
    with pytest.raises(ValueError, match="triangular"):
        _ = ffsim.NumNumAnsatzOpSpinBalanced(
            norb=norb,
            interaction_pairs=(good_pairs, bad_pairs),
            thetas=(np.zeros(len(good_pairs)), np.zeros(len(bad_pairs))),
        )
