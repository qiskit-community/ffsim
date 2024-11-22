# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the TeNPy orbital rotation gate."""

from copy import deepcopy

import numpy as np
import pytest
from tenpy.algorithms.tebd import TEBDEngine

import ffsim
from ffsim.tenpy.util import bitstring_to_mps


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
def test_apply_orbital_rotation(
    norb: int,
    nelec: tuple[int, int],
):
    """Test applying an orbital rotation gate to an MPS."""
    rng = np.random.default_rng()

    # generate a random product state
    dim = ffsim.dim(norb, nelec)
    idx = rng.integers(0, high=dim)
    original_vec = ffsim.linalg.one_hot(dim, idx)

    # convert random product state to MPS
    strings_a, strings_b = ffsim.addresses_to_strings(
        [idx],
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.STRING,
        concatenate=False,
    )
    mps = bitstring_to_mps((int(strings_a[0], 2), int(strings_b[0], 2)), norb)
    original_mps = deepcopy(mps)

    # generate a random orbital rotation
    mat = ffsim.random.random_unitary(norb, seed=rng)

    # apply random orbital rotation to state vector
    vec = ffsim.apply_orbital_rotation(original_vec, mat, norb, nelec)

    # apply random orbital rotation to MPS
    eng = TEBDEngine(mps, None, {})
    ffsim.tenpy.apply_orbital_rotation(eng, mat)

    # test expectation is preserved
    original_expectation = np.vdot(original_vec, vec)
    mpo_expectation = mps.overlap(original_mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
