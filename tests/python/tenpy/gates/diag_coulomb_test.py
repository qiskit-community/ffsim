# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the TeNPy diagonal Coulomb evolution gate."""

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
def test_apply_diag_coulomb_evolution(
    norb: int,
    nelec: tuple[int, int],
):
    """Test applying a diagonal Coulomb evolution gate to an MPS."""
    rng = np.random.default_rng()

    # generate a random product state
    dim = ffsim.dim(norb, nelec)
    idx = rng.integers(0, high=dim)
    original_vec = np.zeros(dim, dtype=complex)
    original_vec[idx] = 1

    # convert random product state to MPS
    strings_a, strings_b = ffsim.addresses_to_strings(
        range(dim),
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.STRING,
        concatenate=False,
    )
    mps = bitstring_to_mps((int(strings_a[idx], 2), int(strings_b[idx], 2)), norb)
    original_mps = deepcopy(mps)

    # generate random diagonal Coulomb evolution parameters
    mat_aa = np.diag(rng.standard_normal(norb - 1), k=-1)
    mat_aa += mat_aa.T
    mat_ab = np.diag(rng.standard_normal(norb))
    diag_coulomb_mats = np.array([mat_aa, mat_ab, mat_aa])

    # apply random diagonal Coulomb evolution to state vector
    vec = ffsim.apply_diag_coulomb_evolution(
        original_vec, diag_coulomb_mats, 1, norb, nelec
    )

    # apply random diagonal Coulomb evolution to MPS
    options = {"trunc_params": {"chi_max": 16, "svd_min": 1e-6}}
    eng = TEBDEngine(mps, None, options)
    ffsim.tenpy.apply_diag_coulomb_evolution(eng, diag_coulomb_mats[:2])

    # test expectation is preserved
    original_expectation = np.vdot(original_vec, vec)
    mpo_expectation = original_mps.overlap(mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
