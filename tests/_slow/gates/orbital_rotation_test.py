# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import numpy as np
from pyscf.fci import cistring
from scipy.special import comb

import ffsim
from ffsim._lib import (
    apply_givens_rotation_in_place,
    apply_single_column_transformation_in_place,
    gen_orbital_rotation_index_in_place,
)
from ffsim._slow.gates.orbital_rotation import (
    apply_givens_rotation_in_place_slow,
    apply_single_column_transformation_in_place_slow,
    gen_orbital_rotation_index_in_place_slow,
)
from ffsim.gates.orbital_rotation import (
    _zero_one_subspace_indices,
    gen_orbital_rotation_index,
)


def test_apply_givens_rotation_in_place_slow():
    """Test applying Givens rotation."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        vec_slow = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        c = rng.uniform(0, 1)
        s = (1j) ** rng.uniform(0, 4) * np.sqrt(1 - c**2)
        indices = _zero_one_subspace_indices(norb, n_alpha, (1, 3))
        slice1 = indices[: len(indices) // 2]
        slice2 = indices[len(indices) // 2 :]
        apply_givens_rotation_in_place_slow(vec_slow, c, s, slice1, slice2)
        apply_givens_rotation_in_place(vec_fast, c, s, slice1, slice2)
        np.testing.assert_allclose(vec_slow, vec_fast)


def test_gen_orbital_rotation_index_in_place_slow():
    """Test generating orbital rotation index."""
    norb = 5
    rng = np.random.default_rng()
    nocc = rng.integers(1, norb + 1)
    linkstr_index = cistring.gen_linkstr_index(range(norb), nocc)

    dim_diag = comb(norb - 1, nocc - 1, exact=True)
    dim_off_diag = comb(norb - 1, nocc, exact=True)
    dim = dim_diag + dim_off_diag

    diag_strings_slow = np.empty((norb, dim_diag), dtype=np.uint)
    off_diag_strings_slow = np.empty((norb, dim_off_diag), dtype=np.uint)
    off_diag_index_slow = np.empty((norb, dim_off_diag, nocc, 3), dtype=np.int32)
    off_diag_strings_index_slow = np.empty((norb, dim), dtype=np.uint)

    diag_strings_fast = np.empty((norb, dim_diag), dtype=np.uint)
    off_diag_strings_fast = np.empty((norb, dim_off_diag), dtype=np.uint)
    off_diag_index_fast = np.empty((norb, dim_off_diag, nocc, 3), dtype=np.int32)
    off_diag_strings_index_fast = np.empty((norb, dim), dtype=np.uint)

    gen_orbital_rotation_index_in_place_slow(
        norb=norb,
        nocc=nocc,
        linkstr_index=linkstr_index,
        diag_strings=diag_strings_slow,
        off_diag_strings=off_diag_strings_slow,
        off_diag_strings_index=off_diag_strings_index_slow,
        off_diag_index=off_diag_index_slow,
    )
    gen_orbital_rotation_index_in_place(
        norb=norb,
        nocc=nocc,
        linkstr_index=linkstr_index,
        diag_strings=diag_strings_fast,
        off_diag_strings=off_diag_strings_fast,
        off_diag_strings_index=off_diag_strings_index_fast,
        off_diag_index=off_diag_index_fast,
    )
    np.testing.assert_array_equal(diag_strings_slow, diag_strings_fast)
    np.testing.assert_array_equal(off_diag_strings_slow, off_diag_strings_fast)
    np.testing.assert_array_equal(off_diag_index_slow, off_diag_index_fast)


def test_apply_single_column_transformation_in_place_slow():
    """Test applying single column transformation."""
    norb = 5
    rng = np.random.default_rng()
    for _ in range(5):
        n_alpha = rng.integers(1, norb + 1)
        n_beta = rng.integers(1, norb + 1)
        dim_a = comb(norb, n_alpha, exact=True)
        dim_b = comb(norb, n_beta, exact=True)
        orbital_rotation_index = gen_orbital_rotation_index(norb, n_alpha)
        column = rng.uniform(size=norb) + 1j * rng.uniform(size=norb)
        diag_val = rng.uniform() + 1j * rng.uniform()
        vec_slow = ffsim.random.random_statevector(dim_a * dim_b, seed=rng).reshape(
            (dim_a, dim_b)
        )
        vec_fast = vec_slow.copy()
        index = [a[0] for a in orbital_rotation_index]
        apply_single_column_transformation_in_place_slow(
            vec_slow, column, diag_val, *index
        )
        apply_single_column_transformation_in_place(vec_fast, column, diag_val, *index)
        np.testing.assert_allclose(vec_slow, vec_fast)
