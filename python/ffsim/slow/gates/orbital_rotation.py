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
from scipy.linalg.lapack import zrot


def gen_orbital_rotation_index_in_place_slow(
    norb: int,
    nocc: int,
    linkstr_index: np.ndarray,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_strings_index: np.ndarray,
    off_diag_index: np.ndarray,
) -> None:
    """Generate orbital rotation index."""
    diag_counter = np.zeros(norb, dtype=np.uint)
    off_diag_counter = np.zeros(norb, dtype=np.uint)
    for str0, tab in enumerate(linkstr_index[:, :, 0]):
        for orb in tab[:nocc]:
            count = diag_counter[orb]
            diag_strings[orb, count] = str0
            diag_counter[orb] += 1
        for orb in tab[nocc:norb]:
            count = off_diag_counter[orb]
            off_diag_strings[orb, count] = str0
            off_diag_strings_index[orb, str0] = count
            off_diag_counter[orb] += 1

    index_counter = np.zeros_like(off_diag_strings)
    for str0, tab in enumerate(linkstr_index):
        for orb_c, orb_d, str1, sign in tab[nocc:]:
            # str0 -> annihilate orb_d -> create orb_c -> str1
            index = off_diag_strings_index[orb_c, str0]
            count = index_counter[orb_c, index]
            off_diag_index[orb_c, index, count] = orb_d, str1, sign
            index_counter[orb_c, index] += 1


def apply_single_column_transformation_in_place_slow(
    vec: np.ndarray,
    column: np.ndarray,
    diag_val: complex,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_index: np.ndarray,
) -> None:
    """Apply a single-column orbital rotation."""
    for str0, tab in zip(off_diag_strings, off_diag_index):
        for orb, str1, sign in tab:
            vec[str0] += sign * column[orb] * vec[str1]
    for str0 in diag_strings:
        vec[str0] *= diag_val


def apply_givens_rotation_in_place_slow(
    vec: np.ndarray,
    c: float,
    s: complex,
    slice1: np.ndarray,
    slice2: np.ndarray,
) -> None:
    """Apply a Givens rotation to slices of a state vector."""
    for i, j in zip(slice1, slice2):
        vec[i], vec[j] = zrot(vec[i], vec[j], c, s)
