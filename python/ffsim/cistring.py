# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tools for handling FCI strings."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyscf.fci import cistring
from scipy.special import comb

from ffsim._lib import gen_orbital_rotation_index_in_place


@lru_cache(maxsize=None)
def make_strings(orbitals: range, nocc: int) -> np.ndarray:
    """Cached version of pyscf.fci.cistring.make_strings."""
    return cistring.make_strings(orbitals, nocc)


@lru_cache(maxsize=None)
def gen_occslst(orbitals: range, nocc: int) -> np.ndarray:
    """Cached version of pyscf.fci.cistring.gen_occslst."""
    return cistring.gen_occslst(orbitals, nocc).astype(np.uint, copy=False)


@lru_cache(maxsize=None)
def gen_linkstr_index(orbitals: range, nocc: int):
    """Cached version of pyscf.fci.cistring.gen_linkstr_index."""
    return cistring.gen_linkstr_index(orbitals, nocc)


@lru_cache(maxsize=None)
def gen_orbital_rotation_index(
    norb: int, nocc: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate string index used for performing orbital rotations.

    Returns a tuple (diag_strings, off_diag_strings, off_diag_index)
    of three Numpy arrays.

    diag_strings is a norb x binom(norb - 1, nocc - 1) array.
    The i-th row of this array contains all the strings with orbital i occupied.

    off_diag_strings is a norb x binom(norb - 1, nocc) array.
    The i-th row of this array contains all the strings with orbital i unoccupied.

    off_diag_index is a norb x binom(norb - 1, nocc) x nocc x 3 array.
    The first two axes of this array are in one-to-one correspondence with
    off_diag_strings. For a fixed choice (i, str0) for the first two axes,
    the last two axes form a nocc x 3 array. Each row of this array is a tuple
    (j, str1, sign) where str1 is formed by annihilating orbital j in str0 and creating
    orbital i, with sign giving the fermionic parity of this operation.
    """
    if nocc == 0:
        diag_strings = np.zeros((norb, 0), dtype=np.uint)
        off_diag_strings = np.zeros((norb, 1), dtype=np.uint)
        off_diag_index = np.zeros((norb, 1, 0, 3), dtype=np.int32)
        return diag_strings, off_diag_strings, off_diag_index

    linkstr_index = gen_linkstr_index(range(norb), nocc)
    dim_diag = comb(norb - 1, nocc - 1, exact=True)
    dim_off_diag = comb(norb - 1, nocc, exact=True)
    dim = dim_diag + dim_off_diag
    diag_strings = np.empty((norb, dim_diag), dtype=np.uint)
    off_diag_strings = np.empty((norb, dim_off_diag), dtype=np.uint)
    # TODO should this be int64? pyscf uses int32 for linkstr_index though
    off_diag_index = np.empty((norb, dim_off_diag, nocc, 3), dtype=np.int32)
    off_diag_strings_index = np.empty((norb, dim), dtype=np.uint)
    gen_orbital_rotation_index_in_place(
        norb=norb,
        nocc=nocc,
        linkstr_index=linkstr_index,
        diag_strings=diag_strings,
        off_diag_strings=off_diag_strings,
        off_diag_strings_index=off_diag_strings_index,
        off_diag_index=off_diag_index,
    )
    return diag_strings, off_diag_strings, off_diag_index


def init_cache(norb: int, nelec: tuple[int, int]) -> None:
    """Initialize cached objects.

    Call this function to prepare ffsim for performing operations with given values
    of `norb` and `nelec`. Typically there is no need to call this function, but it
    should be called before benchmarking to avoid counting the cost of initializing
    cached lookup tables.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
    """
    for nocc in nelec:
        make_strings(range(norb), nocc)
        gen_occslst(range(norb), nocc)
        gen_linkstr_index(range(norb), nocc)
        gen_orbital_rotation_index(norb, nocc)
