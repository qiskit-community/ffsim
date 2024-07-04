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

from functools import cache

import numpy as np
from pyscf.fci import cistring


@cache
def make_strings(orbitals: range, nocc: int) -> np.ndarray:
    """Cached version of pyscf.fci.cistring.make_strings."""
    return cistring.make_strings(orbitals, nocc)


@cache
def gen_occslst(orbitals: range, nocc: int) -> np.ndarray:
    """Cached version of pyscf.fci.cistring.gen_occslst."""
    return cistring.gen_occslst(orbitals, nocc).astype(np.uint, copy=False)


@cache
def gen_linkstr_index(orbitals: range, nocc: int) -> np.ndarray:
    """Cached version of pyscf.fci.cistring.gen_linkstr_index."""
    return cistring.gen_linkstr_index(orbitals, nocc)


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
