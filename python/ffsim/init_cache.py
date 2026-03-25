# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Function for initializing cached objects.

The :func:`~init_cache` function prepares ffsim to perform operations with given values
of ``norb`` and ``nelec``. Typically there is no need to call this function.
"""

from ffsim.cistring import (
    gen_linkstr_index,
    gen_linkstr_index_trilidx,
    gen_occslst,
    make_strings,
)


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
        gen_linkstr_index_trilidx(range(norb), nocc)
