# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ffsim import states
from ffsim.cistring import make_strings


def qiskit_vec_to_ffsim_vec(
    vec: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    """Convert a Qiskit statevector to an ffsim statevector.

    Args:
        vec: A statevector in Qiskit format. It should be a one-dimensional vector
            of length ``2 ** (2 * norb)``.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
    """
    assert vec.shape == (1 << (2 * norb),)
    return vec[_ffsim_indices(norb, nelec)]


def ffsim_vec_to_qiskit_vec(
    vec: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    """Convert an ffsim statevector to a Qiskit statevector.

    Args:
        vec: A statevector in ffsim/PySCF format. It should be a one-dimensional vector
            of length ``comb(norb, n_alpha) * comb(norb, n_beta)``.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
    """
    assert vec.shape == (states.dim(norb, nelec),)
    qiskit_vec = np.zeros(1 << (2 * norb), dtype=vec.dtype)
    qiskit_vec[_ffsim_indices(norb, nelec)] = vec
    return qiskit_vec


@lru_cache(maxsize=None)
def _ffsim_indices(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    n_alpha, n_beta = nelec
    strings_a = make_strings(range(norb), n_alpha)
    strings_b = make_strings(range(norb), n_beta) << norb
    # Compute [a + b for a, b in product(strings_a, strings_b)]
    return (strings_a.reshape(-1, 1) + strings_b).reshape(-1).copy()
