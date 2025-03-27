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

from functools import cache

import numpy as np

from ffsim import states
from ffsim.cistring import make_strings


def qiskit_vec_to_ffsim_vec(
    vec: np.ndarray, norb: int, nelec: int | tuple[int, int]
) -> np.ndarray:
    """Convert a Qiskit state vector to an ffsim state vector.

    Args:
        vec: A state vector in Qiskit format. It should be a one-dimensional vector
            of length ``2 ** (2 * norb)`` in the spinful case, and ``2 ** norb`` in the
            spinless case.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
    """
    n_qubits = norb if isinstance(nelec, int) else 2 * norb
    assert vec.shape == (1 << n_qubits,)
    return vec[_ffsim_indices(norb, nelec)]


def ffsim_vec_to_qiskit_vec(
    vec: np.ndarray, norb: int, nelec: int | tuple[int, int]
) -> np.ndarray:
    """Convert an ffsim state vector to a Qiskit state vector.

    Args:
        vec: A state vector in ffsim/PySCF format. It should be a one-dimensional vector
            of length ``comb(norb, n_alpha) * comb(norb, n_beta)`` in the spinful case,
            and ``comb(norb, nelec)`` in the spinless case.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
    """
    n_qubits = norb if isinstance(nelec, int) else 2 * norb
    assert vec.shape == (states.dim(norb, nelec),)
    qiskit_vec = np.zeros(1 << n_qubits, dtype=vec.dtype)
    qiskit_vec[_ffsim_indices(norb, nelec)] = vec
    return qiskit_vec


@cache
def _ffsim_indices(norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
    if isinstance(nelec, int):
        return make_strings(range(norb), nelec)
    n_alpha, n_beta = nelec
    strings_a = make_strings(range(norb), n_alpha)
    strings_b = make_strings(range(norb), n_beta) << norb
    # Compute [a + b for a, b in product(strings_a, strings_b)]
    return (strings_a.reshape(-1, 1) + strings_b).reshape(-1)
