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

from collections.abc import Sequence

import numpy as np
from pyscf.fci import cistring
from scipy.special import comb

from ffsim.gates.orbital_rotation import apply_orbital_rotation


def dims(norb: int, nelec: tuple[int, int]) -> tuple[int, int]:
    """Get the dimensions of the FCI space.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        A pair of integers (dim_a, dim_b) representing the dimensions of the
        alpha- and beta- FCI space.
    """
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    return dim_a, dim_b


def dim(norb: int, nelec: tuple[int, int]) -> int:
    """Get the dimension of the FCI space.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        The dimension of the FCI space.
    """
    dim_a, dim_b = dims(norb, nelec)
    return dim_a * dim_b


def one_hot(shape: tuple[int, ...], index, *, dtype=complex):
    """Return an array of all zeros except for a one at a specified index.

    Args:
        shape: The desired shape of the array.
        index: The index at which to place a one.

    Returns:
        The one-hot vector.
    """
    vec = np.zeros(shape, dtype=dtype)
    vec[index] = 1
    return vec


def slater_determinant(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Return a Slater determinant.

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: A pair of lists of integers. The first list specifies
            the occupied alpha orbitals and the second list specifies the occupied
            beta orbitals.
        orbital_rotation: An optional orbital rotation to apply to the
            electron configuration. In other words, this is a unitary matrix that
            describes the orbitals of the Slater determinant.
        dtype:

    Returns:
        The Slater determinant.
    """
    alpha_orbitals, beta_orbitals = occupied_orbitals
    n_alpha = len(alpha_orbitals)
    n_beta = len(beta_orbitals)
    nelec = (n_alpha, n_beta)
    dim1, dim2 = dims(norb, nelec)
    alpha_bits = np.zeros(norb, dtype=bool)
    alpha_bits[list(alpha_orbitals)] = 1
    alpha_string = int("".join("1" if b else "0" for b in alpha_bits[::-1]), base=2)
    alpha_index = cistring.str2addr(norb, n_alpha, alpha_string)
    beta_bits = np.zeros(norb, dtype=bool)
    beta_bits[list(beta_orbitals)] = 1
    beta_string = int("".join("1" if b else "0" for b in beta_bits[::-1]), base=2)
    beta_index = cistring.str2addr(norb, n_beta, beta_string)
    vec = one_hot((dim1, dim2), (alpha_index, beta_index), dtype=complex).reshape(-1)
    if orbital_rotation is not None:
        vec = apply_orbital_rotation(vec, orbital_rotation, norb=norb, nelec=nelec)
    return vec


def slater_determinant_one_rdm(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    dtype: type = complex,
) -> np.ndarray:
    """Return the one-particle reduced density matrix of a Slater determinant.

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: A tuple of two sequences of integers. The first
            sequence contains the indices of the occupied alpha orbitals, and
            the second sequence similarly for the beta orbitals.

    Returns:
        The one-particle reduced density matrix of the Slater determinant.
    """
    # TODO figure out why mypy complains about this line with
    # error: Need type annotation for "one_rdm"  [var-annotated]
    one_rdm = np.zeros((2 * norb, 2 * norb), dtype=dtype)  # type: ignore
    alpha_orbitals = np.array(occupied_orbitals[0])
    beta_orbitals = np.array(occupied_orbitals[1]) + norb
    one_rdm[(alpha_orbitals, alpha_orbitals)] = 1
    one_rdm[(beta_orbitals, beta_orbitals)] = 1
    return one_rdm
