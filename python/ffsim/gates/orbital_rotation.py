# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Orbital rotation."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from pyscf.fci import cistring

from ffsim import linalg
from ffsim._lib import (
    apply_givens_rotation_in_place,
    apply_phase_shift_in_place,
)
from ffsim.linalg import givens_decomposition
from ffsim.spin import Spin


def apply_orbital_rotation(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    spin: Spin = Spin.ALPHA_AND_BETA,
    *,
    copy: bool = True,
    validate: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> np.ndarray:
    r"""Apply an orbital rotation to a vector.

    An orbital rotation maps creation operators as

    .. math::

        a^\dagger_{\sigma, i} \mapsto \sum_{j} U_{ji} a^\dagger_{\sigma, j}

    where :math:`U` is a unitary matrix. This is equivalent to applying the
    transformation given by

    .. math::

        \prod_{\sigma}
        \exp\left(\sum_{ij} \log(U)_{ij} a^\dagger_{\sigma, i} a_{\sigma, j}\right)

    Args:
        vec: The state vector to be transformed.
        mat: The unitary matrix :math:`U` describing the orbital rotation.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        spin: Choice of spin sector(s) to act on.

            - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
            - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
            - To act on both spin alpha and spin beta, pass
              :const:`ffsim.Spin.ALPHA_AND_BETA` (this is the default value).
        copy: Whether to copy the vector before operating on it.

            - If `copy=True` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If `copy=False` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.
        validate: Whether to check that the input matrix is unitary and raise an error
            if it isn't.
        rtol: Relative numerical tolerance for input validation.
        atol: Absolute numerical tolerance for input validation.

    Returns:
        The transformed vector.

    Raises:
        ValueError: The input matrix is not unitary.
    """
    if validate and not linalg.is_unitary(mat, rtol=rtol, atol=atol):
        raise ValueError("The input orbital rotation matrix is not unitary.")
    if copy:
        vec = vec.copy()

    givens_rotations, phase_shifts = givens_decomposition(mat)
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    vec = vec.reshape((dim_a, dim_b))

    if spin & Spin.ALPHA:
        # transform alpha
        for c, s, i, j in givens_rotations:
            _apply_orbital_rotation_adjacent_spin_in_place(
                vec, c, s.conjugate(), (i, j), norb, n_alpha
            )
        for i, phase_shift in enumerate(phase_shifts):
            indices = _one_subspace_indices(norb, n_alpha, (i,))
            apply_phase_shift_in_place(vec, phase_shift, indices)

    if spin & Spin.BETA:
        # transform beta
        # transpose vector to align memory layout
        vec = vec.T.copy()
        for c, s, i, j in givens_rotations:
            _apply_orbital_rotation_adjacent_spin_in_place(
                vec, c, s.conjugate(), (i, j), norb, n_beta
            )
        for i, phase_shift in enumerate(phase_shifts):
            indices = _one_subspace_indices(norb, n_beta, (i,))
            apply_phase_shift_in_place(vec, phase_shift, indices)
        vec = vec.T.copy()

    return vec.reshape(-1)


def _apply_orbital_rotation_adjacent_spin_in_place(
    vec: np.ndarray,
    c: float,
    s: complex,
    target_orbs: tuple[int, int],
    norb: int,
    nocc: int,
) -> None:
    """Apply an orbital rotation to adjacent orbitals.

    Args:
        vec: Vector to be transformed.
        mat: A 2x2 unitary matrix describing the orbital rotation.
        target_orbs: The orbitals to transform.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
    """
    i, j = target_orbs
    assert i == j + 1 or i == j - 1, "Target orbitals must be adjacent."
    indices = _zero_one_subspace_indices(norb, nocc, target_orbs)
    slice1 = indices[: len(indices) // 2]
    slice2 = indices[len(indices) // 2 :]
    apply_givens_rotation_in_place(vec, c, s, slice1, slice2)


@lru_cache(maxsize=None)
def _zero_one_subspace_indices(
    norb: int, nocc: int, target_orbs: tuple[int, int]
) -> np.ndarray:
    """Return the indices where the target orbitals are 01 or 10."""
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n00 = math.comb(norb - 2, nocc)
    n11 = math.comb(norb - 2, nocc - 2) if nocc >= 2 else 0
    return indices[n00 : len(indices) - n11].astype(np.uint, copy=False)


@lru_cache(maxsize=None)
def _one_subspace_indices(
    norb: int, nocc: int, target_orbs: tuple[int, ...]
) -> np.ndarray:
    """Return the indices where the target orbitals are 1."""
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n0 = math.comb(norb, nocc)
    if nocc >= len(target_orbs):
        n0 -= math.comb(norb - len(target_orbs), nocc - len(target_orbs))
    return indices[n0:].astype(np.uint, copy=False)


def _shifted_orbitals(norb: int, target_orbs: tuple[int, ...]) -> np.ndarray:
    """Return orbital list with targeted orbitals shifted to the end."""
    orbitals = np.arange(norb - len(target_orbs))
    values = sorted(zip(target_orbs, range(norb - len(target_orbs), norb)))
    for index, val in values:
        orbitals = np.insert(orbitals, index, val)
    return orbitals
