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
from functools import cache
from typing import cast, overload

import numpy as np
from pyscf.fci import cistring

from ffsim._lib import apply_givens_rotation_in_place, apply_phase_shift_in_place
from ffsim.linalg import GivensRotation, givens_decomposition


@overload
def apply_orbital_rotation(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: int,
    *,
    copy: bool = True,
) -> np.ndarray: ...
@overload
def apply_orbital_rotation(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray: ...
def apply_orbital_rotation(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    norb: int,
    nelec: int | tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    r"""Apply an orbital rotation to a vector.

    An orbital rotation maps creation operators as

    .. math::

        a^\dagger_{\sigma, i} \mapsto \sum_{j} U^{(\sigma)}_{ji} a^\dagger_{\sigma, j}

    where :math:`U^{(\sigma)}` is a unitary matrix representing the action of the
    orbital rotation on spin sector :math:`\sigma`.
    This is equivalent to applying the transformation given by

    .. math::

        \prod_{\sigma}
        \exp\left(\sum_{ij}
        \log(U^{(\sigma)})_{ij} a^\dagger_{\sigma, i} a_{\sigma, j}\right)

    Args:
        vec: The state vector to be transformed.
        mat: The unitary matrix :math:`U` describing the orbital rotation.
            You can pass either a single Numpy array specifying the orbital rotation
            to apply to both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent orbital rotations for spin alpha and spin beta.
            If passing a pair, you can use ``None`` for one of the
            values in the pair to indicate that no operation should be applied to
            that spin sector.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        copy: Whether to copy the vector before operating on it.

            - If `copy=True` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If `copy=False` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.

    Returns:
        The rotated vector.
    """
    if copy:
        vec = vec.copy()
    if isinstance(nelec, int):
        return _apply_orbital_rotation_spinless(vec, cast(np.ndarray, mat), norb, nelec)
    return _apply_orbital_rotation_spinful(vec, mat, norb, nelec)


def _apply_orbital_rotation_spinless(
    vec: np.ndarray, mat: np.ndarray, norb: int, nelec: int
):
    givens_rotations, phase_shifts = givens_decomposition(mat)
    vec = np.ascontiguousarray(vec.reshape((-1, 1)))
    for c, s, i, j in givens_rotations:
        _apply_orbital_rotation_adjacent_spin_in_place(
            vec, c, s.conjugate(), (i, j), norb, nelec
        )
    for i, phase_shift in enumerate(phase_shifts):
        indices = _one_subspace_indices(norb, nelec, (i,))
        apply_phase_shift_in_place(vec, phase_shift, indices)
    return vec.reshape(-1)


def _apply_orbital_rotation_spinful(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
    norb: int,
    nelec: tuple[int, int],
):
    givens_decomp_a, givens_decomp_b = _get_givens_decomposition(mat)
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    vec = np.ascontiguousarray(vec.reshape((dim_a, dim_b)))

    if givens_decomp_a is not None:
        # transform alpha
        givens_rotations, phase_shifts = givens_decomp_a
        for c, s, i, j in givens_rotations:
            _apply_orbital_rotation_adjacent_spin_in_place(
                vec, c, s.conjugate(), (i, j), norb, n_alpha
            )
        for i, phase_shift in enumerate(phase_shifts):
            indices = _one_subspace_indices(norb, n_alpha, (i,))
            apply_phase_shift_in_place(vec, phase_shift, indices)

    if givens_decomp_b is not None:
        # transform beta
        # copy transposed vector to align memory layout
        vec = np.ascontiguousarray(vec.T)
        givens_rotations, phase_shifts = givens_decomp_b
        for c, s, i, j in givens_rotations:
            _apply_orbital_rotation_adjacent_spin_in_place(
                vec, c, s.conjugate(), (i, j), norb, n_beta
            )
        for i, phase_shift in enumerate(phase_shifts):
            indices = _one_subspace_indices(norb, n_beta, (i,))
            apply_phase_shift_in_place(vec, phase_shift, indices)
        vec = vec.T

    return vec.reshape(-1)


def _get_givens_decomposition(
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
) -> tuple[
    tuple[list[GivensRotation], np.ndarray] | None,
    tuple[list[GivensRotation], np.ndarray] | None,
]:
    if isinstance(mat, np.ndarray) and mat.ndim == 2:
        decomp = givens_decomposition(mat)
        return decomp, decomp
    else:
        mat_a, mat_b = mat
        decomp_a = None
        decomp_b = None
        if mat_a is not None:
            decomp_a = givens_decomposition(mat_a)
        if mat_b is not None:
            decomp_b = givens_decomposition(mat_b)
        return decomp_a, decomp_b


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


@cache
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


@cache
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


@cache
def _shifted_orbitals(norb: int, target_orbs: tuple[int, ...]) -> np.ndarray:
    """Return orbital list with targeted orbitals shifted to the end."""
    orbitals = np.arange(norb - len(target_orbs))
    values = sorted(zip(target_orbs, range(norb - len(target_orbs), norb)))
    for index, val in values:
        orbitals = np.insert(orbitals, index, val)
    return orbitals
