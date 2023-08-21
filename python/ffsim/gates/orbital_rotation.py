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

from functools import lru_cache
from typing import Literal, overload

import numpy as np
import scipy.linalg
from pyscf.fci import cistring
from scipy.special import comb

from ffsim._ffsim import (
    apply_givens_rotation_in_place,
    apply_phase_shift_in_place,
    apply_single_column_transformation_in_place,
    gen_orbital_rotation_index_in_place,
)
from ffsim.linalg import givens_decomposition, lup


def gen_orbital_rotation_index(
    norb: int, nocc: int, linkstr_index: np.ndarray | None = None
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

    if linkstr_index is None:
        linkstr_index = cistring.gen_linkstr_index(range(norb), nocc)
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


@overload
def apply_orbital_rotation(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    ...


@overload
def apply_orbital_rotation(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    allow_row_permutation: Literal[True],
    orbital_rotation_index_a: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    orbital_rotation_index_b: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    copy: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    ...


@overload
def apply_orbital_rotation(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    allow_col_permutation: Literal[True],
    orbital_rotation_index_a: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    orbital_rotation_index_b: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    copy: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    ...


def apply_orbital_rotation(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    allow_row_permutation: bool = False,
    allow_col_permutation: bool = False,
    orbital_rotation_index_a: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    orbital_rotation_index_b: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    copy: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    r"""Apply an orbital rotation to a vector.

    An orbital rotation maps creation operators as

    .. math::

        a^\dagger_i \mapsto \sum_{j} U_{ji} a^\dagger_j

    where :math:`U` is a unitary matrix. This is equivalent to applying the
    transformation given by

    .. math::

        \exp(\sum_{ij} log(U)_{ij} a^\dagger{i} a_j)

    Args:
        vec: The state vector to be transformed.
        mat: The unitary matrix :math:`U` describing the orbital rotation.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        allow_row_permutation: Whether to allow a permutation of the rows
            of the orbital rotation matrix.
        allow_col_permutation: Whether to allow a permutation of the columns
            of the orbital rotation matrix.
        orbital_rotation_index_a: The orbital rotation index for alpha strings.
        orbital_rotation_index_b: The orbital rotation index for beta strings.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.

    Returns:
        The transformed vector. If a row or column permutation was allowed,
        the permutation matrix ``P`` is returned as well.
        If a row permutation was allowed, then the transformation
        actually effected is given by the matrix ``P @ mat``. If a column
        permutation was allowed, then it is ``mat @ P``.

    Raises:
        ValueError: If both ``allow_row_permutation`` and ``allow_col_permutation``
        are set to True. Only one of these is allowed to be set to True at a time.
    """
    if allow_row_permutation and allow_col_permutation:
        raise ValueError(
            "You can choose to allow either row or column permutations, but not both."
        )
    if allow_row_permutation or allow_col_permutation:
        n_alpha, n_beta = nelec
        if orbital_rotation_index_a is None:
            orbital_rotation_index_a = gen_orbital_rotation_index(norb, n_alpha)
        if orbital_rotation_index_b is None:
            orbital_rotation_index_b = gen_orbital_rotation_index(norb, n_beta)
        return _apply_orbital_rotation_lu(
            vec,
            mat,
            norb,
            nelec,
            permute_rows=allow_row_permutation,
            orbital_rotation_index_a=orbital_rotation_index_a,
            orbital_rotation_index_b=orbital_rotation_index_b,
            copy=copy,
        )
    return _apply_orbital_rotation_givens(vec, mat, norb, nelec, copy=copy)


def _apply_orbital_rotation_lu(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    permute_rows: bool,
    orbital_rotation_index_a: tuple[np.ndarray, np.ndarray, np.ndarray],
    orbital_rotation_index_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    copy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if copy:
        vec = vec.copy()
    if permute_rows:
        lower, upper, perm = lup(mat.T.conj())
    else:
        perm, lower, upper = scipy.linalg.lu(mat.T.conj())
    eye = np.eye(norb, dtype=complex)
    transformation_mat = eye - lower + scipy.linalg.solve_triangular(upper, eye)
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    vec = vec.reshape((dim_a, dim_b))
    # transform alpha
    _apply_orbital_rotation_spin_in_place(
        vec,
        transformation_mat,
        norb=norb,
        orbital_rotation_index=orbital_rotation_index_a,
    )
    # transform beta
    # transpose vector to align memory layout
    vec = vec.T.copy()
    _apply_orbital_rotation_spin_in_place(
        vec,
        transformation_mat,
        norb=norb,
        orbital_rotation_index=orbital_rotation_index_b,
    )
    return vec.T.copy().reshape(-1), perm


def _apply_orbital_rotation_spin_in_place(
    vec: np.ndarray,
    transformation_mat: np.ndarray,
    norb: int,
    orbital_rotation_index: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    diag_strings, off_diag_strings, off_diag_index = orbital_rotation_index
    for i in range(norb):
        apply_single_column_transformation_in_place(
            vec,
            transformation_mat[:, i],
            diag_val=transformation_mat[i, i],
            diag_strings=diag_strings[i],
            off_diag_strings=off_diag_strings[i],
            off_diag_index=off_diag_index[i],
        )


def _apply_orbital_rotation_givens(
    vec: np.ndarray, mat: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool
) -> np.ndarray:
    if copy:
        vec = vec.copy()
    givens_rotations, phase_shifts = givens_decomposition(mat)

    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    vec = vec.reshape((dim_a, dim_b))
    # transform alpha
    for givens_mat, target_orbs in givens_rotations:
        _apply_orbital_rotation_adjacent_spin_in_place(
            vec, givens_mat.conj(), target_orbs, norb, n_alpha
        )
    for i, phase_shift in enumerate(phase_shifts):
        indices = _one_subspace_indices(norb, n_alpha, (i,))
        apply_phase_shift_in_place(vec, phase_shift, indices)

    # transform beta
    # transpose vector to align memory layout
    vec = vec.T.copy()
    for givens_mat, target_orbs in givens_rotations:
        _apply_orbital_rotation_adjacent_spin_in_place(
            vec, givens_mat.conj(), target_orbs, norb, n_beta
        )
    for i, phase_shift in enumerate(phase_shifts):
        indices = _one_subspace_indices(norb, n_beta, (i,))
        apply_phase_shift_in_place(vec, phase_shift, indices)

    return vec.T.copy().reshape(-1)


def _apply_orbital_rotation_adjacent_spin_in_place(
    vec: np.ndarray, mat: np.ndarray, target_orbs: tuple[int, int], norb: int, nocc: int
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
    c, s = mat[0]
    apply_givens_rotation_in_place(vec, c.real, abs(s), s / abs(s), slice1, slice2)


@lru_cache(maxsize=None)
def _zero_one_subspace_indices(
    norb: int, nocc: int, target_orbs: tuple[int, int]
) -> np.ndarray:
    """Return the indices where the target orbitals are 01 or 10."""
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n00 = comb(norb - 2, nocc, exact=True)
    n11 = comb(norb - 2, nocc - 2, exact=True)
    return indices[n00 : len(indices) - n11].astype(np.uint, copy=False)


@lru_cache(maxsize=None)
def _one_subspace_indices(
    norb: int, nocc: int, target_orbs: tuple[int, ...]
) -> np.ndarray:
    """Return the indices where the target orbitals are 1."""
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n0 = comb(norb, nocc, exact=True) - comb(
        norb - len(target_orbs), nocc - len(target_orbs), exact=True
    )
    return indices[n0:].astype(np.uint, copy=False)


def _shifted_orbitals(norb: int, target_orbs: tuple[int, ...]) -> np.ndarray:
    """Return orbital list with targeted orbitals shifted to the end."""
    orbitals = np.arange(norb - len(target_orbs))
    values = sorted(zip(target_orbs, range(norb - len(target_orbs), norb)))
    for index, val in values:
        orbitals = np.insert(orbitals, index, val)
    return orbitals
