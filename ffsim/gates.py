# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermionic quantum computation gates."""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import numpy as np
import scipy.linalg
from pyscf.fci import cistring
from scipy.special import comb

from ffsim._ffsim import (
    apply_diag_coulomb_evolution_in_place,
    apply_givens_rotation_in_place,
    apply_num_op_sum_evolution_in_place,
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
        nocc=n_alpha,
        orbital_rotation_index=orbital_rotation_index_a,
    )
    # transform beta
    # transpose vector to align memory layout
    vec = vec.T.copy()
    _apply_orbital_rotation_spin_in_place(
        vec,
        transformation_mat,
        norb=norb,
        nocc=n_beta,
        orbital_rotation_index=orbital_rotation_index_b,
    )
    return vec.T.copy().reshape(-1), perm


def _apply_orbital_rotation_spin_in_place(
    vec: np.ndarray,
    transformation_mat: np.ndarray,
    norb: int,
    nocc: int,
    orbital_rotation_index=tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    dim_diag = comb(norb - 1, nocc - 1, exact=True)
    dim_off_diag = comb(norb - 1, nocc, exact=True)
    dim = dim_diag + dim_off_diag
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
) -> np.ndarray:
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


def _apply_phase_shift(
    vec: np.ndarray,
    phase: complex,
    target_orbs: tuple[tuple[int, ...], tuple[int, ...]],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    """Apply a phase shift controlled on target orbitals.

    Multiplies by the phase each coefficient corresponding to a string in which
    the target orbitals are all 1 (occupied).
    """
    if copy:
        vec = vec.copy()
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    vec = vec.reshape((dim_a, dim_b))
    target_orbs_a, target_orbs_b = target_orbs
    indices_a = _one_subspace_indices(norb, n_alpha, target_orbs_a)
    indices_b = _one_subspace_indices(norb, n_beta, target_orbs_b)
    vec[np.ix_(indices_a, indices_b)] *= phase
    return vec.reshape(-1)


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


def apply_num_op_sum_evolution(
    vec: np.ndarray,
    coeffs: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    orbital_rotation: np.ndarray | None = None,
    *,
    occupations_a: np.ndarray | None = None,
    occupations_b: np.ndarray | None = None,
    orbital_rotation_index_a: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    orbital_rotation_index_b: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    copy: bool = True,
):
    """Apply time evolution by a (rotated) linear combination of number operators.

    Applies

    .. math::
        \mathcal{U}
        \exp(-i t \sum_{i, \sigma} \lambda_i n_{i, \sigma})
        \mathcal{U}^\dagger

    where :math:`n_{i, \sigma}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, the :math:`\lambda_i` are real numbers, and
    :math:`\mathcal{U}` is an optional orbital rotation.

    Args:
        vec: The state vector to be transformed.
        coeffs: The coefficients of the linear combination.
        time: The evolution time.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        orbital_rotation: A unitary matrix describing the optional orbital rotation.
        occupations_a: List of occupied orbital lists for alpha strings.
        occupations_b: List of occupied orbital lists for beta strings.
        orbital_rotation_index_a: The orbital rotation index for alpha strings.
        orbital_rotation_index_b: The orbital rotation index for beta strings.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.
    """
    if copy:
        vec = vec.copy()

    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    if occupations_a is None:
        occupations_a = cistring._gen_occslst(range(norb), n_alpha)
    if occupations_b is None:
        occupations_b = cistring._gen_occslst(range(norb), n_beta)
    occupations_a = occupations_a.astype(np.uint, copy=False)
    occupations_b = occupations_b.astype(np.uint, copy=False)

    if orbital_rotation is not None:
        vec, perm0 = apply_orbital_rotation(
            vec,
            orbital_rotation.T.conj(),
            norb,
            nelec,
            allow_row_permutation=True,
            orbital_rotation_index_a=orbital_rotation_index_a,
            orbital_rotation_index_b=orbital_rotation_index_b,
            copy=False,
        )
        coeffs = perm0 @ coeffs

    phases = np.exp(-1j * time * coeffs)
    vec = vec.reshape((dim_a, dim_b))
    # apply alpha
    apply_num_op_sum_evolution_in_place(vec, phases, occupations=occupations_a)
    # apply beta
    vec = vec.T
    apply_num_op_sum_evolution_in_place(vec, phases, occupations=occupations_b)
    vec = vec.T.reshape(-1)

    if orbital_rotation is not None:
        vec, perm1 = apply_orbital_rotation(
            vec,
            orbital_rotation,
            norb,
            nelec,
            allow_col_permutation=True,
            orbital_rotation_index_a=orbital_rotation_index_a,
            orbital_rotation_index_b=orbital_rotation_index_b,
            copy=False,
        )
        np.testing.assert_allclose(perm1.T, perm0)

    return vec


def apply_diag_coulomb_evolution(
    vec: np.ndarray,
    mat: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    orbital_rotation: np.ndarray | None = None,
    *,
    mat_alpha_beta: np.ndarray | None = None,
    occupations_a: np.ndarray | None = None,
    occupations_b: np.ndarray | None = None,
    orbital_rotation_index_a: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    orbital_rotation_index_b: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    copy: bool = True,
) -> np.ndarray:
    r"""Apply time evolution by a (rotated) diagonal Coulomb operator.

    Applies

    .. math::
        \mathcal{U}
        \exp(-i t \sum_{i, j, \sigma, \tau} Z_{ij} n_{i, \sigma} n_{j, \tau} / 2)
        \mathcal{U}^\dagger

    where :math:`n_{i, \sigma}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, :math:`Z` is a real symmetric matrix,
    and :math:`\mathcal{U}` is an optional orbital rotation.
    If ``mat_alpha_beta`` is also given, then it is used in place of :math:`Z`
    for the terms in the sum where the spins differ (:math:`\sigma \neq \tau`).

    Args:
        vec: The state vector to be transformed.
        mat: The real symmetric matrix :math:`Z`.
        time: The evolution time.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        orbital_rotation: A unitary matrix describing the optional orbital rotation.
        mat_alpha_beta: A matrix of coefficients to use for interactions between
            orbitals with differing spin.
        occupations_a: List of occupied orbital lists for alpha strings.
        occupations_b: List of occupied orbital lists for beta strings.
        orbital_rotation_index_a: The orbital rotation index for alpha strings.
        orbital_rotation_index_b: The orbital rotation index for beta strings.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.
    """
    if copy:
        vec = vec.copy()
    if mat_alpha_beta is None:
        mat_alpha_beta = mat

    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    if occupations_a is None:
        occupations_a = cistring._gen_occslst(range(norb), n_alpha)
    if occupations_b is None:
        occupations_b = cistring._gen_occslst(range(norb), n_beta)
    occupations_a = occupations_a.astype(np.uint, copy=False)
    occupations_b = occupations_b.astype(np.uint, copy=False)

    if orbital_rotation is not None:
        vec, perm0 = apply_orbital_rotation(
            vec,
            orbital_rotation.T.conj(),
            norb,
            nelec,
            allow_row_permutation=True,
            orbital_rotation_index_a=orbital_rotation_index_a,
            orbital_rotation_index_b=orbital_rotation_index_b,
            copy=False,
        )
        mat = perm0 @ mat @ perm0.T
        mat_alpha_beta = perm0 @ mat_alpha_beta @ perm0.T

    mat_exp = mat.copy()
    mat_exp[np.diag_indices(norb)] *= 0.5
    mat_exp = np.exp(-1j * time * mat_exp)
    mat_alpha_beta_exp = np.exp(-1j * time * cast(np.ndarray, mat_alpha_beta))
    vec = vec.reshape((dim_a, dim_b))
    apply_diag_coulomb_evolution_in_place(
        vec,
        mat_exp,
        norb=norb,
        mat_alpha_beta_exp=mat_alpha_beta_exp,
        occupations_a=occupations_a,
        occupations_b=occupations_b,
    )
    vec = vec.reshape(-1)

    if orbital_rotation is not None:
        vec, perm1 = apply_orbital_rotation(
            vec,
            orbital_rotation,
            norb,
            nelec,
            allow_col_permutation=True,
            orbital_rotation_index_a=orbital_rotation_index_a,
            orbital_rotation_index_b=orbital_rotation_index_b,
            copy=False,
        )
        np.testing.assert_allclose(perm0, perm1.T)

    return vec


def apply_givens_rotation(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a Givens rotation gate.

    The Givens rotation gate is

    .. math::
        G(\theta) = \exp(\theta (a^\dagger_i a_j - a\dagger_j a_i))

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals (i, j) to rotate.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(norb)
    mat[np.ix_(target_orbs, target_orbs)] = [[c, s], [-s, c]]
    return apply_orbital_rotation(vec, mat, norb=norb, nelec=nelec, copy=copy)


def apply_tunneling_interaction(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a tunneling interaction gate.

    The tunneling interaction gate is

    .. math::
        T(\theta) = \exp(i \theta (a^\dagger_i a_j + a\dagger_j a_i))

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals (i, j) on which to apply the interaction.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    vec = apply_num_interaction(
        vec, -np.pi / 2, target_orbs[0], norb=norb, nelec=nelec, copy=copy
    )
    vec = apply_givens_rotation(
        vec, theta, target_orbs, norb=norb, nelec=nelec, copy=False
    )
    vec = apply_num_interaction(
        vec, np.pi / 2, target_orbs[0], norb=norb, nelec=nelec, copy=False
    )
    return vec


def apply_num_interaction(
    vec: np.ndarray,
    theta: float,
    target_orb: int,
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a number interaction gate.

    The number interaction gate is

    .. math::
        N(\theta) = \exp(i \theta a^\dagger_i a_i)

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orb: The orbital on which to apply the interaction.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.
    """
    if copy:
        vec = vec.copy()
    for sigma in range(2):
        vec = apply_num_op_prod_interaction(
            vec, theta, ((target_orb, sigma),), norb=norb, nelec=nelec, copy=False
        )
    return vec


def apply_num_op_prod_interaction(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[tuple[int, int], ...],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply interaction gate for product of number operators.

    The gate is

    .. math::
        NP(\theta) = \exp(i \theta \prod a^\dagger_{i, \sigma} a_{i, \sigma})

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals on which to apply the interaction. This should
            be a tuple of (orbital, spin) pairs.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.
    """
    if copy:
        vec = vec.copy()
    orbitals: list[set[int]] = [set(), set()]
    for i, spin_i in target_orbs:
        orbitals[spin_i].add(i)
    vec = _apply_phase_shift(
        vec,
        np.exp(1j * theta),
        (tuple(orbitals[0]), tuple(orbitals[1])),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec
