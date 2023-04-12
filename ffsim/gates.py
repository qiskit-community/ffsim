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

import numpy as np
import scipy.linalg
from pyscf.fci import cistring
from scipy.special import comb

from ffsim._ffsim import (
    apply_diag_coulomb_evolution_in_place,
    apply_num_op_sum_evolution_in_place,
    apply_single_column_transformation_in_place,
    gen_orbital_rotation_index_in_place,
)
from ffsim.linalg import apply_matrix_to_slices, givens_decomposition, lup


def apply_orbital_rotation(
    mat: np.ndarray,
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    allow_row_permutation: bool = False,
    allow_col_permutation: bool = False,
    # TODO rename "copy" to "overwrite_vec"
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
        mat: Unitary matrix :math:`U` describing the orbital rotation.
        vec: Vector to be transformed.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
    """
    if allow_row_permutation and allow_col_permutation:
        raise ValueError(
            "You can choose to allow either row or column permutations, but not both."
        )
    if allow_row_permutation or allow_col_permutation:
        return _apply_orbital_rotation_lu(
            mat,
            vec,
            norb,
            nelec,
            permute_rows=allow_row_permutation,
            copy=copy,
        )
    return _apply_orbital_rotation_givens(mat, vec, norb, nelec)


def _apply_orbital_rotation_lu(
    mat: np.ndarray,
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    permute_rows: bool = False,
    copy: bool = True,
) -> np.ndarray:
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
        transformation_mat, vec, norb=norb, nocc=n_alpha
    )
    # transform beta
    # transpose vector to align memory layout
    vec = vec.T.copy()
    _apply_orbital_rotation_spin_in_place(
        transformation_mat, vec, norb=norb, nocc=n_beta
    )
    return vec.T.copy().reshape(-1), perm


def _apply_orbital_rotation_spin_in_place(
    transformation_mat: np.ndarray, vec: np.ndarray, norb: int, nocc: int
) -> None:
    linkstr_index = cistring.gen_linkstr_index(range(norb), nocc)
    dim_diag = comb(norb - 1, nocc - 1, exact=True)
    dim_off_diag = comb(norb - 1, nocc, exact=True)
    dim = dim_diag + dim_off_diag
    # TODO double check dtypes
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
    for i in range(norb):
        apply_single_column_transformation_in_place(
            transformation_mat[:, i],
            vec,
            diag_val=transformation_mat[i, i],
            diag_strings=diag_strings[i],
            off_diag_strings=off_diag_strings[i],
            off_diag_index=off_diag_index[i],
        )


def _apply_orbital_rotation_givens(
    mat: np.ndarray, vec: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    givens_rotations, phase_shifts = givens_decomposition(mat)
    buf1 = np.empty_like(vec)
    buf2 = np.empty_like(vec)
    buf1[...] = vec[...]
    for givens_mat, target_orbs in givens_rotations:
        vec = _apply_orbital_rotation_adjacent(
            givens_mat.conj(), buf1, target_orbs, norb, nelec, out=buf2
        )
        buf1, buf2 = buf2, buf1
    for i, phase_shift in enumerate(phase_shifts):
        apply_phase_shift(phase_shift, vec, ((i,), ()), norb, nelec, copy=False)
        apply_phase_shift(phase_shift, vec, ((), (i,)), norb, nelec, copy=False)
    return vec


def _apply_orbital_rotation_adjacent(
    mat: np.ndarray,
    vec: np.ndarray,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    out: np.ndarray | None = None,
):
    """Apply an orbital rotation to adjacent orbitals.

    Args:
        mat: A 2x2 unitary matrix describing the orbital rotation.
        vec: Vector to be transformed.
        target_orbs: The orbitals to transform.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
    """
    if out is vec:
        raise ValueError("Output buffer cannot be the same as the input")
    if out is None:
        out = np.empty_like(vec)
    i, j = target_orbs
    assert i == j + 1 or i == j - 1, "Target orbitals must be adjacent."
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    vec = vec.reshape((dim_a, dim_b))
    out = out.reshape((dim_a, dim_b))
    indices = _zero_one_subspace_indices(norb, n_alpha, target_orbs)
    slice1 = indices[: len(indices) // 2]
    slice2 = indices[len(indices) // 2 :]
    buf = np.empty_like(vec)
    apply_matrix_to_slices(mat, vec, [slice1, slice2], out=buf)
    indices = _zero_one_subspace_indices(norb, n_beta, target_orbs)
    slice1 = indices[: len(indices) // 2]
    slice2 = indices[len(indices) // 2 :]
    apply_matrix_to_slices(mat, buf, [(Ellipsis, slice1), (Ellipsis, slice2)], out=out)
    return out.reshape(-1)


@lru_cache(maxsize=None)
def _zero_one_subspace_indices(norb: int, nocc: int, target_orbs: tuple[int, int]):
    """Return the indices where the target orbitals are 01 or 10."""
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n00 = comb(norb - 2, nocc, exact=True)
    n11 = comb(norb - 2, nocc - 2, exact=True)
    return indices[n00 : len(indices) - n11]


def apply_phase_shift(
    phase: complex,
    vec: np.ndarray,
    target_orbs: tuple[tuple[int], tuple[int]],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    """Apply a phase shift controlled on target orbitals.

    The phase is applied to all strings in which the target orbitals are
    all 1 (occupied).
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
def _zero_subspace_indices(norb: int, nocc: int, target_orbs: tuple[int]):
    """Return the indices where the target orbitals are 0."""
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n0 = comb(norb - len(target_orbs), nocc, exact=True)
    return indices[:n0]


@lru_cache(maxsize=None)
def _one_subspace_indices(norb: int, nocc: int, target_orbs: tuple[int]):
    """Return the indices where the target orbitals are 1."""
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n0 = comb(norb, nocc, exact=True) - comb(
        norb - len(target_orbs), nocc - len(target_orbs), exact=True
    )
    return indices[n0:]


def _shifted_orbitals(norb: int, target_orbs: tuple[int]):
    """Return orbital list with targeted orbitals shifted to the end."""
    orbitals = np.arange(norb - len(target_orbs))
    values = sorted(zip(target_orbs, range(norb - len(target_orbs), norb)))
    for index, val in values:
        orbitals = np.insert(orbitals, index, val)
    return orbitals


def apply_num_op_sum_evolution(
    coeffs: np.ndarray,
    vec: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    orbital_rotation: np.ndarray | None = None,
    *,
    copy: bool = True,
):
    """Apply time evolution by a linear combination of number operators."""
    if copy:
        vec = vec.copy()

    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
        np.uint, copy=False
    )
    occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
        np.uint, copy=False
    )

    if orbital_rotation is not None:
        vec, perm0 = apply_orbital_rotation(
            orbital_rotation.T.conj(),
            vec,
            norb,
            nelec,
            allow_row_permutation=True,
            copy=False,
        )
        coeffs = coeffs @ perm0.T

    phases = np.exp(-1j * time * coeffs)
    vec = vec.reshape((dim_a, dim_b))
    # apply alpha
    apply_num_op_sum_evolution_in_place(phases, vec, occupations=occupations_a)
    # apply beta
    vec = vec.T
    apply_num_op_sum_evolution_in_place(phases, vec, occupations=occupations_b)
    vec = vec.T.reshape(-1)

    if orbital_rotation is not None:
        vec, perm1 = apply_orbital_rotation(
            orbital_rotation, vec, norb, nelec, allow_col_permutation=True, copy=False
        )
        np.testing.assert_allclose(perm0, perm1.T)

    return vec


def apply_diag_coulomb_evolution(
    mat: np.ndarray,
    vec: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    orbital_rotation: np.ndarray | None = None,
    *,
    mat_alpha_beta: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    r"""Apply time evolution by a diagonal Coulomb operator.

    Applies

    .. math::
        \exp(-i t \sum_{i, j, \sigma, \tau} Z_{ij} n_{i, \sigma} n_{j, \tau} / 2)

    where :math:`n_{i, \sigma}` denotes the number operator on orbital :math:`i`
    and spin :math:`\sigma`, and :math:`Z` is the matrix input as ``mat``.
    If ``mat_alpha_beta`` is also given, then it is used in place of :math:`Z`
    for the terms in the sum where the spins differ (:math:`\sigma \neq \tau`).
    """
    if copy:
        vec = vec.copy()
    if mat_alpha_beta is None:
        mat_alpha_beta = mat

    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
        np.uint, copy=False
    )
    occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
        np.uint, copy=False
    )

    if orbital_rotation is not None:
        vec, perm0 = apply_orbital_rotation(
            orbital_rotation.T.conj(),
            vec,
            norb,
            nelec,
            allow_row_permutation=True,
            copy=False,
        )
        mat = perm0 @ mat @ perm0.T
        mat_alpha_beta = perm0 @ mat_alpha_beta @ perm0.T

    mat_exp = mat.copy()
    mat_exp[np.diag_indices(norb)] *= 0.5
    mat_exp = np.exp(-1j * time * mat_exp)
    mat_alpha_beta_exp = np.exp(-1j * time * mat_alpha_beta)
    vec = vec.reshape((dim_a, dim_b))
    apply_diag_coulomb_evolution_in_place(
        mat_exp,
        vec,
        norb=norb,
        mat_alpha_beta_exp=mat_alpha_beta_exp,
        occupations_a=occupations_a,
        occupations_b=occupations_b,
    )
    vec = vec.reshape(-1)

    if orbital_rotation is not None:
        vec, perm1 = apply_orbital_rotation(
            orbital_rotation, vec, norb, nelec, allow_col_permutation=True, copy=False
        )
        np.testing.assert_allclose(perm0, perm1.T)

    return vec


def apply_givens_rotation(
    theta: float,
    vec: np.ndarray,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
):
    r"""Apply a Givens rotation gate.

    The Givens rotation gate is

    .. math::
        G(\theta) = \exp(\theta (a^\dagger_i a_j - a\dagger_j a_i))

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbs: The orbitals (i, j) to rotate.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(norb)
    mat[np.ix_(target_orbs, target_orbs)] = [[c, s], [-s, c]]
    return apply_orbital_rotation(mat, vec, norb=norb, nelec=nelec)


def apply_tunneling_interaction(
    theta: float,
    vec: np.ndarray,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
):
    r"""Apply a tunneling interaction gate.

    The tunneling interaction gate is

    .. math::
        T(\theta) = \exp(i \theta (a^\dagger_i a_j + a\dagger_j a_i))

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbs: The orbitals (i, j) on which to apply the interaction.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    vec = apply_num_interaction(
        -np.pi / 2,
        vec,
        target_orbs[0],
        norb=norb,
        nelec=nelec,
    )
    vec = apply_givens_rotation(theta, vec, target_orbs, norb=norb, nelec=nelec)
    vec = apply_num_interaction(
        np.pi / 2,
        vec,
        target_orbs[0],
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec


def apply_num_interaction(
    theta: float,
    vec: np.ndarray,
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
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orb: The orbital on which to apply the interaction.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
        copy: Whether to copy the input vector. If False, the operation is applied
            in-place.
    """
    if copy:
        vec = vec.copy()
    for sigma in range(2):
        vec = apply_num_op_prod_interaction(
            theta,
            vec,
            ((target_orb, sigma),),
            norb=norb,
            nelec=nelec,
            copy=False,
        )
    return vec


def apply_num_op_prod_interaction(
    theta: float,
    vec: np.ndarray,
    target_orbs: tuple[tuple[int, bool], ...],
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
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbs: The orbitals on which to apply the interaction. This should
            be a tuple of (orbital, spin) pairs.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
        copy: Whether to copy the input vector. If False, the operation is applied
            in-place.
    """
    if copy:
        vec = vec.copy()
    orbitals = [set() for _ in range(2)]
    for i, spin_i in target_orbs:
        orbitals[spin_i].add(i)
    vec = apply_phase_shift(
        np.exp(1j * theta),
        vec,
        tuple(tuple(orbs) for orbs in orbitals),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec
