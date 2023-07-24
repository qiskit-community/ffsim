# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Time evolution by diagonal Coulomb operator."""

from __future__ import annotations

from typing import cast

import numpy as np
from pyscf.fci import cistring
from scipy.special import comb

from ffsim._ffsim import (
    apply_diag_coulomb_evolution_in_place_num_rep,
    apply_diag_coulomb_evolution_in_place_z_rep,
)
from ffsim.gates.orbital_rotation import apply_orbital_rotation


def apply_diag_coulomb_evolution(
    vec: np.ndarray,
    mat: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    orbital_rotation: np.ndarray | None = None,
    *,
    mat_alpha_beta: np.ndarray | None = None,
    z_representation: bool = False,
    occupations_a: np.ndarray | None = None,
    occupations_b: np.ndarray | None = None,
    strings_a: np.ndarray | None = None,
    strings_b: np.ndarray | None = None,
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

    Returns:
        The evolved state vector.
    """
    if copy:
        vec = vec.copy()
    if mat_alpha_beta is None:
        mat_alpha_beta = mat

    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)

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
    mat_alpha_beta_exp = cast(np.ndarray, mat_alpha_beta).copy()
    mat_exp[np.diag_indices(norb)] *= 0.5
    if z_representation:
        mat_exp *= 0.25
        mat_alpha_beta_exp *= 0.25
    mat_exp = np.exp(-1j * time * mat_exp)
    mat_alpha_beta_exp = np.exp(-1j * time * cast(np.ndarray, mat_alpha_beta_exp))
    vec = vec.reshape((dim_a, dim_b))

    if z_representation:
        if strings_a is None:
            strings_a = cistring.make_strings(range(norb), n_alpha)
        if strings_b is None:
            strings_b = cistring.make_strings(range(norb), n_beta)
        apply_diag_coulomb_evolution_in_place_z_rep(
            vec,
            mat_exp,
            mat_exp.conj(),
            norb=norb,
            mat_alpha_beta_exp=mat_alpha_beta_exp,
            mat_alpha_beta_exp_conj=mat_alpha_beta_exp.conj(),
            strings_a=strings_a,
            strings_b=strings_b,
        )
    else:
        if occupations_a is None:
            occupations_a = cistring.gen_occslst(range(norb), n_alpha)
        if occupations_b is None:
            occupations_b = cistring.gen_occslst(range(norb), n_beta)
        occupations_a = occupations_a.astype(np.uint, copy=False)
        occupations_b = occupations_b.astype(np.uint, copy=False)
        apply_diag_coulomb_evolution_in_place_num_rep(
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
