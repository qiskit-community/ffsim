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
from scipy.special import comb

from ffsim._lib import (
    apply_diag_coulomb_evolution_in_place_num_rep,
    apply_diag_coulomb_evolution_in_place_z_rep,
)
from ffsim.cistring import gen_occslst, make_strings
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
    copy: bool = True,
) -> np.ndarray:
    r"""Apply time evolution by a (rotated) diagonal Coulomb operator.

    Applies

    .. math::

        \mathcal{U}
        \exp\left(-i t \sum_{\sigma, \tau, i, j}
        Z_{ij} n_{\sigma, i} n_{\tau, j} / 2\right)
        \mathcal{U}^\dagger

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, :math:`Z` is a real symmetric matrix,
    and :math:`\mathcal{U}` is an optional orbital rotation.
    If `mat_alpha_beta` is also given, then it is used in place of :math:`Z`
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
        z_representation: Whether the input matrices are in the "Z" representation.
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
        strings_a = make_strings(range(norb), n_alpha)
        strings_b = make_strings(range(norb), n_beta)
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
        occupations_a = gen_occslst(range(norb), n_alpha)
        occupations_b = gen_occslst(range(norb), n_beta)
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
            copy=False,
        )
        np.testing.assert_allclose(perm0, perm1.T)

    return vec
