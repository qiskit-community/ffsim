# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contract diagonal Coulomb operator."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
from scipy.special import comb

from ffsim._lib import (
    contract_diag_coulomb_into_buffer_num_rep,
    contract_diag_coulomb_into_buffer_z_rep,
)
from ffsim.cistring import gen_occslst, make_strings
from ffsim.gates.orbital_rotation import apply_orbital_rotation
from ffsim.states import dim


def contract_diag_coulomb(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray | None = None,
    z_representation: bool = False,
) -> np.ndarray:
    r"""Contract a diagonal Coulomb operator with a vector.

    A diagonal Coulomb operator has the form

    .. math::

        \sum_{i, j, \sigma, \tau} Z_{ij} n_{\sigma, i} n_{\tau, j} / 2

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma` and :math:`Z` is a real symmetric matrix
    If ``mat_alpha_beta`` is also given, then it is used in place of :math:`Z`
    for the terms in the sum where the spins differ (:math:`\sigma \neq \tau`).

    Args:
        vec: The state vector to be transformed.
        mat: The real symmetric matrix :math:`Z`.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        mat_alpha_beta: A matrix of coefficients to use for interactions between
            orbitals with differing spin.
        z_representation: Whether the input matrices are in the "Z" representation.

    Returns:
        The result of applying the diagonal Coulomb operator on the input state vector.
    """
    vec = vec.astype(complex, copy=False)
    if mat_alpha_beta is None:
        mat_alpha_beta = mat

    if z_representation:
        return _contract_diag_coulomb_z_rep(
            vec,
            mat,
            norb=norb,
            nelec=nelec,
            mat_alpha_beta=mat_alpha_beta,
        )

    return _contract_diag_coulomb_num_rep(
        vec,
        mat,
        norb=norb,
        nelec=nelec,
        mat_alpha_beta=mat_alpha_beta,
    )


def _contract_diag_coulomb_num_rep(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray,
) -> np.ndarray:
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)

    occupations_a = gen_occslst(range(norb), n_alpha)
    occupations_b = gen_occslst(range(norb), n_beta)

    mat = mat.copy()
    mat[np.diag_indices(norb)] *= 0.5
    vec = vec.reshape((dim_a, dim_b))
    out = np.zeros_like(vec)
    contract_diag_coulomb_into_buffer_num_rep(
        vec,
        mat,
        norb=norb,
        mat_alpha_beta=mat_alpha_beta,
        occupations_a=occupations_a,
        occupations_b=occupations_b,
        out=out,
    )

    return out.reshape(-1)


def _contract_diag_coulomb_z_rep(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray,
) -> np.ndarray:
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)

    strings_a = make_strings(range(norb), n_alpha)
    strings_b = make_strings(range(norb), n_beta)

    vec = vec.reshape((dim_a, dim_b))
    out = np.zeros_like(vec)
    contract_diag_coulomb_into_buffer_z_rep(
        vec,
        mat,
        norb=norb,
        mat_alpha_beta=mat_alpha_beta,
        strings_a=strings_a,
        strings_b=strings_b,
        out=out,
    )

    return out.reshape(-1)


def diag_coulomb_linop(
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    orbital_rotation: np.ndarray | None = None,
    mat_alpha_beta: np.ndarray | None = None,
    z_representation: bool = False,
) -> scipy.sparse.linalg.LinearOperator:
    r"""Convert a (rotated) diagonal Coulomb matrix to a linear operator.

    A rotated diagonal Coulomb operator has the form

    .. math::

        \mathcal{U}
        (\sum_{i, j, \sigma, \tau} Z_{ij} n_{\sigma, i} n_{\tau, j} / 2)
        \mathcal{U}^\dagger

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, :math:`Z` is a real symmetric matrix,
    and :math:`\mathcal{U}` is an optional orbital rotation.
    If ``mat_alpha_beta`` is also given, then it is used in place of :math:`Z`
    for the terms in the sum where the spins differ (:math:`\sigma \neq \tau`).

    Args:
        mat: The real symmetric matrix :math:`Z`.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        orbital_rotation: A unitary matrix describing the optional orbital rotation.
        mat_alpha_beta: A matrix of coefficients to use for interactions between
            orbitals with differing spin.
        z_representation: Whether the input matrices are in the "Z" representation.

    Returns:
        A LinearOperator that implements the action of the diagonal Coulomb operator.
    """
    if mat_alpha_beta is None:
        mat_alpha_beta = mat
    dim_ = dim(norb, nelec)

    def matvec(vec):
        this_mat = mat
        this_mat_alpha_beta = mat_alpha_beta
        if orbital_rotation is not None:
            vec, perm0 = apply_orbital_rotation(
                vec,
                orbital_rotation.T.conj(),
                norb,
                nelec,
                allow_row_permutation=True,
            )
            this_mat = perm0 @ mat @ perm0.T
            this_mat_alpha_beta = perm0 @ mat_alpha_beta @ perm0.T
        vec = contract_diag_coulomb(
            vec,
            this_mat,
            norb=norb,
            nelec=nelec,
            mat_alpha_beta=this_mat_alpha_beta,
            z_representation=z_representation,
        )
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

    return scipy.sparse.linalg.LinearOperator(
        (dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
    )
