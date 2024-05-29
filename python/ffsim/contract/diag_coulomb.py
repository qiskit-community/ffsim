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

import math

import numpy as np
import scipy.sparse.linalg

from ffsim._lib import (
    contract_diag_coulomb_into_buffer_num_rep,
    contract_diag_coulomb_into_buffer_z_rep,
)
from ffsim.cistring import gen_occslst, make_strings
from ffsim.gates.orbital_rotation import apply_orbital_rotation
from ffsim.states import dim


def _conjugate_orbital_rotation(
    orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None],
) -> np.ndarray | tuple[np.ndarray | None, np.ndarray | None]:
    if isinstance(orbital_rotation, np.ndarray) and orbital_rotation.ndim == 2:
        return orbital_rotation.T.conj()
    orbital_rotation_a, orbital_rotation_b = orbital_rotation
    if orbital_rotation_a is not None:
        orbital_rotation_a = orbital_rotation_a.T.conj()
    if orbital_rotation_b is not None:
        orbital_rotation_b = orbital_rotation_b.T.conj()
    return (orbital_rotation_a, orbital_rotation_b)


def contract_diag_coulomb(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    norb: int,
    nelec: tuple[int, int],
    *,
    z_representation: bool = False,
) -> np.ndarray:
    r"""Contract a diagonal Coulomb operator with a vector.

    A diagonal Coulomb operator has the form

    .. math::

        \sum_{i, j, \sigma, \tau} Z^{(\sigma \tau)}_{ij} n_{\sigma, i} n_{\tau, j} / 2

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma` and :math:`Z^{(\sigma \tau)}` is a real-valued matrix

    Args:
        vec: The state vector to be transformed.
        mat: The diagonal Coulomb matrix :math:`Z`.
            You can pass either a single Numpy array specifying the coefficients
            to use for all spin interactions, or you can pass a tuple of three Numpy
            arrays specifying independent coefficients for alpha-alpha, alpha-beta, and
            beta-beta interactions (in that order). If passing a tuple, you can set a
            tuple element to ``None`` to indicate the absence of interactions of that
            type.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        z_representation: Whether the input matrices are in the "Z" representation.

    Returns:
        The result of applying the diagonal Coulomb operator on the input state vector.
    """
    mat_aa, mat_ab, mat_bb = _get_mats(
        mat, norb=norb, z_representation=z_representation
    )
    vec = vec.astype(complex, copy=False)

    if z_representation:
        return _contract_diag_coulomb_z_rep(
            vec,
            mat_aa,
            mat_ab,
            mat_bb,
            norb=norb,
            nelec=nelec,
        )

    return _contract_diag_coulomb_num_rep(
        vec,
        mat_aa,
        mat_ab,
        mat_bb,
        norb=norb,
        nelec=nelec,
    )


def _get_mats(
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    norb: int,
    z_representation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat_aa: np.ndarray | None
    mat_ab: np.ndarray | None
    mat_bb: np.ndarray | None
    if isinstance(mat, np.ndarray):
        mat_aa, mat_ab = mat, mat
        if not z_representation:
            mat_aa = mat_aa.copy()
            mat_aa[np.diag_indices(norb)] *= 0.5
        return mat_aa, mat_ab, mat_aa
    else:
        mat_aa, mat_ab, mat_bb = mat
        if mat_aa is None:
            mat_aa = np.zeros((norb, norb))
        elif not z_representation:
            mat_aa = mat_aa.copy()
            mat_aa[np.diag_indices(norb)] *= 0.5
        if mat_bb is None:
            mat_bb = np.zeros((norb, norb))
        elif not z_representation:
            mat_bb = mat_bb.copy()
            mat_bb[np.diag_indices(norb)] *= 0.5
        if mat_ab is None:
            mat_ab = np.zeros((norb, norb))
        return mat_aa, mat_ab, mat_bb


def _contract_diag_coulomb_num_rep(
    vec: np.ndarray,
    mat_aa: np.ndarray,
    mat_ab: np.ndarray,
    mat_bb: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)

    occupations_a = gen_occslst(range(norb), n_alpha)
    occupations_b = gen_occslst(range(norb), n_beta)

    vec = vec.reshape((dim_a, dim_b))
    out = np.zeros_like(vec)
    contract_diag_coulomb_into_buffer_num_rep(
        vec,
        mat_aa,
        mat_ab,
        mat_bb,
        norb=norb,
        occupations_a=occupations_a,
        occupations_b=occupations_b,
        out=out,
    )

    return out.reshape(-1)


def _contract_diag_coulomb_z_rep(
    vec: np.ndarray,
    mat_aa: np.ndarray,
    mat_ab: np.ndarray,
    mat_bb: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)

    strings_a = make_strings(range(norb), n_alpha)
    strings_b = make_strings(range(norb), n_beta)

    vec = vec.reshape((dim_a, dim_b))
    out = np.zeros_like(vec)
    contract_diag_coulomb_into_buffer_z_rep(
        vec,
        mat_aa,
        mat_ab,
        mat_bb,
        norb=norb,
        strings_a=strings_a,
        strings_b=strings_b,
        out=out,
    )

    return out.reshape(-1)


def diag_coulomb_linop(
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    norb: int,
    nelec: tuple[int, int],
    *,
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    z_representation: bool = False,
) -> scipy.sparse.linalg.LinearOperator:
    r"""Convert a (rotated) diagonal Coulomb matrix to a linear operator.

    A rotated diagonal Coulomb operator has the form

    .. math::

        \mathcal{U}
        (\sum_{i, j, \sigma, \tau} Z^{(\sigma \tau)}_{ij} n_{\sigma, i} n_{\tau, j} / 2)
        \mathcal{U}^\dagger

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, :math:`Z^{(\sigma \tau)}` is a real-valued matrix,
    and :math:`\mathcal{U}` is an optional orbital rotation.

    Args:
        mat: The diagonal Coulomb matrix :math:`Z`.
            You can pass either a single Numpy array specifying the coefficients
            to use for all spin interactions, or you can pass a tuple of three Numpy
            arrays specifying independent coefficients for alpha-alpha, alpha-beta, and
            beta-beta interactions (in that order). If passing a tuple, you can set a
            tuple element to ``None`` to indicate the absence of interactions of that
            type.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        orbital_rotation: The optional orbital rotation.
            You can pass either a single Numpy array specifying the orbital rotation
            to apply to both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent orbital rotations for spin alpha and spin beta.
            If passing a pair, you can use ``None`` for one of the
            values in the pair to indicate that no operation should be applied to that
            spin sector.
        mat_alpha_beta: A matrix of coefficients to use for interactions between
            orbitals with differing spin.
        z_representation: Whether the input matrices are in the "Z" representation.

    Returns:
        A LinearOperator that implements the action of the diagonal Coulomb operator.
    """
    dim_ = dim(norb, nelec)

    def matvec(vec):
        if orbital_rotation is not None:
            vec = apply_orbital_rotation(
                vec,
                _conjugate_orbital_rotation(orbital_rotation),
                norb,
                nelec,
            )
        vec = contract_diag_coulomb(
            vec,
            mat,
            norb=norb,
            nelec=nelec,
            z_representation=z_representation,
        )
        if orbital_rotation is not None:
            vec = apply_orbital_rotation(
                vec,
                orbital_rotation,
                norb,
                nelec,
                copy=False,
            )
        return vec

    return scipy.sparse.linalg.LinearOperator(
        (dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
    )
