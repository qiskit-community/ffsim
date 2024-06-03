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

import math
from typing import overload

import numpy as np

from ffsim._lib import (
    apply_diag_coulomb_evolution_in_place_num_rep,
    apply_diag_coulomb_evolution_in_place_z_rep,
)
from ffsim.cistring import gen_occslst, make_strings
from ffsim.gates.orbital_rotation import apply_orbital_rotation


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


@overload
def apply_diag_coulomb_evolution(
    vec: np.ndarray,
    mat: np.ndarray,
    time: float,
    norb: int,
    nelec: int,
    *,
    orbital_rotation: np.ndarray | None = None,
    z_representation: bool = False,
    copy: bool = True,
) -> np.ndarray: ...
@overload
def apply_diag_coulomb_evolution(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
    nelec: tuple[int, int],
    *,
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    z_representation: bool = False,
    copy: bool = True,
) -> np.ndarray: ...
def apply_diag_coulomb_evolution(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
    nelec: int | tuple[int, int],
    *,
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    z_representation: bool = False,
    copy: bool = True,
) -> np.ndarray:
    r"""Apply time evolution by a (rotated) diagonal Coulomb operator.

    Applies

    .. math::

        \mathcal{U}
        \exp\left(-i t \sum_{\sigma, \tau, i, j}
        Z^{(\sigma \tau)}_{ij} n_{\sigma, i} n_{\tau, j} / 2\right)
        \mathcal{U}^\dagger

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, :math:`Z^{(\sigma \tau)}` is a real-valued matrix,
    and :math:`\mathcal{U}` is an optional orbital rotation.

    Args:
        vec: The state vector to be transformed.
        mat: The diagonal Coulomb matrix :math:`Z`.
            You can pass either a single Numpy array specifying the coefficients
            to use for all spin interactions, or you can pass a tuple of three Numpy
            arrays specifying independent coefficients for alpha-alpha, alpha-beta,
            and beta-beta interactions (in that order). If passing a tuple, you can
            set a tuple element to ``None`` to indicate the absence of interactions
            of that type. The alpha-alpha and beta-beta matrices are assumed to be
            symmetric, and only their upper triangular entries are used.
        time: The evolution time.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        orbital_rotation: The optional orbital rotation.
            You can pass either a single Numpy array specifying the orbital rotation
            to apply to both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent orbital rotations for spin alpha and spin beta.
            If passing a pair, you can use ``None`` for one of the
            values in the pair to indicate that no operation should be applied to that
            spin sector.
        z_representation: Whether the input matrices are in the "Z" representation.
        copy: Whether to copy the vector before operating on it.

            - If `copy=True` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If `copy=False` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.

    Returns:
        The evolved state vector.
    """
    if copy:
        vec = vec.copy()
    if isinstance(nelec, int):
        if z_representation:
            raise NotImplementedError
        return _apply_diag_coulomb_evolution_spinful(
            vec=vec,
            mat=mat,
            time=time,
            norb=norb,
            nelec=(nelec, 0),
            orbital_rotation=orbital_rotation,
            z_representation=False,
        )
    return _apply_diag_coulomb_evolution_spinful(
        vec=vec,
        mat=mat,
        time=time,
        norb=norb,
        nelec=nelec,
        orbital_rotation=orbital_rotation,
        z_representation=z_representation,
    )


def _apply_diag_coulomb_evolution_spinful(
    vec: np.ndarray,
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
    nelec: tuple[int, int],
    orbital_rotation: np.ndarray | tuple[np.ndarray | None, np.ndarray | None] | None,
    z_representation: bool,
) -> np.ndarray:
    mat_exp_aa, mat_exp_ab, mat_exp_bb = _get_mat_exp(
        mat, time, norb=norb, z_representation=z_representation
    )
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)

    if orbital_rotation is not None:
        vec = apply_orbital_rotation(
            vec,
            _conjugate_orbital_rotation(orbital_rotation),
            norb,
            nelec,
            copy=False,
        )

    vec = vec.reshape((dim_a, dim_b))
    if z_representation:
        strings_a = make_strings(range(norb), n_alpha)
        strings_b = make_strings(range(norb), n_beta)
        apply_diag_coulomb_evolution_in_place_z_rep(
            vec,
            mat_exp_aa,
            mat_exp_ab,
            mat_exp_bb,
            mat_exp_aa.conj(),
            mat_exp_ab.conj(),
            mat_exp_bb.conj(),
            norb=norb,
            strings_a=strings_a,
            strings_b=strings_b,
        )
    else:
        occupations_a = gen_occslst(range(norb), n_alpha)
        occupations_b = gen_occslst(range(norb), n_beta)
        apply_diag_coulomb_evolution_in_place_num_rep(
            vec,
            mat_exp_aa,
            mat_exp_ab,
            mat_exp_bb,
            norb=norb,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
    vec = vec.reshape(-1)

    if orbital_rotation is not None:
        vec = apply_orbital_rotation(
            vec,
            orbital_rotation,
            norb,
            nelec,
            copy=False,
        )

    return vec


def _get_mat_exp(
    mat: np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    time: float,
    norb: int,
    z_representation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat_aa: np.ndarray | None
    mat_ab: np.ndarray | None
    mat_bb: np.ndarray | None
    if isinstance(mat, np.ndarray) and mat.ndim == 2:
        mat_aa, mat_ab = mat.copy(), mat.copy()
        mat_aa[np.diag_indices(norb)] *= 0.5
        if z_representation:
            mat_aa *= 0.25
            mat_ab *= 0.25
        mat_exp_aa = np.exp(-1j * time * mat_aa)
        mat_exp_ab = np.exp(-1j * time * mat_ab)
        return mat_exp_aa, mat_exp_ab, mat_exp_aa
    else:
        mat_aa, mat_ab, mat_bb = mat
        if mat_aa is None:
            mat_exp_aa = np.ones((norb, norb), dtype=complex)
        else:
            mat_aa = mat_aa.copy()
            mat_aa[np.diag_indices(norb)] *= 0.5
            if z_representation:
                mat_aa *= 0.25
            mat_exp_aa = np.exp(-1j * time * mat_aa)
        if mat_bb is None:
            mat_exp_bb = np.ones((norb, norb), dtype=complex)
        else:
            mat_bb = mat_bb.copy()
            mat_bb[np.diag_indices(norb)] *= 0.5
            if z_representation:
                mat_bb *= 0.25
            mat_exp_bb = np.exp(-1j * time * mat_bb)
        if mat_ab is None:
            mat_exp_ab = np.ones((norb, norb), dtype=complex)
        else:
            if z_representation:
                mat_ab = mat_ab.copy()
                mat_ab *= 0.25
            mat_exp_ab = np.exp(-1j * time * mat_ab)
        return mat_exp_aa, mat_exp_ab, mat_exp_bb
