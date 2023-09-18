# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Time evolution by linear combination of number operators."""

from __future__ import annotations

import numpy as np
from pyscf.fci import cistring
from scipy.special import comb

from ffsim._ffsim import apply_num_op_sum_evolution_in_place
from ffsim.gates.orbital_rotation import apply_orbital_rotation


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
    r"""Apply time evolution by a (rotated) linear combination of number operators.

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

    Raises:
        ValueError: ``coeffs`` must be a one-dimensional vector with length ``norb``.
    """
    if coeffs.shape != (norb,):
        raise ValueError(
            "coeffs must be a one-dimensional vector with length norb. "
            f"Got norb = {norb} but coeffs had shape {coeffs.shape}"
        )

    if copy:
        vec = vec.copy()

    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    if occupations_a is None:
        occupations_a = cistring.gen_occslst(range(norb), n_alpha)
    if occupations_b is None:
        occupations_b = cistring.gen_occslst(range(norb), n_beta)
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
