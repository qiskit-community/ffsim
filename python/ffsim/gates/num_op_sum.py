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
from scipy.special import comb

from ffsim._lib import apply_num_op_sum_evolution_in_place
from ffsim.cistring import gen_occslst
from ffsim.gates.orbital_rotation import apply_orbital_rotation
from ffsim.spin import Spin


def apply_num_op_sum_evolution(
    vec: np.ndarray,
    coeffs: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    spin: Spin = Spin.ALPHA_AND_BETA,
    *,
    orbital_rotation: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    r"""Apply time evolution by a (rotated) linear combination of number operators.

    Applies

    .. math::

        \mathcal{U}
        \exp\left(-i t \sum_{\sigma, i} \lambda_i n_{\sigma, i}\right)
        \mathcal{U}^\dagger

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, the :math:`\lambda_i` are real numbers, and
    :math:`\mathcal{U}` is an optional orbital rotation.

    Args:
        vec: The state vector to be transformed.
        coeffs: The coefficients of the linear combination.
        time: The evolution time.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        spin: Choice of spin sector(s) to act on.

            - To act on only spin alpha, pass :const:`ffsim.Spin.ALPHA`.
            - To act on only spin beta, pass :const:`ffsim.Spin.BETA`.
            - To act on both spin alpha and spin beta, pass
              :const:`ffsim.Spin.ALPHA_AND_BETA` (this is the default value).
        orbital_rotation: A unitary matrix describing the optional orbital rotation.
        copy: Whether to copy the vector before operating on it.

            - If `copy=True` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If `copy=False` then this function may still return a newly allocated
              vector, but the original vector may have its data overwritten.
              It is also possible that the original vector is returned,
              modified in-place.

    Returns:
        The evolved state vector.

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
    occupations_a = gen_occslst(range(norb), n_alpha)
    occupations_b = gen_occslst(range(norb), n_beta)

    if orbital_rotation is not None:
        vec, perm0 = apply_orbital_rotation(
            vec,
            orbital_rotation.T.conj(),
            norb,
            nelec,
            spin=spin,
            allow_row_permutation=True,
            copy=False,
        )
        coeffs = perm0 @ coeffs

    phases = np.exp(-1j * time * coeffs)
    vec = vec.reshape((dim_a, dim_b))

    if spin & Spin.ALPHA:
        # apply alpha
        apply_num_op_sum_evolution_in_place(vec, phases, occupations=occupations_a)
    if spin & Spin.BETA:
        # apply beta
        vec = vec.T
        apply_num_op_sum_evolution_in_place(vec, phases, occupations=occupations_b)
        vec = vec.T
    vec = vec.reshape(-1)

    if orbital_rotation is not None:
        vec, perm1 = apply_orbital_rotation(
            vec,
            orbital_rotation,
            norb,
            nelec,
            spin=spin,
            allow_col_permutation=True,
            copy=False,
        )
        np.testing.assert_allclose(perm1.T, perm0)

    return vec
