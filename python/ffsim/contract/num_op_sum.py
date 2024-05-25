# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contract linear combination of number operators."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse.linalg

from ffsim._lib import (
    contract_num_op_sum_spin_into_buffer,
)
from ffsim.cistring import gen_occslst
from ffsim.gates.orbital_rotation import apply_orbital_rotation
from ffsim.states import dim


def contract_num_op_sum(
    vec: np.ndarray, coeffs: np.ndarray, norb: int, nelec: tuple[int, int]
):
    r"""Contract a linear combination of number operators with a vector.

    A linear combination of number operators has the form

    .. math::

        \sum_{\sigma, i} \lambda_i n_{\sigma, i}

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma` and the :math:`\lambda_i` are real numbers.

    Args:
        vec: The state vector to be transformed.
        coeffs: The coefficients of the linear combination.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        The result of applying the linear combination of number operators on the input
        state vector.
    """
    vec = vec.astype(complex, copy=False)
    n_alpha, n_beta = nelec

    occupations_a = gen_occslst(range(norb), n_alpha)
    occupations_b = gen_occslst(range(norb), n_beta)

    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    vec = vec.reshape((dim_a, dim_b))
    out = np.zeros_like(vec)
    # apply alpha
    contract_num_op_sum_spin_into_buffer(
        vec, coeffs, occupations=occupations_a, out=out
    )
    # apply beta
    vec = vec.T
    out = out.T
    contract_num_op_sum_spin_into_buffer(
        vec, coeffs, occupations=occupations_b, out=out
    )

    return out.T.reshape(-1)


def num_op_sum_linop(
    coeffs: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    orbital_rotation: np.ndarray | None = None,
) -> scipy.sparse.linalg.LinearOperator:
    r"""Convert a (rotated) linear combination of number operators to a linear operator.

    A rotated linear combination of number operators has the form

    .. math::

        \mathcal{U}
        (\sum_{\sigma, i} \lambda_i n_{\sigma, i})
        \mathcal{U}^\dagger

    where :math:`n_{\sigma, i}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma`, the :math:`\lambda_i` are real numbers, and
    :math:`\mathcal{U}` is an optional orbital rotation.

    Args:
        coeffs: The coefficients of the linear combination.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        orbital_rotation: A unitary matrix describing the optional orbital rotation.

    Returns:
        A LinearOperator that implements the action of the linear combination of number
        operators.
    """
    dim_ = dim(norb, nelec)

    def matvec(vec):
        these_coeffs = coeffs
        if orbital_rotation is not None:
            vec = apply_orbital_rotation(
                vec,
                orbital_rotation.T.conj(),
                norb,
                nelec,
            )
        vec = contract_num_op_sum(vec, these_coeffs, norb=norb, nelec=nelec)
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
