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

import numpy as np
import scipy.sparse.linalg
from pyscf.fci import cistring
from scipy.special import comb

from ffsim._ffsim import (
    contract_num_op_sum_spin_into_buffer,
)
from ffsim.gates.orbital_rotation import (
    apply_orbital_rotation,
    gen_orbital_rotation_index,
)
from ffsim.states import dim


def contract_num_op_sum(
    vec: np.ndarray,
    coeffs: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    occupations_a: np.ndarray | None = None,
    occupations_b: np.ndarray | None = None,
):
    r"""Contract a linear combination of number operators with a vector.

    A linear combination of number operators has the form

    .. math::

        \sum_{i, \sigma} \lambda_i n_{i, \sigma}

    where :math:`n_{i, \sigma}` denotes the number operator on orbital :math:`i`
    with spin :math:`\sigma` and the :math:`\lambda_i` are real numbers.

    Args:
        vec: The state vector to be transformed.
        coeffs: The coefficients of the linear combination.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        occupations_a: List of occupied orbital lists for alpha strings.
        occupations_b: List of occupied orbital lists for beta strings.

    Returns:
        The result of applying the linear combination of number operators on the input
        state vector.
    """
    vec = vec.astype(complex, copy=False)
    n_alpha, n_beta = nelec

    if occupations_a is None:
        occupations_a = cistring.gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
    if occupations_b is None:
        occupations_b = cistring.gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )

    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
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
        (\sum_{i, \sigma} \lambda_i n_{i, \sigma})
        \mathcal{U}^\dagger

    where :math:`n_{i, \sigma}` denotes the number operator on orbital :math:`i`
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
    n_alpha, n_beta = nelec
    dim_ = dim(norb, nelec)
    occupations_a = cistring.gen_occslst(range(norb), n_alpha).astype(
        np.uint, copy=False
    )
    occupations_b = cistring.gen_occslst(range(norb), n_beta).astype(
        np.uint, copy=False
    )
    orbital_rotation_index_a = gen_orbital_rotation_index(norb, n_alpha)
    orbital_rotation_index_b = gen_orbital_rotation_index(norb, n_beta)

    def matvec(vec):
        these_coeffs = coeffs
        if orbital_rotation is not None:
            vec, perm0 = apply_orbital_rotation(
                vec,
                orbital_rotation.T.conj(),
                norb,
                nelec,
                allow_row_permutation=True,
                orbital_rotation_index_a=orbital_rotation_index_a,
                orbital_rotation_index_b=orbital_rotation_index_b,
            )
            these_coeffs = perm0 @ these_coeffs
        vec = contract_num_op_sum(
            vec,
            these_coeffs,
            norb=norb,
            nelec=nelec,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )
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

    return scipy.sparse.linalg.LinearOperator(
        (dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
    )
