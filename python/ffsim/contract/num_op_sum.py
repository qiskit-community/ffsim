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
from ffsim.contract.hamiltonian import get_dimension


def contract_num_op_sum(
    vec: np.ndarray,
    coeffs: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    occupations_a: np.ndarray | None = None,
    occupations_b: np.ndarray | None = None,
):
    """Contract a sum of number operators with a vector."""
    vec = vec.astype(complex, copy=False)
    n_alpha, n_beta = nelec

    if occupations_a is None:
        occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
    if occupations_b is None:
        occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
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
    coeffs: np.ndarray, norb: int, nelec: tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Convert a sum of number operators to a linear operator."""
    n_alpha, n_beta = nelec
    dim = get_dimension(norb, nelec)
    occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
        np.uint, copy=False
    )
    occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
        np.uint, copy=False
    )

    def matvec(vec):
        return contract_num_op_sum(
            vec,
            coeffs,
            norb=norb,
            nelec=nelec,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )

    return scipy.sparse.linalg.LinearOperator(
        (dim, dim), matvec=matvec, rmatvec=matvec, dtype=complex
    )
