# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contract one-body operator."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
from pyscf.fci import cistring
from pyscf.fci.direct_nosym import contract_1e

from ffsim.states import dim


def contract_one_body(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    linkstr_index_a: np.ndarray,
    linkstr_index_b: np.ndarray,
) -> np.ndarray:
    link_index = (linkstr_index_a, linkstr_index_b)
    result = contract_1e(mat.real, vec.real, norb, nelec, link_index=link_index).astype(
        complex
    )
    result += 1j * contract_1e(mat.imag, vec.real, norb, nelec, link_index=link_index)
    result += 1j * contract_1e(mat.real, vec.imag, norb, nelec, link_index=link_index)
    result -= contract_1e(mat.imag, vec.imag, norb, nelec, link_index=link_index)
    return result


def one_body_linop(
    mat: np.ndarray, norb: int, nelec: tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    r"""Convert a one-body tensor to a linear operator.

    A one-body tensor has the form

    .. math::

        \sum_{ij} M_{ij} a^\dagger{i} a_j

    where :math:`M` is a complex-valued matrix.

    Args:
        mat: The one-body tensor.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        A LinearOperator that implements the action of the one-body tensor.
    """
    dim_ = dim(norb, nelec)
    n_alpha, n_beta = nelec
    linkstr_index_a = cistring.gen_linkstr_index(range(norb), n_alpha)
    linkstr_index_b = cistring.gen_linkstr_index(range(norb), n_beta)

    def matvec(vec: np.ndarray):
        return contract_one_body(
            vec,
            mat,
            norb=norb,
            nelec=nelec,
            linkstr_index_a=linkstr_index_a,
            linkstr_index_b=linkstr_index_b,
        )

    def rmatvec(vec: np.ndarray):
        return contract_one_body(
            vec,
            mat.T.conj(),
            norb=norb,
            nelec=nelec,
            linkstr_index_a=linkstr_index_a,
            linkstr_index_b=linkstr_index_b,
        )

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=rmatvec, dtype=complex
    )
