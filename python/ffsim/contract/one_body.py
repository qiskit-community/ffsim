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
from pyscf.fci.direct_nosym import contract_1e

from ffsim.cistring import gen_linkstr_index
from ffsim.states import dim


def contract_one_body(
    vec: np.ndarray, mat: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    r"""Contract a one-body tensor with a vector.

    A one-body tensor has the form

    .. math::

        \sum_{ij} M_{ij} a^\dagger_i a_j

    where :math:`M` is a complex-valued matrix.

    Args:
        mat: The one-body tensor.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        A LinearOperator that implements the action of the one-body tensor.
    """
    n_alpha, n_beta = nelec
    link_index_a = gen_linkstr_index(range(norb), n_alpha)
    link_index_b = gen_linkstr_index(range(norb), n_beta)
    link_index = (link_index_a, link_index_b)
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

        \sum_{ij} M_{ij} a^\dagger_i a_j

    where :math:`M` is a complex-valued matrix.

    Args:
        mat: The one-body tensor.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        A LinearOperator that implements the action of the one-body tensor.
    """
    dim_ = dim(norb, nelec)

    def matvec(vec: np.ndarray):
        return contract_one_body(vec, mat, norb=norb, nelec=nelec)

    def rmatvec(vec: np.ndarray):
        return contract_one_body(vec, mat.T.conj(), norb=norb, nelec=nelec)

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=rmatvec, dtype=complex
    )
