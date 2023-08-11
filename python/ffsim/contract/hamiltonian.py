# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contract a molecular Hamiltonian."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
from pyscf.fci import cistring
from pyscf.fci.direct_nosym import absorb_h1e, contract_1e, make_hdiag
from pyscf.fci.fci_slow import contract_2e

from ffsim.states import dim


def hamiltonian_trace(
    *,
    norb: int,
    nelec: tuple[int, int],
    one_body_tensor: np.ndarray | None = None,
    two_body_tensor: np.ndarray | None = None,
    constant: float = 0.0,
) -> float:
    """Get the trace of the Hamiltonian in the FCI basis."""
    return constant * dim(norb, nelec) + np.sum(
        make_hdiag(one_body_tensor, two_body_tensor, norb, nelec)
    )


def hamiltonian_linop(
    *,
    norb: int,
    nelec: tuple[int, int],
    one_body_tensor: np.ndarray | None = None,
    two_body_tensor: np.ndarray | None = None,
    constant: float = 0.0,
) -> scipy.sparse.linalg.LinearOperator:
    r"""Convert a molecular Hamiltonian to a linear operator.

    A molecular Hamiltonian has the form

    .. math::

        H = \sum_{pq, \sigma} h_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
            + \frac12 \sum_{pqrs, \sigma} h_{pqrs, \sigma\tau}
            a^\dagger_{p, \sigma} a^\dagger_{r, \tau} a_{s, \tau} a_{q, \sigma}.

    Here :math:`h_{pq}` is called the one-body tensor and :math:`h_{pqrs}` is called
    the two-body tensor.

    Args:
        one_body_tensor: The one-body tensor.
        two_body_tensor: The two-body tensor.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        constant: The constant.

    Returns:
        A LinearOperator that implements the action of the Hamiltonian.
    """
    if one_body_tensor is None:
        one_body_tensor = np.zeros((norb, norb))
    if two_body_tensor is None:
        return _one_body_tensor_linop(
            one_body_tensor, norb=norb, nelec=nelec, constant=constant
        )

    n_alpha, n_beta = nelec
    linkstr_index_a = cistring.gen_linkstr_index(range(norb), n_alpha)
    linkstr_index_b = cistring.gen_linkstr_index(range(norb), n_beta)
    link_index = (linkstr_index_a, linkstr_index_b)
    two_body = absorb_h1e(one_body_tensor, two_body_tensor, norb, nelec, 0.5)
    dim_ = dim(norb, nelec)

    def matvec(vec: np.ndarray):
        return constant * vec + contract_2e(
            two_body, vec, norb, nelec, link_index=link_index
        )

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
    )


def _one_body_tensor_linop(
    mat: np.ndarray, norb: int, nelec: tuple[int, int], constant: float = 0.0
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
    link_index = (linkstr_index_a, linkstr_index_b)

    def matvec(vec: np.ndarray):
        result = constant * vec.astype(complex)
        result += contract_1e(mat.real, vec.real, norb, nelec, link_index=link_index)
        result += 1j * contract_1e(
            mat.imag, vec.real, norb, nelec, link_index=link_index
        )
        result += 1j * contract_1e(
            mat.real, vec.imag, norb, nelec, link_index=link_index
        )
        result -= contract_1e(mat.imag, vec.imag, norb, nelec, link_index=link_index)
        return result

    def rmatvec(vec: np.ndarray):
        one_body_tensor_H = mat.T.conj()
        result = contract_1e(
            one_body_tensor_H.real, vec.real, norb, nelec, link_index=link_index
        ).astype(complex)
        result += 1j * contract_1e(
            one_body_tensor_H.imag, vec.real, norb, nelec, link_index=link_index
        )
        result += 1j * contract_1e(
            one_body_tensor_H.real, vec.imag, norb, nelec, link_index=link_index
        )
        result -= contract_1e(
            one_body_tensor_H.imag, vec.imag, norb, nelec, link_index=link_index
        )
        return result

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=rmatvec, dtype=complex
    )
