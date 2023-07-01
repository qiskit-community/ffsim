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
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.direct_nosym import absorb_h1e, contract_1e, make_hdiag
from scipy.special import comb


def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    # source: pyscf
    # modified to accept cached link index
    """Compute E_{pq}E_{rs}|CI>"""
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na, nb)
    t1 = np.zeros((norb, norb, na, nb), dtype=fcivec.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a, i, str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a, i, :, str1] += sign * ci0[:, str0]

    t1 = lib.einsum("bjai,aiAB->bjAB", eri.reshape([norb] * 4), t1)

    fcinew = np.zeros_like(ci0, dtype=fcivec.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1[a, i, str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:, str1] += sign * t1[a, i, :, str0]
    return fcinew.reshape(fcivec.shape)


def get_dimension(norb: int, nelec: tuple[int, int]) -> int:
    """Get the dimension of the FCI space."""
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    return dim_a * dim_b


def get_trace(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> float:
    """Get the trace of the Hamiltonian in the FCI basis."""
    return np.sum(make_hdiag(one_body_tensor, two_body_tensor, norb, nelec))


def get_hamiltonian_linop(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """Get the Hamiltonian in the FCI basis."""
    n_alpha, n_beta = nelec
    linkstr_index_a = cistring.gen_linkstr_index(range(norb), n_alpha)
    linkstr_index_b = cistring.gen_linkstr_index(range(norb), n_beta)
    link_index = (linkstr_index_a, linkstr_index_b)
    two_body = absorb_h1e(one_body_tensor, two_body_tensor, norb, nelec, 0.5)
    dim = get_dimension(norb, nelec)

    def matvec(vec: np.ndarray):
        return contract_2e(two_body, vec, norb, nelec, link_index=link_index)

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=matvec, rmatvec=matvec, dtype=complex
    )


def one_body_tensor_to_linop(
    one_body_tensor: np.ndarray, norb: int, nelec: tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Convert the one-body tensor to a matrix in the FCI basis."""
    dim = get_dimension(norb, nelec)
    n_alpha, n_beta = nelec
    linkstr_index_a = cistring.gen_linkstr_index(range(norb), n_alpha)
    linkstr_index_b = cistring.gen_linkstr_index(range(norb), n_beta)
    link_index = (linkstr_index_a, linkstr_index_b)

    def matvec(vec: np.ndarray):
        result = contract_1e(
            one_body_tensor.real, vec.real, norb, nelec, link_index=link_index
        ).astype(complex)
        result += 1j * contract_1e(
            one_body_tensor.imag, vec.real, norb, nelec, link_index=link_index
        )
        result += 1j * contract_1e(
            one_body_tensor.real, vec.imag, norb, nelec, link_index=link_index
        )
        result -= contract_1e(
            one_body_tensor.imag, vec.imag, norb, nelec, link_index=link_index
        )
        return result

    def rmatvec(vec: np.ndarray):
        one_body_tensor_H = one_body_tensor.T.conj()
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
        shape=(dim, dim), matvec=matvec, rmatvec=rmatvec, dtype=complex
    )
