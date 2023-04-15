# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import scipy.sparse.linalg
from pyscf import fci
from pyscf.fci.direct_spin1 import make_hdiag
from pyscf.fci.fci_slow import absorb_h1e, contract_1e, contract_2e
from scipy.special import comb

from ffsim.states import one_hot


def get_dimension(norb: int, nelec: tuple[int, int]) -> int:
    """Get the dimension of the FCI space."""
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    return dim_a * dim_b


def get_hamiltonian_linop(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    nelec: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """Get the Hamiltonian in the FCI basis."""
    # TODO use cached link_indexa and link_indexb
    # TODO support complex one-body tensor
    norb, _ = one_body_tensor.shape
    two_body = absorb_h1e(one_body_tensor, two_body_tensor, norb, nelec, 0.5)
    dim = get_dimension(norb, nelec)

    def matvec(vec: np.ndarray):
        return contract_2e(two_body, vec, norb, nelec)

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=matvec, rmatvec=matvec
    )


def get_trace(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    nelec: tuple[int, int],
) -> float:
    """Get the trace of the Hamiltonian in the FCI basis."""
    norb, _ = one_body_tensor.shape
    hdiag = make_hdiag(one_body_tensor, two_body_tensor, norb, nelec)
    return np.sum(hdiag)


@lru_cache(maxsize=None)
def generate_num_map(norb: int, nelec: tuple[int, int]):
    """Tabulates n_p |x) for all orbitals p and determinants |x)"""
    dim = get_dimension(norb, nelec)
    n_alpha, n_beta = nelec
    num = np.zeros((norb, 2, dim, dim))
    for i in range(dim):
        vec = one_hot(dim, i)
        for p in range(norb):
            tmp = fci.addons.des_a(vec, norb, (n_alpha, n_beta), p)
            num[p, 0, :, i] = fci.addons.cre_a(
                tmp, norb, (n_alpha - 1, n_beta), p
            ).ravel()
            tmp = fci.addons.des_b(vec, norb, (n_alpha, n_beta), p)
            num[p, 1, :, i] = fci.addons.cre_b(
                tmp, norb, (n_alpha, n_beta - 1), p
            ).ravel()
    return num


def one_body_tensor_to_linop(
    one_body_tensor: np.ndarray, nelec: tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Convert the one-body tensor to a matrix in the FCI basis."""
    # TODO use cached link_indexa and link_indexb
    norb, _ = one_body_tensor.shape
    dim = get_dimension(norb, nelec)

    def matvec(vec: np.ndarray):
        return contract_1e(one_body_tensor, vec, norb, nelec)

    def rmatvec(vec: np.ndarray):
        return contract_1e(one_body_tensor.T.conj(), vec, norb, nelec)

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=matvec, rmatvec=rmatvec
    )


def contract_num_op_sum(
    coeffs: np.ndarray,
    vec: np.ndarray,
    nelec: tuple[int, int],
    *,
    num_map: np.ndarray | None = None,
):
    """Contract a sum of number operators with a vector."""
    norb = len(coeffs)
    if num_map is None:
        num_map = generate_num_map(norb, nelec)
    result = np.zeros_like(vec)
    for p in range(norb):
        for sigma in range(2):
            result += coeffs[p] * num_map[p, sigma] @ vec
    return result


def num_op_sum_to_linop(
    coeffs: np.ndarray, nelec: tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Convert a sum of number operators to a linear operator."""
    norb = len(coeffs)
    dim = get_dimension(norb, nelec)
    num_map = generate_num_map(norb, nelec)

    def matvec(vec):
        return contract_num_op_sum(coeffs, vec, nelec, num_map=num_map)

    return scipy.sparse.linalg.LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)


def contract_diag_coulomb(
    mat: np.ndarray,
    vec: np.ndarray,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray | None = None,
    num_map: np.ndarray | None = None,
):
    """Contract a diagonal Coulomb operator with a vector."""
    norb, _ = mat.shape
    if mat_alpha_beta is None:
        mat_alpha_beta = mat
    if num_map is None:
        num_map = generate_num_map(norb, nelec)
    result = np.zeros_like(vec)
    for p, q in itertools.combinations_with_replacement(range(norb), 2):
        coeff = 0.5 if p == q else 1.0
        for sigma in range(2):
            result += (
                coeff * mat[p, q] * (num_map[p, sigma] @ (num_map[q, sigma] @ vec))
            )
            result += (
                coeff
                * mat_alpha_beta[p, q]
                * (num_map[p, sigma] @ (num_map[q, 1 - sigma] @ vec))
            )
    return result


def diag_coulomb_to_linop(
    mat: np.ndarray,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray | None = None,
) -> scipy.sparse.linalg.LinearOperator:
    """Convert a core tensor to a linear operator."""
    norb, _ = mat.shape
    dim = get_dimension(norb, nelec)
    num_map = generate_num_map(norb, nelec)

    def matvec(vec):
        return contract_diag_coulomb(
            mat,
            vec,
            nelec,
            mat_alpha_beta=mat_alpha_beta,
            num_map=num_map,
        )

    return scipy.sparse.linalg.LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)
