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
from pyscf.fci import cistring
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
    norb: int,
    nelec: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """Get the Hamiltonian in the FCI basis."""
    # TODO use cached link_indexa and link_indexb
    two_body = absorb_h1e(one_body_tensor, two_body_tensor, norb, nelec, 0.5)
    dim = get_dimension(norb, nelec)

    def matvec(vec: np.ndarray):
        return contract_2e(two_body, vec, norb, nelec)

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=matvec, rmatvec=matvec, dtype=one_body_tensor.dtype
    )


def get_trace(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> float:
    """Get the trace of the Hamiltonian in the FCI basis."""
    return np.sum(make_hdiag(one_body_tensor, two_body_tensor, norb, nelec))


def one_body_tensor_to_linop(
    one_body_tensor: np.ndarray, norb: int, nelec: tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Convert the one-body tensor to a matrix in the FCI basis."""
    # TODO use cached link_indexa and link_indexb
    dim = get_dimension(norb, nelec)

    def matvec(vec: np.ndarray):
        return contract_1e(one_body_tensor, vec, norb, nelec)

    def rmatvec(vec: np.ndarray):
        return contract_1e(one_body_tensor.T.conj(), vec, norb, nelec)

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=matvec, rmatvec=rmatvec, dtype=one_body_tensor.dtype
    )


def contract_num_op_sum(
    coeffs: np.ndarray,
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    occupations_a: np.ndarray | None = None,
    occupations_b: np.ndarray | None = None,
):
    """Contract a sum of number operators with a vector."""
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
    _contract_num_op_sum_spin(coeffs, vec, occupations=occupations_a, out=out)
    # apply beta
    vec = vec.T
    out = out.T
    _contract_num_op_sum_spin(coeffs, vec, occupations=occupations_b, out=out)

    return out.T.reshape(-1)


def _contract_num_op_sum_spin(
    coeffs: np.ndarray,
    vec: np.ndarray,
    occupations: np.ndarray,
    out: np.ndarray,
) -> None:
    for source_row, target_row, orbs in zip(vec, out, occupations):
        coeff = 0
        for orb in orbs:
            coeff += coeffs[orb]
        target_row += coeff * source_row


def num_op_sum_to_linop(
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
            coeffs,
            vec,
            norb=norb,
            nelec=nelec,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )

    return scipy.sparse.linalg.LinearOperator(
        (dim, dim), matvec=matvec, rmatvec=matvec, dtype=coeffs.dtype
    )


def contract_diag_coulomb(
    mat: np.ndarray,
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray | None = None,
    occupations_a: np.ndarray | None = None,
    occupations_b: np.ndarray | None = None,
) -> np.ndarray:
    """Contract a diagonal Coulomb operator with a vector."""
    if mat_alpha_beta is None:
        mat_alpha_beta = mat

    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    if occupations_a is None:
        occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
    if occupations_b is None:
        occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )

    mat = mat.copy()
    mat[np.diag_indices(norb)] *= 0.5
    vec = vec.reshape((dim_a, dim_b))
    out = np.zeros_like(vec)
    _contract_diag_coulomb(
        mat,
        vec,
        norb=norb,
        mat_alpha_beta=mat_alpha_beta,
        occupations_a=occupations_a,
        occupations_b=occupations_b,
        out=out,
    )

    return out.reshape(-1)


def _contract_diag_coulomb(
    mat: np.ndarray,
    vec: np.ndarray,
    norb: int,
    mat_alpha_beta: np.ndarray,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
    out: np.ndarray | None = None,
) -> None:
    dim_a, dim_b = vec.shape
    alpha_coeffs = np.empty((dim_a,), dtype=complex)
    beta_coeffs = np.empty((dim_b,), dtype=complex)
    coeff_map = np.zeros((dim_a, norb), dtype=complex)

    for i, occ in enumerate(occupations_a):
        coeff = 0
        for orb_1, orb_2 in itertools.combinations_with_replacement(occ, 2):
            coeff += mat[orb_1, orb_2]
        alpha_coeffs[i] = coeff

    for i, occ in enumerate(occupations_b):
        coeff = 0
        for orb_1, orb_2 in itertools.combinations_with_replacement(occ, 2):
            coeff += mat[orb_1, orb_2]
        beta_coeffs[i] = coeff

    for row, orbs in zip(coeff_map, occupations_a):
        for orb in orbs:
            row += mat_alpha_beta[orb]

    for source, target, alpha_coeff, coeff_map in zip(
        vec, out, alpha_coeffs, coeff_map
    ):
        for j, occ_b in enumerate(occupations_b):
            coeff = alpha_coeff + beta_coeffs[j]
            for orb_b in occ_b:
                coeff += coeff_map[orb_b]
            target[j] += coeff * source[j]


def diag_coulomb_to_linop(
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray | None = None,
) -> scipy.sparse.linalg.LinearOperator:
    """Convert a diagonal Coulomb matrix to a linear operator."""
    n_alpha, n_beta = nelec
    dim = get_dimension(norb, nelec)
    occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
        np.uint, copy=False
    )
    occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
        np.uint, copy=False
    )

    def matvec(vec):
        return contract_diag_coulomb(
            mat,
            vec,
            norb=norb,
            nelec=nelec,
            mat_alpha_beta=mat_alpha_beta,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
        )

    return scipy.sparse.linalg.LinearOperator(
        (dim, dim), matvec=matvec, rmatvec=matvec, dtype=mat.dtype
    )
