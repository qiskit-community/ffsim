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

import numpy as np
import scipy.sparse.linalg
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.direct_nosym import absorb_h1e, contract_1e, make_hdiag
from scipy.special import comb

from ffsim._ffsim import (
    contract_diag_coulomb_into_buffer_num_rep,
    contract_diag_coulomb_into_buffer_z_rep,
    contract_num_op_sum_spin_into_buffer,
)
from ffsim.gates import apply_orbital_rotation
from ffsim.gates.orbital_rotation import gen_orbital_rotation_index


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


def contract_diag_coulomb(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray | None = None,
    z_representation: bool = False,
    occupations_a: np.ndarray | None = None,
    occupations_b: np.ndarray | None = None,
    strings_a: np.ndarray | None = None,
    strings_b: np.ndarray | None = None,
) -> np.ndarray:
    """Contract a diagonal Coulomb operator with a vector."""
    if mat_alpha_beta is None:
        mat_alpha_beta = mat

    n_alpha, n_beta = nelec

    if z_representation:
        if strings_a is None:
            strings_a = cistring.make_strings(range(norb), n_alpha)
        if strings_b is None:
            strings_b = cistring.make_strings(range(norb), n_beta)
        return _contract_diag_coulomb_z_rep(
            vec,
            mat,
            norb=norb,
            nelec=nelec,
            mat_alpha_beta=mat_alpha_beta,
            strings_a=strings_a,
            strings_b=strings_b,
        )

    if occupations_a is None:
        occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
    if occupations_b is None:
        occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )
    return _contract_diag_coulomb_num_rep(
        vec,
        mat,
        norb=norb,
        nelec=nelec,
        mat_alpha_beta=mat_alpha_beta,
        occupations_a=occupations_a,
        occupations_b=occupations_b,
    )


def _contract_diag_coulomb_num_rep(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
) -> np.ndarray:
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)

    mat = mat.copy()
    mat[np.diag_indices(norb)] *= 0.5
    vec = vec.reshape((dim_a, dim_b))
    out = np.zeros_like(vec)
    contract_diag_coulomb_into_buffer_num_rep(
        vec,
        mat,
        norb=norb,
        mat_alpha_beta=mat_alpha_beta,
        occupations_a=occupations_a,
        occupations_b=occupations_b,
        out=out,
    )

    return out.reshape(-1)


def _contract_diag_coulomb_z_rep(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta: np.ndarray,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
) -> np.ndarray:
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)

    mat = mat.copy()
    mat[np.diag_indices(norb)] *= 0.5
    vec = vec.reshape((dim_a, dim_b))
    out = np.zeros_like(vec)
    contract_diag_coulomb_into_buffer_z_rep(
        vec,
        mat,
        norb=norb,
        mat_alpha_beta=mat_alpha_beta,
        strings_a=strings_a,
        strings_b=strings_b,
        out=out,
    )

    return out.reshape(-1)


def diag_coulomb_to_linop(
    mat: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    orbital_rotation: np.ndarray | None = None,
    *,
    mat_alpha_beta: np.ndarray | None = None,
    z_representation: bool = False,
) -> scipy.sparse.linalg.LinearOperator:
    """Convert a diagonal Coulomb matrix to a linear operator."""
    if mat_alpha_beta is None:
        mat_alpha_beta = mat
    n_alpha, n_beta = nelec
    dim = get_dimension(norb, nelec)

    occupations_a = None
    occupations_b = None
    strings_a = None
    strings_b = None
    if z_representation:
        strings_a = cistring.make_strings(range(norb), n_alpha)
        strings_b = cistring.make_strings(range(norb), n_beta)
    else:
        occupations_a = cistring._gen_occslst(range(norb), n_alpha).astype(
            np.uint, copy=False
        )
        occupations_b = cistring._gen_occslst(range(norb), n_beta).astype(
            np.uint, copy=False
        )
    orbital_rotation_index_a = gen_orbital_rotation_index(norb, n_alpha)
    orbital_rotation_index_b = gen_orbital_rotation_index(norb, n_beta)

    def matvec(vec):
        this_mat = mat
        this_mat_alpha_beta = mat_alpha_beta
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
            this_mat = perm0 @ mat @ perm0.T
            this_mat_alpha_beta = perm0 @ mat_alpha_beta @ perm0.T
        vec = contract_diag_coulomb(
            vec,
            this_mat,
            norb=norb,
            nelec=nelec,
            mat_alpha_beta=this_mat_alpha_beta,
            z_representation=z_representation,
            occupations_a=occupations_a,
            occupations_b=occupations_b,
            strings_a=strings_a,
            strings_b=strings_b,
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
        (dim, dim), matvec=matvec, rmatvec=matvec, dtype=complex
    )
