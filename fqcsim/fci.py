# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import itertools
from functools import lru_cache

import numpy as np
import scipy.sparse.linalg
from pyscf import fci, lib
from pyscf.fci import cistring
from pyscf.fci.direct_spin1 import make_hdiag
from pyscf.fci.fci_slow import absorb_h1e
from scipy.special import comb
from fqcsim.states import one_hot


def contract_1e(f1e, fcivec, norb, nelec):
    # source: pyscf/fci/fci_slow.py
    # modified to support complex dtypes
    # TODO contribute dtype modification back to pyscf
    # TODO use cached link_indexa and link_indexb
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
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
    fcinew = np.dot(f1e.reshape(-1), t1.reshape(-1, na * nb))
    return fcinew.reshape(fcivec.shape)


def contract_2e(eri, fcivec, norb, nelec):
    # source: pyscf/fci/fci_slow.py
    # modified to support complex dtypes
    # TODO contribute dtype modification back to pyscf
    # TODO use cached link_indexa and link_indexb
    """Compute E_{pq}E_{rs}|CI>"""
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
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


def get_dimension(n_orbitals: int, n_electrons: tuple[int, int]) -> int:
    """Get the dimension of the FCI space."""
    n_alpha, n_beta = n_electrons
    dim_a = comb(n_orbitals, n_alpha, exact=True)
    dim_b = comb(n_orbitals, n_beta, exact=True)
    return dim_a * dim_b


def get_hamiltonian_linop(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    n_electrons: tuple[int, int],
) -> scipy.sparse.linalg.LinearOperator:
    """Get the Hamiltonian in the FCI basis."""
    # TODO use cached link_indexa and link_indexb
    # TODO support complex one-body tensor
    n_orbitals, _ = one_body_tensor.shape
    two_body = absorb_h1e(
        one_body_tensor, two_body_tensor, n_orbitals, n_electrons, 0.5
    )
    dim = get_dimension(n_orbitals, n_electrons)

    def matvec(vec: np.ndarray):
        return contract_2e(two_body, vec, n_orbitals, n_electrons)

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=matvec, rmatvec=matvec
    )


def get_trace(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    n_electrons: tuple[int, int],
) -> float:
    """Get the trace of the Hamiltonian in the FCI basis."""
    n_orbitals, _ = one_body_tensor.shape
    hdiag = make_hdiag(one_body_tensor, two_body_tensor, n_orbitals, n_electrons)
    return np.sum(hdiag)


@lru_cache(maxsize=None)
def generate_num_map(n_orbitals: int, n_electrons: tuple[int, int]):
    """Tabulates n_p |x) for all orbitals p and determinants |x)"""
    dim = get_dimension(n_orbitals, n_electrons)
    n_alpha, n_beta = n_electrons
    num = np.zeros((n_orbitals, 2, dim, dim))
    for i in range(dim):
        vec = one_hot(dim, i)
        for p in range(n_orbitals):
            tmp = fci.addons.des_a(vec, n_orbitals, (n_alpha, n_beta), p)
            num[p, 0, :, i] = fci.addons.cre_a(
                tmp, n_orbitals, (n_alpha - 1, n_beta), p
            ).ravel()
            tmp = fci.addons.des_b(vec, n_orbitals, (n_alpha, n_beta), p)
            num[p, 1, :, i] = fci.addons.cre_b(
                tmp, n_orbitals, (n_alpha, n_beta - 1), p
            ).ravel()
    return num


def one_body_tensor_to_linop(
    one_body_tensor: np.ndarray, n_electrons: tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Convert the one-body tensor to a matrix in the FCI basis."""
    # TODO use cached link_indexa and link_indexb
    n_orbitals, _ = one_body_tensor.shape
    dim = get_dimension(n_orbitals, n_electrons)

    def matvec(vec: np.ndarray):
        return contract_1e(one_body_tensor, vec, n_orbitals, n_electrons)

    def rmatvec(vec: np.ndarray):
        return contract_1e(one_body_tensor.T.conj(), vec, n_orbitals, n_electrons)

    return scipy.sparse.linalg.LinearOperator(
        shape=(dim, dim), matvec=matvec, rmatvec=rmatvec
    )


def contract_num_op_sum(
    coeffs: np.ndarray,
    vec: np.ndarray,
    n_electrons: tuple[int, int],
    *,
    num_map: np.ndarray | None = None,
):
    """Contract a sum of number operators with a vector."""
    n_orbitals = len(coeffs)
    if num_map is None:
        num_map = generate_num_map(n_orbitals, n_electrons)
    result = np.zeros_like(vec)
    for p in range(n_orbitals):
        for sigma in range(2):
            result += coeffs[p] * num_map[p, sigma] @ vec
    return result


def num_op_sum_to_linop(
    coeffs: np.ndarray, n_electrons: tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Convert a sum of number operators to a linear operator."""
    n_orbitals = len(coeffs)
    dim = get_dimension(n_orbitals, n_electrons)
    num_map = generate_num_map(n_orbitals, n_electrons)

    def matvec(vec):
        return contract_num_op_sum(coeffs, vec, n_electrons, num_map=num_map)

    return scipy.sparse.linalg.LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)


def contract_core_tensor(
    core_tensor: np.ndarray,
    vec: np.ndarray,
    n_electrons: tuple[int, int],
    *,
    core_tensor_alpha_beta: np.ndarray | None = None,
    num_map: np.ndarray | None = None,
):
    """Contract a core tensor with a vector."""
    n_orbitals, _ = core_tensor.shape
    if core_tensor_alpha_beta is None:
        core_tensor_alpha_beta = core_tensor
    if num_map is None:
        num_map = generate_num_map(n_orbitals, n_electrons)
    result = np.zeros_like(vec)
    for p, q in itertools.combinations_with_replacement(range(n_orbitals), 2):
        coeff = 0.5 if p == q else 1.0
        for sigma in range(2):
            result += (
                coeff
                * core_tensor[p, q]
                * (num_map[p, sigma] @ (num_map[q, sigma] @ vec))
            )
            result += (
                coeff
                * core_tensor_alpha_beta[p, q]
                * (num_map[p, sigma] @ (num_map[q, 1 - sigma] @ vec))
            )
    return result


def core_tensor_to_linop(
    core_tensor: np.ndarray,
    n_electrons: tuple[int, int],
    *,
    core_tensor_alpha_beta: np.ndarray | None = None,
) -> scipy.sparse.linalg.LinearOperator:
    """Convert a core tensor to a linear operator."""
    n_orbitals, _ = core_tensor.shape
    dim = get_dimension(n_orbitals, n_electrons)
    num_map = generate_num_map(n_orbitals, n_electrons)

    def matvec(vec):
        return contract_core_tensor(
            core_tensor,
            vec,
            n_electrons,
            core_tensor_alpha_beta=core_tensor_alpha_beta,
            num_map=num_map,
        )

    return scipy.sparse.linalg.LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)
