# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contract diagonal Coulomb operator."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
from pyscf.fci import cistring
from scipy.special import comb

from ffsim._ffsim import (
    contract_diag_coulomb_into_buffer_num_rep,
    contract_diag_coulomb_into_buffer_z_rep,
)
from ffsim.contract.hamiltonian import get_dimension
from ffsim.gates import apply_orbital_rotation
from ffsim.gates.orbital_rotation import gen_orbital_rotation_index


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
    vec = vec.astype(complex, copy=False)
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
