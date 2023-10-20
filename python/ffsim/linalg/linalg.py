# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear algebra utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse.linalg


def expm_multiply_taylor(
    mat: scipy.sparse.linalg.LinearOperator, vec: np.ndarray, tol: float = 1e-12
) -> np.ndarray:
    """Compute expm(mat) @ vec using a Taylor series expansion."""
    result = vec.copy()
    term = vec
    denominator = 1
    while np.linalg.norm(term) > tol:
        term = mat @ term / denominator
        result += term
        denominator += 1
    return result


def lup(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-pivoted LU decomposition of a matrix.

    The decomposition is:

    .. math::

        A = L U P

    where L is a lower triangular matrix with unit diagonal elements,
    U is upper triangular, and P is a permutation matrix.
    """
    p, ell, u = scipy.linalg.lu(mat.T)
    d = np.diagonal(u)
    ell *= d
    u /= d[:, None]
    return u.T, ell.T, p.T


def reduced_matrix(mat: scipy.sparse.linalg.LinearOperator, vecs: Sequence[np.ndarray]):
    r"""Compute reduced matrix within a subspace spanned by some vectors.

    Given a linear operator :math:`A` and a list of vectors :math:`\{v_i\}`,
    return the matrix M where :math:`M_{ij} = v_i^\dagger A v_j`.
    """
    dim = len(vecs)
    result = np.zeros((dim, dim), dtype=complex)
    for j, state_j in enumerate(vecs):
        mat_state_j = mat @ state_j
        for i, state_i in enumerate(vecs):
            result[i, j] = np.vdot(state_i, mat_state_j)
    return result
