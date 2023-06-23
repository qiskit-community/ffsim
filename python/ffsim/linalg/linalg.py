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
