# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear algebra predicates."""

from typing import Union

import numpy as np

_bool = Union[bool, np.bool_]


def is_hermitian(mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Determine if a matrix is approximately Hermitian.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is Hermitian within the given tolerance.
    """
    m, n = mat.shape
    return m == n and np.allclose(mat, mat.T.conj(), rtol=rtol, atol=atol)


def is_antihermitian(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Determine if a matrix is approximately anti-Hermitian.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is anti-Hermitian within the given tolerance.
    """
    m, n = mat.shape
    return m == n and np.allclose(mat, -mat.T.conj(), rtol=rtol, atol=atol)


def is_real_symmetric(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> _bool:
    """Determine if a matrix is real and approximately symmetric.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is real and symmetric within the given tolerance.
    """
    m, n = mat.shape
    return (
        m == n
        and np.all(np.isreal(mat))
        and np.allclose(mat, mat.T, rtol=rtol, atol=atol)
    )


def is_unitary(mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> _bool:
    """Determine if a matrix is approximately unitary.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is unitary within the given tolerance.
    """
    m, n = mat.shape
    return m == n and np.allclose(mat @ mat.T.conj(), np.eye(m), rtol=rtol, atol=atol)


def is_orthogonal(mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> _bool:
    """Determine if a matrix is approximately orthogonal.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is orthogonal within the given tolerance.
    """
    m, n = mat.shape
    return (
        m == n
        and np.all(np.isreal(mat))
        and np.allclose(mat @ mat.T, np.eye(m), rtol=rtol, atol=atol)
    )


def is_special_orthogonal(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> _bool:
    """Determine if a matrix is approximately special orthogonal.

    Args:
        mat: The matrix.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        Whether the matrix is special orthogonal within the given tolerance.
    """
    m, n = mat.shape
    return (
        m == n
        and np.all(np.isreal(mat))
        and np.allclose(mat @ mat.T, np.eye(m), rtol=rtol, atol=atol)
        and (m == 0 or np.allclose(np.linalg.det(mat), 1, rtol=rtol, atol=atol))
    )
