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
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
import scipy.sparse.linalg

# HACK: Sphinx fails to handle "ellipsis"
# See https://github.com/python/typing/issues/684
if TYPE_CHECKING:
    _SliceAtom = Union[int, slice, np.ndarray, "ellipsis"]
else:
    _SliceAtom = Union[int, slice, np.ndarray, type(Ellipsis)]

_Slice = Union[_SliceAtom, Tuple[_SliceAtom, ...]]


# TODO use scipy.sparse.linalg.expm_multiply instead after dropping Python 3.7 support
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


def givens_matrix(a: complex, b: complex) -> np.ndarray:
    r"""Compute the Givens rotation to zero out a row entry.

    Returns a :math:`2 \times 2` unitary matrix G that satisfies

    .. math::
        G
        \begin{pmatrix}
            a \\
            b
        \end{pmatrix}
        =
        \begin{pmatrix}
            r \\
            0
        \end{pmatrix}

    where :math:`r` is a complex number.

    References:
        - `<https://en.wikipedia.org/wiki/Givens_rotation#Stable_calculation>`_
        - `<https://www.netlib.org/lapack/lawnspdf/lawn148.pdf>`_

    Args:
        a: A complex number representing the first row entry
        b: A complex number representing the second row entry

    Returns:
        The Givens rotation matrix.
    """
    # Handle case that a is zero
    if np.isclose(a, 0.0):
        cosine = 0.0
        sine = 1.0
    # Handle case that b is zero and a is nonzero
    elif np.isclose(b, 0.0):
        cosine = 1.0
        sine = 0.0
    # Handle case that a and b are both nonzero
    else:
        hypotenuse = np.hypot(abs(a), abs(b))
        cosine = abs(a) / hypotenuse
        sign_a = a / abs(a)
        sine = sign_a * b.conjugate() / hypotenuse

    return np.array([[cosine, sine], [-sine.conjugate(), cosine]])


def givens_decomposition(
    mat: np.ndarray,
) -> tuple[list[tuple[np.ndarray, tuple[int, int]]], np.ndarray]:
    """Givens rotation decomposition of a unitary matrix."""
    n, _ = mat.shape
    current_matrix = mat
    left_rotations = []
    right_rotations: list[tuple[np.ndarray, tuple[int, int]]] = []

    # compute left and right Givens rotations
    for i in range(n - 1):
        if i % 2 == 0:
            # rotate columns by right multiplication
            for j in range(i + 1):
                target_index = i - j
                row = n - j - 1
                if not np.isclose(current_matrix[row, target_index], 0.0):
                    # zero out element at target index in given row
                    givens_mat = givens_matrix(
                        current_matrix[row, target_index + 1],
                        current_matrix[row, target_index],
                    )
                    right_rotations.append(
                        (givens_mat, (target_index + 1, target_index))
                    )
                    current_matrix = apply_matrix_to_slices(
                        givens_mat,
                        current_matrix,
                        [(Ellipsis, target_index + 1), (Ellipsis, target_index)],
                    )
        else:
            # rotate rows by left multiplication
            for j in range(i + 1):
                target_index = n - i + j - 1
                col = j
                if not np.isclose(current_matrix[target_index, col], 0.0):
                    # zero out element at target index in given column
                    givens_mat = givens_matrix(
                        current_matrix[target_index - 1, col],
                        current_matrix[target_index, col],
                    )
                    left_rotations.append(
                        (givens_mat, (target_index - 1, target_index))
                    )
                    current_matrix = apply_matrix_to_slices(
                        givens_mat, current_matrix, [target_index - 1, target_index]
                    )

    # convert left rotations to right rotations
    for givens_mat, (i, j) in reversed(left_rotations):
        givens_mat = givens_mat.T.conj().astype(mat.dtype, copy=False)
        givens_mat[:, 0] *= current_matrix[i, i]
        givens_mat[:, 1] *= current_matrix[j, j]
        new_givens_mat = givens_matrix(givens_mat[1, 1], givens_mat[1, 0])
        right_rotations.append((new_givens_mat.T, (i, j)))
        phase_matrix = givens_mat @ new_givens_mat
        current_matrix[i, i] = phase_matrix[0, 0]
        current_matrix[j, j] = phase_matrix[1, 1]

    # return decomposition
    return right_rotations, np.diagonal(current_matrix)


def apply_matrix_to_slices(
    mat: np.ndarray,
    target: np.ndarray,
    slices: Sequence[_Slice],
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a matrix to slices of a target tensor.

    Args:
        mat: The matrix to apply to slices of the target tensor.
        target: The tensor containing the slices on which to apply the matrix.
        slices: The slices of the target tensor on which to apply the matrix.

    Returns:
        The resulting tensor.
    """
    if out is target:
        raise ValueError("Output buffer cannot be the same as the input")
    if out is None:
        out = np.empty_like(target)
    out[...] = target[...]
    for i, slice_i in enumerate(slices):
        out[slice_i] *= mat[i, i]
        for j, slice_j in enumerate(slices):
            if j != i:
                out[slice_i] += mat[i, j] * target[slice_j]
    return out


def modified_cholesky(
    mat: np.ndarray, *, error_threshold: float = 1e-8, max_vecs: int | None = None
) -> np.ndarray:
    r"""Modified Cholesky decomposition.

    The modified Cholesky decomposition of a square matrix :math:`M` has the form

    .. math::
        M = \sum_{i=1}^N v_i v_i^\dagger

    where each :math:`v_i` is a vector. `M` must be positive definite.
    No checking is performed to verify whether `M` is positive definite.
    The number of terms :math:`N` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the `max_vecs` parameter specifies an optional upper bound
    on :math:`N`. The `max_vecs` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    References:
        - `arXiv:1711.02242`_

    Args:
        mat: The matrix to decompose.
        error_threshold: Threshold for allowed error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: The maximum number of vectors to include in the decomposition.

    Returns:
        The Cholesky vectors v_i assembled into a 2-dimensional Numpy array
        whose columns are the vectors.

    .. _arXiv:1711.02242: https://arxiv.org/abs/1711.02242
    """
    dim, _ = mat.shape

    if max_vecs is None:
        max_vecs = dim

    cholesky_vecs = np.zeros((dim, max_vecs + 1), dtype=mat.dtype)
    errors = np.real(np.diagonal(mat).copy())
    for index in range(max_vecs + 1):
        max_error_index = np.argmax(errors)
        max_error = errors[max_error_index]
        if max_error < error_threshold:
            break
        cholesky_vecs[:, index] = mat[:, max_error_index]
        if index:
            cholesky_vecs[:, index] -= (
                cholesky_vecs[:, 0:index]
                @ cholesky_vecs[max_error_index, 0:index].conj()
            )
        cholesky_vecs[:, index] /= np.sqrt(max_error)
        errors -= np.abs(cholesky_vecs[:, index]) ** 2

    return cholesky_vecs[:, :index]


def double_factorized(
    two_body_tensor: np.ndarray,
    *,
    error_threshold: float = 1e-8,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Double-factorized decomposition of a two-body tensor.

    The double-factorized decomposition is a representation of a two-body tensor
    :math:`h_{pqrs}` as

    .. math::
        h_{pqrs} = \sum_{t=1}^N \sum_{k\ell} U^{t}_{pk} U^{t}_{qk}
            Z^{t}_{k\ell} U^{t}_{r\ell} U^{t}_{s\ell}

    Here each :math:`Z^{(t)}` is a real symmetric matrix, referred to as a
    "diagonal Coulomb matrix," and each :math:`U^{t}` is a unitary matrix, referred to
    as an "orbital rotation."

    The number of terms :math:`N` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the ``max_vecs`` parameter specifies an optional upper bound
    on :math:`N`. The ``max_vecs`` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    Args:
        two_body_tensor: The two-body tensor to decompose.
        error_threshold: Threshold for allowed error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: An optional limit on the number of terms to keep in the decomposition
            of the two-body tensor.

    Returns:
        The diagonal Coulomb matrices and the orbital rotations. Each list of matrices
        is collected into a numpy array, so this method returns a tuple of two numpy
        arrays, the first containing the diagonal Coulomb matrices and the second
        containing the orbital rotations. Each numpy array will have shape (t, n, n)
        where t is the rank of the decomposition and n is the number of orbitals.

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    """
    n_modes, _, _, _ = two_body_tensor.shape

    if max_vecs is None:
        max_vecs = n_modes * (n_modes + 1) // 2

    reshaped_tensor = np.reshape(two_body_tensor, (n_modes**2, n_modes**2))
    cholesky_vecs = modified_cholesky(
        reshaped_tensor, error_threshold=error_threshold, max_vecs=max_vecs
    )

    _, rank = cholesky_vecs.shape
    diag_coulomb_mats = np.zeros((rank, n_modes, n_modes), dtype=two_body_tensor.dtype)
    orbital_rotations = np.zeros((rank, n_modes, n_modes), dtype=two_body_tensor.dtype)
    for i in range(rank):
        mat = np.reshape(cholesky_vecs[:, i], (n_modes, n_modes))
        eigs, vecs = np.linalg.eigh(mat)
        diag_coulomb_mats[i] = np.outer(eigs, eigs.conj())
        orbital_rotations[i] = vecs

    return diag_coulomb_mats, orbital_rotations


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


def is_real_symmetric(
    mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
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


def is_unitary(mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
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


def is_orthogonal(mat: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
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
) -> bool:
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
