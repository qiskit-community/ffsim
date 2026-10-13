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

import cmath
import math
from collections.abc import Sequence

import numpy as np
import scipy.linalg
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


def logm_unitary(mat: np.ndarray) -> np.ndarray:
    """Compute the principal matrix logarithm of a unitary matrix.

    The logarithm is computed from the Schur decomposition, which for a unitary
    matrix takes the form ``mat = vecs @ diag(eigs) @ vecs.conj().T``, where ``vecs``
    is unitary and the eigenvalues ``eigs`` lie on the unit circle. The logarithm is
    obtained by replacing each eigenvalue with the imaginary number given by its phase.

    This is faster than the general-purpose :func:`scipy.linalg.logm`, and unlike that
    function it returns an antihermitian matrix by construction.

    Args:
        mat: The unitary matrix.

    Returns:
        The antihermitian principal matrix logarithm of the unitary matrix.
    """
    schur_form, vecs = scipy.linalg.schur(mat, output="complex")
    return (vecs * 1j * np.angle(np.diag(schur_form))) @ vecs.conj().T


def logm_special_orthogonal(mat: np.ndarray) -> np.ndarray:
    """Compute a real antisymmetric matrix logarithm of a special orthogonal matrix.

    The logarithm is computed from the real Schur decomposition, which for a special
    orthogonal matrix is block diagonal, with 2x2 rotation blocks and scalar blocks
    equal to +1 or -1. Each rotation block is replaced by its rotation angle, which is
    taken in the interval (-pi, pi], and the -1 blocks are paired into rotations by pi.
    The eigenvalues of the result therefore have imaginary parts in [-pi, pi].

    Unlike ``logm_unitary(mat).real``, this function returns a valid logarithm when
    the matrix has -1 eigenvalues. :func:`logm_unitary` maps each -1 eigenvalue to the
    imaginary number i*pi, so the real part loses the corresponding rotation.

    The logarithm is not unique, and its entries can jump between branches as the
    input varies. This function does not check that the input is orthogonal.

    Args:
        mat: The special orthogonal matrix. A complex dtype is accepted as long as
            the imaginary part is zero.

    Returns:
        The real antisymmetric matrix logarithm of the special orthogonal matrix.

    Raises:
        ValueError: The matrix has a nonzero imaginary part.
        ValueError: The matrix has an odd number of -1 eigenvalues, so it is not
            special orthogonal.
    """
    if np.iscomplexobj(mat):
        if np.any(mat.imag):
            raise ValueError("The matrix has a nonzero imaginary part.")
        mat = mat.real
    schur_form, vecs = scipy.linalg.schur(mat, output="real")
    dim = schur_form.shape[0]
    log = np.zeros_like(schur_form)
    negative_indices = []
    index = 0
    while index < dim:
        # LAPACK marks each 2x2 block with a nonzero subdiagonal entry, and sets the
        # subdiagonal entry below each scalar block to exactly zero.
        if index + 1 < dim and schur_form[index + 1, index] != 0:
            # The block is approximately [[cos(t), -sin(t)], [sin(t), cos(t)]], whose
            # logarithm is [[0, -t], [t, 0]]. The angle computed here is -t.
            angle = math.atan2(
                schur_form[index, index + 1] - schur_form[index + 1, index],
                schur_form[index, index] + schur_form[index + 1, index + 1],
            )
            log[index, index + 1] = angle
            log[index + 1, index] = -angle
            index += 2
        else:
            # The block is a real eigenvalue, which is +1 or -1 up to rounding error.
            if schur_form[index, index] < 0:
                negative_indices.append(index)
            index += 1
    if len(negative_indices) % 2:
        raise ValueError(
            "The matrix has an odd number of -1 eigenvalues, so it is not special "
            "orthogonal."
        )
    # The -1 eigenvectors are orthonormal, so any two of them span a plane in which
    # the matrix acts as a rotation by pi.
    for first, second in zip(negative_indices[::2], negative_indices[1::2]):
        log[first, second] = math.pi
        log[second, first] = -math.pi
    log = vecs @ log @ vecs.T
    return (log - log.T) / 2


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


def reduced_matrix(
    mat: scipy.sparse.linalg.LinearOperator, vecs: Sequence[np.ndarray]
) -> np.ndarray:
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


def match_global_phase(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phase the given arrays so that their phases match at one entry.

    Args:
        a: A Numpy array.
        b: Another Numpy array.

    Returns:
        A pair of arrays (a', b') that are equal if and only if a == b * exp(i phi)
        for some real number phi.
    """
    if a.shape != b.shape:
        return a, b
    # use the largest entry of one of the matrices to maximize precision
    index = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    phase_a = cmath.phase(a[index])
    phase_b = cmath.phase(b[index])
    return a * cmath.rect(1, -phase_a), b * cmath.rect(1, -phase_b)


def one_hot(shape: int | tuple[int, ...], index, *, dtype=complex):
    """Return an array of all zeros except for a one at a specified index.

    Args:
        shape: The desired shape of the array.
        index: The index at which to place a one.

    Returns:
        The one-hot vector.
    """
    vec = np.zeros(shape, dtype=dtype)
    vec[index] = 1
    return vec
