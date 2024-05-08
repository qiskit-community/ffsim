# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for performing the Givens decomposition."""

from __future__ import annotations

import cmath
from typing import NamedTuple

import numpy as np
from scipy.linalg.blas import zrotg as zrotg_
from scipy.linalg.lapack import zrot


class GivensRotation(NamedTuple):
    r"""A Givens rotation.

    A Givens rotation acts on the two-dimensional subspace spanned by the :math:`i`-th
    and :math:`j`-th basis vectors as

    .. math::

        \begin{pmatrix}
            c & s \\
            -s^* & c \\
        \end{pmatrix}

    where :math:`c` is a real number and :math:`s` is a complex number.
    """

    c: float
    s: complex
    i: int
    j: int


def apply_matrix_to_slices(
    target: np.ndarray,
    mat: np.ndarray,
    slices,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a matrix to slices of a target tensor.

    Args:
        target: The tensor containing the slices on which to apply the matrix.
        mat: The matrix to apply to slices of the target tensor.
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


def zrotg(a: complex, b: complex, tol=1e-12) -> tuple[float, complex]:
    r"""Safe version of the zrotg BLAS function.

    The BLAS implementation of zrotg can return NaN values if either a or b is very
    close to zero. This function detects if either a or b is close to zero up to the
    specified tolerance, in which case it behaves as if it were exactly zero.

    Note that in contrast to `scipy.linalg.blas.zrotg`, this function returns c as a
    float rather than a complex.
    """
    if cmath.isclose(a, 0.0, abs_tol=tol):
        return 0.0, 1 + 0j
    if cmath.isclose(b, 0.0, abs_tol=tol):
        return 1.0, 0j
    c, s = zrotg_(a, b)
    return c.real, s


def givens_decomposition(
    mat: np.ndarray,
) -> tuple[list[GivensRotation], np.ndarray]:
    r"""Givens rotation decomposition of a unitary matrix.

    The Givens rotation decomposition of an :math:`n \times n` unitary matrix :math:`U`
    is given by

    .. math::

        U = D G_L^* G_{L-1}^* \cdots G_1^*

    where :math:`D` is a diagonal matrix and each :math:`G_k` is a Givens rotation.
    Here, the star :math:`*` denotes the element-wise complex conjugate.
    A Givens rotation acts on the two-dimensional subspace spanned by the :math:`i`-th
    and :math:`j`-th basis vectors as

    .. math::

        \begin{pmatrix}
            c & s \\
            -s^* & c \\
        \end{pmatrix}

    where :math:`c` is a real number and :math:`s` is a complex number.
    Therefore, a Givens rotation is described by a 4-tuple
    :math:`(c, s, i, j)`, where :math:`c` and :math:`s` are the numbers appearing
    in the rotation matrix, and :math:`i` and :math:`j` are the
    indices of the basis vectors of the subspace being rotated.
    This function always returns Givens rotations with the property that
    :math:`i` and :math:`j` differ by at most one, that is, either :math:`j = i + 1`
    or :math:`j = i - 1`.

    The number of Givens rotations :math:`L` is at most :math:`\frac{n (n-1)}{2}`,
    but it may be less. If we think of Givens rotations acting on disjoint indices
    as operations that can be performed in parallel, then the entire sequence of
    rotations can always be performed using at most `n` layers of parallel operations.
    The decomposition algorithm is described in :ref:`[1] <reference>`.

    .. _reference:
    
    [1] William R. Clements et al.
    `Optimal design for universal multiport interferometers`_.

    Args:
        mat: The unitary matrix to decompose into Givens rotations.

    Returns:
        - A list containing the Givens rotations :math:`G_1, \ldots, G_L`.
          Each Givens rotation is represented as a 4-tuple
          :math:`(c, s, i, j)`, where :math:`c` and :math:`s` are the numbers appearing
          in the rotation matrix, and :math:`i` and :math:`j` are the
          indices of the basis vectors of the subspace being rotated.
        - A Numpy array containing the diagonal elements of the matrix :math:`D`.

    .. _Optimal design for universal multiport interferometers: https://doi.org/10.1364/OPTICA.3.001460
    """
    n, _ = mat.shape
    current_matrix = mat.astype(complex, copy=True)
    left_rotations = []
    right_rotations = []

    # compute left and right Givens rotations
    for i in range(n - 1):
        if i % 2 == 0:
            # rotate columns by right multiplication
            for j in range(i + 1):
                target_index = i - j
                row = n - j - 1
                if not cmath.isclose(current_matrix[row, target_index], 0.0):
                    # zero out element at target index in given row
                    c, s = zrotg(
                        current_matrix[row, target_index + 1],
                        current_matrix[row, target_index],
                    )
                    right_rotations.append(
                        GivensRotation(c, s, target_index + 1, target_index)
                    )
                    (
                        current_matrix[:, target_index + 1],
                        current_matrix[:, target_index],
                    ) = zrot(
                        current_matrix[:, target_index + 1],
                        current_matrix[:, target_index],
                        c,
                        s,
                    )
        else:
            # rotate rows by left multiplication
            for j in range(i + 1):
                target_index = n - i + j - 1
                col = j
                if not cmath.isclose(current_matrix[target_index, col], 0.0):
                    # zero out element at target index in given column
                    c, s = zrotg(
                        current_matrix[target_index - 1, col],
                        current_matrix[target_index, col],
                    )
                    left_rotations.append(
                        GivensRotation(c, s, target_index - 1, target_index)
                    )
                    (
                        current_matrix[target_index - 1],
                        current_matrix[target_index],
                    ) = zrot(
                        current_matrix[target_index - 1],
                        current_matrix[target_index],
                        c,
                        s,
                    )

    # convert left rotations to right rotations
    for c, s, i, j in reversed(left_rotations):
        c, s = zrotg(c * current_matrix[j, j], s.conjugate() * current_matrix[i, i])
        right_rotations.append(GivensRotation(c, -s.conjugate(), i, j))

        givens_mat = np.array([[c, -s], [s.conjugate(), c]])
        givens_mat[:, 0] *= current_matrix[i, i]
        givens_mat[:, 1] *= current_matrix[j, j]
        c, s = zrotg(givens_mat[1, 1], givens_mat[1, 0])
        new_givens_mat = np.array([[c, s], [-s.conjugate(), c]])
        phase_matrix = givens_mat @ new_givens_mat
        current_matrix[i, i] = phase_matrix[0, 0]
        current_matrix[j, j] = phase_matrix[1, 1]

    # return decomposition
    return right_rotations, np.diagonal(current_matrix)
