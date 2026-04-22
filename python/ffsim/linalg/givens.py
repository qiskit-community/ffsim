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

import numpy as np

from ffsim import _lib


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


def givens_decomposition(
    mat: np.ndarray, tol: float = 1e-12
) -> tuple[list[tuple[float, complex, int, int]], np.ndarray]:
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
    rotations can always be performed using at most :math:`n` layers of parallel
    operations. The decomposition algorithm is described in the reference below.

    References:
        - `Clements et al., "Optimal design for universal multiport interferometers" (2016)`_

    Args:
        mat: The unitary matrix to decompose into Givens rotations.
        tol: Matrix entries smaller than this value will be treated as equal to zero.

    Returns:
        - A list containing the Givens rotations :math:`G_1, \ldots, G_L`.
          Each Givens rotation is represented as a 4-tuple
          :math:`(c, s, i, j)`, where :math:`c` and :math:`s` are the numbers appearing
          in the rotation matrix, and :math:`i` and :math:`j` are the
          indices of the basis vectors of the subspace being rotated.
        - A Numpy array containing the diagonal elements of the matrix :math:`D`.

    .. _Clements et al., "Optimal design for universal multiport interferometers" (2016): https://doi.org/10.1364/OPTICA.3.001460
    """  # noqa: E501
    return _lib.givens_decomposition(mat.astype(complex, copy=False), tol=tol)
