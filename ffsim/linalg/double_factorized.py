# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for performing the double-factorized decomposition."""

from __future__ import annotations

import numpy as np


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
