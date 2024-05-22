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
import scipy.linalg
import scipy.optimize
from opt_einsum import contract


def modified_cholesky(
    mat: np.ndarray, *, tol: float = 1e-8, max_vecs: int | None = None
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
        tol: Tolerance for error in the decomposition.
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

    if not dim:
        return np.empty((0, 0))

    if max_vecs is None:
        max_vecs = dim

    cholesky_vecs = np.zeros((dim, max_vecs + 1), dtype=mat.dtype)
    errors = np.real(np.diagonal(mat).copy())
    for index in range(max_vecs + 1):
        max_error_index = np.argmax(errors)
        max_error = errors[max_error_index]
        if max_error < tol:
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


def _validate_diag_coulomb_indices(indices: list[tuple[int, int]] | None):
    if indices is not None:
        for i, j in indices:
            if i > j:
                raise ValueError(
                    "When specifying diagonal Coulomb indices, you must give only "
                    "upper trianglular indices. "
                    f"Got {(i, j)}, which is a lower triangular index."
                )


def double_factorized(
    two_body_tensor: np.ndarray,
    *,
    tol: float = 1e-8,
    max_vecs: int | None = None,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    cholesky: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Double-factorized decomposition of a two-body tensor.

    The double-factorized decomposition is a representation of a two-body tensor
    :math:`h_{pqrs}` as

    .. math::

        h_{pqrs} = \sum_{t=1}^L \sum_{k\ell} Z^{t}_{k\ell} U^{t}_{pk} U^{t}_{qk}
            U^{t}_{r\ell} U^{t}_{s\ell}

    Here each :math:`Z^{(t)}` is a real symmetric matrix, referred to as a
    "diagonal Coulomb matrix," and each :math:`U^{t}` is a unitary matrix, referred to
    as an "orbital rotation."

    The number of terms :math:`L` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the `max_vecs` parameter specifies an optional upper bound
    on :math:`L`. The `max_vecs` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    The default behavior of this routine is to perform a straightforward
    "exact" factorization of the two-body tensor based on a nested
    eigenvalue decomposition. Additionally, one can choose to optimize the
    coefficients stored in the tensor to achieve a "compressed" factorization.
    This option is enabled by setting the `optimize` parameter to `True`.
    The optimization attempts to minimize a least-squares objective function
    quantifying the error in the decomposition.
    It uses `scipy.optimize.minimize`, passing both the objective function
    and its gradient. The diagonal Coulomb matrices returned by the optimization can be
    optionally constrained to have only certain elements allowed to be nonzero.
    This is achieved by passing the `diag_coulomb_indices` parameter, which is a
    list of matrix entry indices (integer pairs) specifying where the diagonal Coulomb
    matrices are allowed to be nonzero. Since the diagonal Coulomb matrices are
    symmetric, only upper triangular indices should be given, i.e.,
    pairs :math:`(i, j)` where :math:`i \leq j`.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    Note: Currently, only real-valued two-body tensors are supported.

    Args:
        two_body_tensor: The two-body tensor to decompose.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: An optional limit on the number of terms to keep in the decomposition
            of the two-body tensor. This argument overrides ``tol``.
        optimize: Whether to optimize the tensors returned by the decomposition.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices. Matrix entries corresponding to indices not in this
            list will be set to zero. This list should contain only upper
            trianglular indices, i.e., pairs :math:`(i, j)` where :math:`i \leq j`.
            Passing a list with lower triangular indices will raise an error.
            This parameter is only used if `optimize` is set to True.
        cholesky: Whether to perform the factorization using a modified Cholesky
            decomposition. If False, a full eigenvalue decomposition is used instead,
            which can be much more expensive. This argument is ignored if ``optimize``
            is set to True.

    Returns:
        The diagonal Coulomb matrices and the orbital rotations. Each list of matrices
        is collected into a Numpy array, so this method returns a tuple of two Numpy
        arrays, the first containing the diagonal Coulomb matrices and the second
        containing the orbital rotations. Each Numpy array will have shape (L, n, n)
        where L is the rank of the decomposition and n is the number of orbitals.

    Raises:
        ValueError: diag_coulomb_indices contains lower triangular indices.

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    _validate_diag_coulomb_indices(diag_coulomb_indices)

    norb, _, _, _ = two_body_tensor.shape

    if not norb:
        return np.empty((0, 0, 0)), np.empty((0, 0, 0))

    if max_vecs is None:
        max_vecs = norb * (norb + 1) // 2
    if optimize:
        if diag_coulomb_indices is None:
            diag_coulomb_mask = None
        else:
            diag_coulomb_mask = np.zeros((norb, norb), dtype=bool)
            rows, cols = zip(*diag_coulomb_indices)
            diag_coulomb_mask[rows, cols] = True
            diag_coulomb_mask[cols, rows] = True
        return _double_factorized_compressed(
            two_body_tensor,
            tol=tol,
            max_vecs=max_vecs,
            method=method,
            callback=callback,
            options=options,
            diag_coulomb_mask=diag_coulomb_mask,
        )
    if cholesky:
        return _double_factorized_explicit_cholesky(
            two_body_tensor, tol=tol, max_vecs=max_vecs
        )
    return _double_factorized_explicit_eigh(two_body_tensor, tol=tol, max_vecs=max_vecs)


def _double_factorized_explicit_cholesky(
    two_body_tensor: np.ndarray,
    *,
    tol: float,
    max_vecs: int,
) -> tuple[np.ndarray, np.ndarray]:
    norb, _, _, _ = two_body_tensor.shape
    reshaped_tensor = np.reshape(two_body_tensor, (norb**2, norb**2))
    cholesky_vecs = modified_cholesky(reshaped_tensor, tol=tol, max_vecs=max_vecs)
    mats = cholesky_vecs.T.reshape((-1, norb, norb))
    eigs, orbital_rotations = np.linalg.eigh(mats)
    diag_coulomb_mats = eigs[:, :, None] * eigs[:, None, :]
    return diag_coulomb_mats, orbital_rotations


def _double_factorized_explicit_eigh(
    two_body_tensor: np.ndarray,
    *,
    tol: float,
    max_vecs: int,
) -> tuple[np.ndarray, np.ndarray]:
    norb, _, _, _ = two_body_tensor.shape
    reshaped_tensor = np.reshape(two_body_tensor, (norb**2, norb**2))
    outer_eigs, outer_vecs = _truncated_eigh(
        reshaped_tensor, tol=tol, max_vecs=max_vecs
    )
    mats = outer_vecs.T.reshape((-1, norb, norb))
    eigs, orbital_rotations = np.linalg.eigh(mats)
    diag_coulomb_mats = outer_eigs[:, None, None] * eigs[:, :, None] * eigs[:, None, :]
    return diag_coulomb_mats, orbital_rotations


def _truncated_eigh(
    mat: np.ndarray,
    *,
    tol: float,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    eigs, vecs = scipy.linalg.eigh(mat)
    # sort by absolute value
    indices = np.argsort(np.abs(eigs))
    eigs = eigs[indices]
    vecs = vecs[:, indices]
    # get index to truncate at
    index = int(np.searchsorted(np.cumsum(np.abs(eigs)), tol))
    # truncate, then reverse to put into descending order of absolute value
    eigs = eigs[index:][::-1]
    vecs = vecs[:, index:][:, ::-1]
    # truncate to final rank
    eigs = eigs[:max_vecs]
    vecs = vecs[:, :max_vecs]
    return eigs, vecs


def optimal_diag_coulomb_mats(
    two_body_tensor: np.ndarray,
    orbital_rotations: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """Compute optimal diagonal Coulomb matrices given fixed orbital rotations."""
    norb, _, _, _ = two_body_tensor.shape
    n_tensors, _, _ = orbital_rotations.shape

    dim = n_tensors * norb**2
    target = contract(
        "pqrs,tpk,tqk,trl,tsl->tkl",
        two_body_tensor,
        orbital_rotations,
        orbital_rotations,
        orbital_rotations,
        orbital_rotations,
        optimize="greedy",
    )
    target = np.reshape(target, (dim,))
    coeffs = np.zeros((n_tensors, norb, norb, n_tensors, norb, norb))
    for i in range(n_tensors):
        for j in range(i, n_tensors):
            metric = (orbital_rotations[i].T @ orbital_rotations[j]) ** 2
            coeffs[i, :, :, j, :, :] = np.einsum("kl,mn->kmln", metric, metric)
            coeffs[j, :, :, i, :, :] = np.einsum("kl,mn->kmln", metric.T, metric.T)
    coeffs = np.reshape(coeffs, (dim, dim))

    eigs, vecs = scipy.linalg.eigh(coeffs)
    pseudoinverse = np.zeros_like(eigs)
    pseudoinverse[eigs > tol] = eigs[eigs > tol] ** -1
    solution = vecs @ (vecs.T @ target * pseudoinverse)

    return np.reshape(solution, (n_tensors, norb, norb))


def _double_factorized_compressed(
    two_body_tensor: np.ndarray,
    *,
    tol: float,
    max_vecs: int,
    method: str,
    callback,
    options: dict | None,
    diag_coulomb_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    diag_coulomb_mats, orbital_rotations = _double_factorized_explicit_cholesky(
        two_body_tensor, tol=tol, max_vecs=max_vecs
    )
    n_tensors, norb, _ = orbital_rotations.shape
    if diag_coulomb_mask is None:
        diag_coulomb_mask = np.ones((norb, norb), dtype=bool)
    diag_coulomb_mask = np.triu(diag_coulomb_mask)

    def fun(x):
        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            x, n_tensors, norb, diag_coulomb_mask
        )
        diff = two_body_tensor - contract(
            "kpi,kqi,kij,krj,ksj->pqrs",
            orbital_rotations,
            orbital_rotations,
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations,
            optimize="greedy",
        )
        return 0.5 * np.sum(diff**2)

    def jac(x):
        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            x, n_tensors, norb, diag_coulomb_mask
        )
        diff = two_body_tensor - contract(
            "kpi,kqi,kij,krj,ksj->pqrs",
            orbital_rotations,
            orbital_rotations,
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations,
            optimize="greedy",
        )
        grad_leaf = -4 * contract(
            "pqrs,tqk,tkl,trl,tsl->tpk",
            diff,
            orbital_rotations,
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations,
            optimize="greedy",
        )
        leaf_logs = _params_to_leaf_logs(x, n_tensors, norb)
        grad_leaf_log = np.ravel(
            [_grad_leaf_log(log, grad) for log, grad in zip(leaf_logs, grad_leaf)]
        )
        grad_core = -2 * contract(
            "pqrs,tpk,tqk,trl,tsl->tkl",
            diff,
            orbital_rotations,
            orbital_rotations,
            orbital_rotations,
            orbital_rotations,
            optimize="greedy",
        )
        grad_core[:, range(norb), range(norb)] /= 2
        param_indices = np.nonzero(diag_coulomb_mask)
        grad_core = np.ravel([mat[param_indices] for mat in grad_core])
        return np.concatenate([grad_leaf_log, grad_core])

    x0 = _df_tensors_to_params(diag_coulomb_mats, orbital_rotations, diag_coulomb_mask)
    result = scipy.optimize.minimize(
        fun, x0, method=method, jac=jac, callback=callback, options=options
    )
    diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
        result.x, n_tensors, norb, diag_coulomb_mask
    )

    return diag_coulomb_mats, orbital_rotations


def _df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_mat_mask: np.ndarray,
):
    _, norb, _ = orbital_rotations.shape
    leaf_logs = [scipy.linalg.logm(mat) for mat in orbital_rotations]
    leaf_param_indices = np.triu_indices(norb, k=1)
    # TODO this discards the imaginary part of the logarithm, see if we can do better
    leaf_params = np.real(
        np.ravel([leaf_log[leaf_param_indices] for leaf_log in leaf_logs])
    )
    core_param_indices = np.nonzero(diag_coulomb_mat_mask)
    core_params = np.ravel(
        [diag_coulomb_mat[core_param_indices] for diag_coulomb_mat in diag_coulomb_mats]
    )
    return np.concatenate([leaf_params, core_params])


def _params_to_leaf_logs(params: np.ndarray, n_tensors: int, norb: int):
    leaf_logs = np.zeros((n_tensors, norb, norb))
    triu_indices = np.triu_indices(norb, k=1)
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_logs[i][triu_indices] = params[i * param_length : (i + 1) * param_length]
        leaf_logs[i] -= leaf_logs[i].T
    return leaf_logs


def _params_to_df_tensors(
    params: np.ndarray, n_tensors: int, norb: int, diag_coulomb_mat_mask: np.ndarray
):
    leaf_logs = _params_to_leaf_logs(params, n_tensors, norb)
    orbital_rotations = _expm_antisymmetric(leaf_logs)

    n_leaf_params = n_tensors * norb * (norb - 1) // 2
    core_params = np.real(params[n_leaf_params:])
    param_indices = np.nonzero(diag_coulomb_mat_mask)
    param_length = len(param_indices[0])
    diag_coulomb_mats = np.zeros((n_tensors, norb, norb))
    for i in range(n_tensors):
        diag_coulomb_mats[i][param_indices] = core_params[
            i * param_length : (i + 1) * param_length
        ]
        diag_coulomb_mats[i] += diag_coulomb_mats[i].T
        diag_coulomb_mats[i][range(norb), range(norb)] /= 2
    return diag_coulomb_mats, orbital_rotations


def _expm_antisymmetric(mats: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mats)
    return np.einsum("tij,tj,tkj->tik", vecs, np.exp(1j * eigs), vecs.conj()).real


def _grad_leaf_log(mat: np.ndarray, grad_leaf: np.ndarray) -> np.ndarray:
    eigs, vecs = scipy.linalg.eigh(-1j * mat)
    eig_i, eig_j = np.meshgrid(eigs, eigs, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        coeffs = -1j * (np.exp(1j * eig_i) - np.exp(1j * eig_j)) / (eig_i - eig_j)
    coeffs[eig_i == eig_j] = np.exp(1j * eig_i[eig_i == eig_j])
    grad = vecs.conj() @ (vecs.T @ grad_leaf @ vecs.conj() * coeffs) @ vecs.T
    grad -= grad.T
    norb, _ = mat.shape
    triu_indices = np.triu_indices(norb, k=1)
    return np.real(grad[triu_indices])


def double_factorized_t2(
    t2_amplitudes: np.ndarray, *, tol: float = 1e-8, max_vecs: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Double-factorized decomposition of t2 amplitudes.

    The double-factorized decomposition of a t2 amplitudes tensor :math:`t_{ijab}` is

    .. math::

        t_{ijab} = i \sum_{k=1}^L \sum_{pq} (
            Z^{k}_{pq} U^{k}_{ap} {U^{k}}^*_{ip} U^{k}_{bq} {U^{k}}^*_{jq}
            - Z^{k}_{pq} {U^{k}}^*_{ap} U^{k}_{ip} {U^{k}}^*_{bq} U^{k}_{jq})

    Here each :math:`Z^{(k)}` is a real symmetric matrix, referred to as a
    "diagonal Coulomb matrix," and each :math:`U^{k}` is a unitary matrix, referred to
    as an "orbital rotation."

    The number of terms :math:`L` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the `max_vecs` parameter specifies an optional upper bound
    on :math:`L`. The `max_vecs` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    Note: Currently, only real-valued t2 amplitudes are supported.

    Args:
        t2_amplitudes: The t2 amplitudes tensor.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: An optional limit on the number of terms to keep in the decomposition
            of the t2 amplitudes tensor. This argument overrides ``tol``.

    Returns:
        The diagonal Coulomb matrices and the orbital rotations. Each list of matrices
        is collected into a Numpy array, so this method returns a tuple of two Numpy
        arrays, the first containing the diagonal Coulomb matrices and the second
        containing the orbital rotations. Each Numpy array will have shape (L, n, n)
        where L is the rank of the decomposition and n is the number of orbitals.
    """
    nocc, _, nvrt, _ = t2_amplitudes.shape
    norb = nocc + nvrt

    two_body_tensor = np.zeros((norb, norb, norb, norb))
    two_body_tensor[:nocc, :nocc, nocc:, nocc:] = t2_amplitudes
    two_body_tensor = two_body_tensor.transpose((2, 0, 3, 1))
    t2_mat = two_body_tensor.reshape((norb**2, norb**2))
    outer_eigs, outer_vecs = _truncated_eigh(t2_mat, tol=tol, max_vecs=max_vecs)

    mats = outer_vecs.T.reshape((-1, norb, norb))
    mats = 0.5 * (1 - 1j) * (mats + 1j * mats.transpose((0, 2, 1)))
    eigs, orbital_rotations = np.linalg.eigh(mats)
    diag_coulomb_mats = outer_eigs[:, None, None] * eigs[:, :, None] * eigs[:, None, :]

    return diag_coulomb_mats, orbital_rotations
