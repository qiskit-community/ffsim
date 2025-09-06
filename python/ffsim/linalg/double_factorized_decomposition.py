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

import itertools
import math
from typing import Literal, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.optimize
from opt_einsum import contract

from ffsim.linalg.util import (
    antihermitians_from_parameters,
    df_tensors_from_params,
    df_tensors_from_params_jax,
    df_tensors_to_params,
)


def _truncated_eigh(
    mat: np.ndarray,
    *,
    tol: float,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    eigs, vecs = scipy.linalg.eigh(mat)
    if max_vecs is None:
        max_vecs = len(eigs)
    indices = np.argsort(np.abs(eigs))[::-1]
    eigs = eigs[indices]
    vecs = vecs[:, indices]
    n_discard = int(np.searchsorted(np.cumsum(np.abs(eigs[::-1])), tol))
    n_vecs = cast(int, min(max_vecs, len(eigs) - n_discard))
    return eigs[:n_vecs], vecs[:, :n_vecs]


def _truncated_svd(
    mat: np.ndarray,
    *,
    tol: float,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_vecs, singular_vals, right_vecs = scipy.linalg.svd(mat, full_matrices=False)
    if max_vecs is None:
        max_vecs = len(singular_vals)
    n_discard = int(np.searchsorted(np.cumsum(singular_vals[::-1]), tol))
    n_vecs = cast(int, min(max_vecs, len(singular_vals) - n_discard))
    return left_vecs[:, :n_vecs], singular_vals[:n_vecs], right_vecs[:n_vecs]


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
        cholesky_vecs[:, index] /= math.sqrt(max_error)
        errors -= np.abs(cholesky_vecs[:, index]) ** 2

    return cholesky_vecs[:, :index]


def _validate_diag_coulomb_indices(indices: list[tuple[int, int]] | None):
    if indices is not None:
        for i, j in indices:
            if i > j:
                raise ValueError(
                    "When specifying diagonal Coulomb indices, you must give only "
                    "upper triangular indices. "
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
    return_optimize_result: bool = False,
    cholesky: bool = True,
):
    r"""Double-factorized decomposition of a two-body tensor.

    The double-factorized decomposition is a representation of a two-body tensor
    :math:`h_{pqrs}` as

    .. math::

        h_{pqrs} = \sum_{t=0}^{L - 1} \sum_{k\ell} Z^{t}_{k\ell} U^{t}_{pk} U^{t}_{qk}
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
    eigenvalue decomposition, and then truncate the terms based on the values of `tol`
    and `max_vecs`.
    If `optimize` is set to ``True``, then the entries of the resulting tensors
    (the diagonal Coulomb matrices and orbital rotations) are further optimized with
    `scipy.optimize.minimize` to reduce the error in the factorization.
    The diagonal Coulomb matrices returned by the optimization can be
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
        optimize: Whether to optimize the tensors returned by the decomposition to
            to minimize the error in the factorization.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
            This argument is ignored if `optimize` is set to ``False``.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
            This argument is ignored if `optimize` is set to ``False``.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
            This argument is ignored if `optimize` is set to ``False``.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices. Matrix entries corresponding to indices not in this
            list will be set to zero. This list should contain only upper
            triangular indices, i.e., pairs :math:`(i, j)` where :math:`i \leq j`.
            This argument is ignored if `optimize` is set to ``False``.
        return_optimize_result: Whether to also return the `OptimizeResult`_ returned
            by `scipy.optimize.minimize`_.
            This argument is ignored if `optimize` is set to ``False``.
        cholesky: Whether to perform the factorization using a modified Cholesky
            decomposition. If False, a full eigenvalue decomposition is used instead,
            which can be much more expensive. This argument is ignored if ``optimize``
            is set to ``True``.

    Returns:
        The diagonal Coulomb matrices and the orbital rotations. Each list of matrices
        is collected into a Numpy array, so this method returns a tuple of two Numpy
        arrays, the first containing the diagonal Coulomb matrices and the second
        containing the orbital rotations. Each Numpy array will have shape (L, n, n)
        where L is the number of terms in the decomposition and n is the number of
        orbitals. If `optimize` and `return_optimize_result` are both set to ``True``,
        the `OptimizeResult`_ returned by `scipy.optimize.minimize`_ is also returned.

    Raises:
        ValueError: diag_coulomb_indices contains lower triangular indices.

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    if np.iscomplexobj(two_body_tensor):
        raise ValueError(
            "Double factorization of complex two-body tensors is not supported."
        )

    norb, _, _, _ = two_body_tensor.shape

    if not norb:
        return np.empty((0, 0, 0)), np.empty((0, 0, 0))

    if max_vecs is None:
        max_vecs = norb * (norb + 1) // 2
    if optimize:
        return _double_factorized_compressed(
            two_body_tensor,
            tol=tol,
            max_vecs=max_vecs,
            method=method,
            callback=callback,
            options=options,
            diag_coulomb_indices=diag_coulomb_indices,
            return_optimize_result=return_optimize_result,
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
    max_vecs: int | None,
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
    max_vecs: int | None,
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
    max_vecs: int | None,
    method: str,
    callback,
    options: dict | None,
    diag_coulomb_indices: list[tuple[int, int]] | None,
    return_optimize_result: bool,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]
):
    diag_coulomb_mats, orbital_rotations = _double_factorized_explicit_cholesky(
        two_body_tensor, tol=tol, max_vecs=max_vecs
    )
    n_tensors, norb, _ = orbital_rotations.shape

    if diag_coulomb_indices is None:
        rows, cols = np.triu_indices(norb)
    else:
        rows, cols = zip(*diag_coulomb_indices)  # type: ignore

    n_triu = norb * (norb - 1) // 2

    def fun(x):
        diag_coulomb_mats, orbital_rotations = df_tensors_from_params(
            x, n_tensors, norb, diag_coulomb_indices, real=True
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
        diag_coulomb_mats, orbital_rotations = df_tensors_from_params(
            x, n_tensors, norb, diag_coulomb_indices, real=True
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
        grad_orbital_rotations = -4 * contract(
            "pqrs,tqk,tkl,trl,tsl->tpk",
            diff,
            orbital_rotations,
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations,
            optimize="greedy",
        )
        # TODO this computation is redundant
        generators = antihermitians_from_parameters(
            x[: n_tensors * n_triu], norb, n_tensors, real=True
        )
        grad_generator = np.ravel(
            [
                _grad_generator(log, grad)
                for log, grad in zip(generators, grad_orbital_rotations)
            ]
        )
        grad_diag_coulomb = -2 * contract(
            "pqrs,tpk,tqk,trl,tsl->tkl",
            diff,
            orbital_rotations,
            orbital_rotations,
            orbital_rotations,
            orbital_rotations,
            optimize="greedy",
        )
        grad_diag_coulomb[:, range(norb), range(norb)] /= 2
        grad_diag_coulomb = np.ravel([mat[rows, cols] for mat in grad_diag_coulomb])
        return np.concatenate([grad_generator, grad_diag_coulomb])

    x0 = df_tensors_to_params(
        diag_coulomb_mats, orbital_rotations, diag_coulomb_indices, real=True
    )
    result = scipy.optimize.minimize(
        fun, x0, method=method, jac=jac, callback=callback, options=options
    )
    diag_coulomb_mats, orbital_rotations = df_tensors_from_params(
        result.x, n_tensors, norb, diag_coulomb_indices, real=True
    )

    if return_optimize_result:
        return diag_coulomb_mats, orbital_rotations, result
    return diag_coulomb_mats, orbital_rotations


def _grad_generator(mat: np.ndarray, grad_orbital_rotation: np.ndarray) -> np.ndarray:
    eigs, vecs = scipy.linalg.eigh(-1j * mat)
    eig_i, eig_j = np.meshgrid(eigs, eigs, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        coeffs = -1j * (np.exp(1j * eig_i) - np.exp(1j * eig_j)) / (eig_i - eig_j)
    coeffs[eig_i == eig_j] = np.exp(1j * eig_i[eig_i == eig_j])
    grad = (
        vecs.conj() @ (vecs.T @ grad_orbital_rotation @ vecs.conj() * coeffs) @ vecs.T
    )
    grad -= grad.T
    norb, _ = mat.shape
    triu_indices = np.triu_indices(norb, k=1)
    return np.real(grad[triu_indices])


@overload
def double_factorized_t2(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = ...,
    max_terms: int | None = ...,
    optimize: bool = ...,
    method: str = ...,
    callback=...,
    options: dict | None = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    regularization: float = ...,
    multi_stage_start: int | None = ...,
    multi_stage_step: int | None = ...,
    return_optimize_result: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def double_factorized_t2(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = ...,
    max_terms: int | None = ...,
    optimize: bool = ...,
    method: str = ...,
    callback=...,
    options: dict | None = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    regularization: float = ...,
    multi_stage_start: int | None = ...,
    multi_stage_step: int | None = ...,
    return_optimize_result: Literal[True],
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]: ...
def double_factorized_t2(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = 1e-8,
    max_terms: int | None = None,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    regularization: float = 0,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    return_optimize_result: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]
):
    r"""Double-factorized decomposition of t2 amplitudes.

    The double-factorized decomposition of a t2 amplitudes tensor :math:`t_{ijab}` is

    .. math::

        t_{ijab} = i \sum_{k=0}^{L - 1} \sum_{pq}
            Z^{(k)}_{pq}
            U^{(k)}_{(\eta + a)p} U^{(k)*}_{ip} U^{(k)}_{(\eta + b)q} U^{(k)*}_{jq}

    Here each :math:`Z^{(k)}` is a real symmetric matrix, referred to as a
    "diagonal Coulomb matrix," each :math:`U^{(k)}` is a unitary matrix,
    referred to as an "orbital rotation," and the shape of the t2 amplitudes tensor is
    :math:`(\eta, \eta, N - \eta, N - \eta)`, where :math:`N` is the number of orbitals
    and :math:`\eta` is the number of orbitals that are occupied.

    The number of terms :math:`L` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the `max_terms` parameter specifies an optional upper bound
    on :math:`L`. The `max_terms` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    The default behavior of this routine is to perform a straightforward
    "exact" factorization of the t2 amplitudes tensor based on a nested
    eigenvalue decomposition, and then truncate the terms based on the values of `tol`
    and `max_terms`.
    If `optimize` is set to ``True``, then the entries of the resulting tensors
    (the diagonal Coulomb matrices and orbital rotations) are further optimized with
    `scipy.optimize.minimize`_ to reduce the error in the factorization.
    The diagonal Coulomb matrices returned by the optimization can be
    optionally constrained to have only certain elements allowed to be nonzero.
    This is achieved by passing the `diag_coulomb_indices` parameter, which is a
    list of matrix entry indices (integer pairs) specifying where the diagonal Coulomb
    matrices are allowed to be nonzero. Since the diagonal Coulomb matrices are
    symmetric, only upper triangular indices should be given, i.e.,
    pairs :math:`(i, j)` where :math:`i \leq j`.

    A "multi-stage" optimization can be requested by passing either `multi_stage_start`
    or `multi_stage_step`. In multi-stage optimization, the number of terms is
    iteratively reduced by `multi_stage_step`, starting from `multi_stage_start`,
    until there are only `max_terms` remaining. This procedure can yield a more accurate
    factorization at the cost of increased running time. If you only pass one of
    `multi_stage_start` or `multi_stage_step`, then the value of the other is chosen
    automatically: `multi_stage_start` defaults to the total number of terms returned by
    the exact factorization, and `multi_stage_step` defaults to 1.

    Note: Currently, only real-valued t2 amplitudes are supported.

    Args:
        t2_amplitudes: The t2 amplitudes tensor.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_terms: An optional limit on the number of terms to keep in the decomposition
            of the t2 amplitudes tensor. This argument overrides `tol`.
        optimize: Whether to optimize the tensors returned by the decomposition to
            to minimize the error in the factorization.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
            This argument is ignored if `optimize` is set to ``False``.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
            This argument is ignored if `optimize` is set to ``False``.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
            This argument is ignored if `optimize` is set to ``False``.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices. Matrix entries corresponding to indices not in this
            list will be set to zero. This list should contain only upper
            triangular indices, i.e., pairs :math:`(i, j)` where :math:`i \leq j`.
            This argument is ignored if `optimize` is set to ``False``.
        regularization: Parameter for regularizing the norm of the diagonal Coulomb
            matrices. This specifies the coefficient to a term added to the loss
            function equal to the difference between the sums of the squares of the
            Frobenius norms of the diagonal Coulomb matrices from the optimized
            factorization and the exact factorization.
            This argument is ignored if `optimize` is set to ``False``.
        multi_stage_start: The number of terms to start multi-stage optimization.
            This argument is ignored if `optimize` is set to ``False``.
        multi_stage_step: The number of terms to eliminate in each stage of multi-stage
            optimization.
            This argument is ignored if `optimize` is set to ``False``.
        return_optimize_result: Whether to also return the `OptimizeResult`_ returned
            by `scipy.optimize.minimize`_.
            This argument is ignored if `optimize` is set to ``False``.

    Returns:
        The diagonal Coulomb matrices and the orbital rotations. Each list of matrices
        is collected into a Numpy array, so this method returns a tuple of two Numpy
        arrays, the first containing the diagonal Coulomb matrices and the second
        containing the orbital rotations. Each Numpy array will have shape (L, n, n)
        where L is the number of terms in the decomposition and n is the number of
        orbitals. If `optimize` and `return_optimize_result` are both set to ``True``,
        the `OptimizeResult`_ returned by `scipy.optimize.minimize`_ is also returned.

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    if optimize:
        return _double_factorized_t2_compressed(
            t2_amplitudes,
            tol=tol,
            max_terms=max_terms,
            diag_coulomb_indices=diag_coulomb_indices,
            method=method,
            callback=callback,
            options=options,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
            regularization=regularization,
            return_optimize_result=return_optimize_result,
        )
    return _double_factorized_t2_explicit(t2_amplitudes, tol=tol, max_terms=max_terms)


def _double_factorized_t2_explicit(
    t2_amplitudes: np.ndarray, *, tol: float = 1e-8, max_terms: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    nocc, _, nvrt, _ = t2_amplitudes.shape
    norb = nocc + nvrt

    t2_mat = t2_amplitudes.transpose(0, 2, 1, 3).reshape(nocc * nvrt, nocc * nvrt)
    outer_eigs, outer_vecs = _truncated_eigh(t2_mat, tol=tol)

    n_vecs = len(outer_eigs)

    one_body_tensors = np.zeros((n_vecs, 2, norb, norb), dtype=complex)
    for outer_vec, one_body_tensor in zip(outer_vecs.T, one_body_tensors):
        mat = np.zeros((norb, norb))
        col, row = zip(*itertools.product(range(nocc), range(nocc, nocc + nvrt)))
        mat[row, col] = outer_vec
        one_body_tensor[0] = _quadrature(mat, sign=1)
        one_body_tensor[1] = _quadrature(mat, sign=-1)

    eigs, orbital_rotations = np.linalg.eigh(one_body_tensors)
    coeffs = np.array([1, -1]) * outer_eigs[:, None]
    diag_coulomb_mats = (
        coeffs[:, :, None, None] * eigs[:, :, :, None] * eigs[:, :, None, :]
    )

    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:max_terms]
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:max_terms]

    return diag_coulomb_mats, orbital_rotations


def _double_factorized_t2_compressed(
    t2_amplitudes: np.ndarray,
    *,
    tol: float,
    max_terms: int | None,
    method: str,
    callback,
    options: dict | None,
    diag_coulomb_indices: list[tuple[int, int]] | None,
    multi_stage_start: int | None,
    multi_stage_step: int | None,
    regularization: float,
    return_optimize_result: bool,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]
):
    nocc, _, nvrt, _ = t2_amplitudes.shape
    norb = nocc + nvrt

    diag_coulomb_mats, orbital_rotations = _double_factorized_t2_explicit(
        t2_amplitudes, tol=tol
    )
    init_diag_coulomb_norm = np.sum(np.abs(diag_coulomb_mats) ** 2)
    n_terms_full, _, _ = orbital_rotations.shape

    if max_terms is None or n_terms_full < max_terms:
        return diag_coulomb_mats, orbital_rotations

    if multi_stage_start is None and multi_stage_step is None:
        list_reps = [max_terms]
    else:
        multi_stage_start = min(n_terms_full, multi_stage_start or n_terms_full)
        multi_stage_step = multi_stage_step or 1
        list_reps = list(range(multi_stage_start, max_terms, -multi_stage_step))
        list_reps.append(max_terms)

    for n_tensors in list_reps:
        diag_coulomb_mats = diag_coulomb_mats[:n_tensors]
        orbital_rotations = orbital_rotations[:n_tensors]

        def fun(x: np.ndarray):
            diag_coulomb_mats, orbital_rotations = df_tensors_from_params_jax(
                x, n_tensors, norb, diag_coulomb_indices
            )
            reconstructed = (
                1j
                * contract(
                    "mpq,map,mip,mbq,mjq->ijab",
                    diag_coulomb_mats,
                    orbital_rotations,
                    orbital_rotations.conj(),
                    orbital_rotations,
                    orbital_rotations.conj(),
                    optimize="greedy",
                )[:nocc, :nocc, nocc:, nocc:]
            )
            loss = 0.5 * jnp.sum(jnp.abs(reconstructed - t2_amplitudes) ** 2)
            if regularization:
                diag_coulomb_norm = jnp.sum(jnp.abs(diag_coulomb_mats) ** 2)
                loss += regularization * jnp.abs(
                    diag_coulomb_norm - init_diag_coulomb_norm
                )
            return loss

        value_and_grad_func = jax.value_and_grad(fun)

        x0 = df_tensors_to_params(
            diag_coulomb_mats, orbital_rotations, diag_coulomb_indices
        )
        result = scipy.optimize.minimize(
            value_and_grad_func,
            x0,
            method=method,
            jac=True,
            callback=callback,
            options=options,
        )
        diag_coulomb_mats, orbital_rotations = df_tensors_from_params(
            result.x, n_tensors, norb, diag_coulomb_indices
        )

    if return_optimize_result:
        return diag_coulomb_mats, orbital_rotations, result
    return diag_coulomb_mats, orbital_rotations


def double_factorized_t2_alpha_beta(
    t2_amplitudes: np.ndarray, *, tol: float = 1e-8, max_terms: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Double-factorized decomposition of alpha-beta t2 amplitudes.

    Decompose alpha-beta t2 amplitudes into diagonal Coulomb matrices with orbital
    rotations. This function returns two arrays:

    - `diagonal_coulomb_mats`, with shape `(n_terms, 3, norb, norb)`.
    - `orbital_rotations`, with shape `(n_terms, 2, norb, norb)`.

    The value of `n_terms` depends on the error tolerance `tol`. A larger error
    tolerance might yield a smaller value for `n_terms`. You can also set an optional
    upper bound on `n_terms` using the `max_terms` argument.

    The original t2 amplitudes tensor can be reconstructed, up to the error tolerance,
    using the following function:

    .. code::

        def reconstruct_t2_alpha_beta(
            diag_coulomb_mats: np.ndarray,
            orbital_rotations: np.ndarray,
            norb: int,
            nocc_a: int,
            nocc_b: int,
        ) -> np.ndarray:
            n_terms = diag_coulomb_mats.shape[0]
            expanded_diag_coulomb_mats = np.zeros((n_terms, 2 * norb, 2 * norb))
            expanded_orbital_rotations = np.zeros(
                (n_terms, 2 * norb, 2 * norb), dtype=complex
            )
            for k in range(n_terms):
                (mat_aa, mat_ab, mat_bb) = diag_coulomb_mats[k]
                expanded_diag_coulomb_mats[k] = np.block(
                    [[mat_aa, mat_ab], [mat_ab.T, mat_bb]]
                )
                orbital_rotation_a, orbital_rotation_b = orbital_rotations[k]
                expanded_orbital_rotations[k] = scipy.linalg.block_diag(
                    orbital_rotation_a, orbital_rotation_b
                )
            return (
                2j
                * contract(
                    "kpq,kap,kip,kbq,kjq->ijab",
                    expanded_diag_coulomb_mats,
                    expanded_orbital_rotations,
                    expanded_orbital_rotations.conj(),
                    expanded_orbital_rotations,
                    expanded_orbital_rotations.conj(),
                )[:nocc_a, norb : norb + nocc_b, nocc_a:norb, norb + nocc_b :]
            )

    Note: Currently, only real-valued t2 amplitudes are supported.

    Args:
        t2_amplitudes: The t2 amplitudes tensor.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_terms: An optional limit on the number of terms to keep in the decomposition
            of the t2 amplitudes tensor. This argument overrides `tol`.


    Returns:
        - The diagonal Coulomb matrices, as a Numpy array of shape
          `(n_terms, 3, norb, norb)`.
          The last two axes index the rows and columns of
          the matrices, and the third from last axis, which has 3 dimensions, indexes
          the spin interaction type of the matrix: alpha-alpha, alpha-beta, and
          beta-beta (in that order).
          The first axis indexes the terms of the decomposition.
        - The orbital rotations, as a Numpy array of shape
          `(n_terms, 2, norb, norb)`. The last two axes index the rows and columns of
          the orbital rotations, and the third from last axis, which has 2 dimensions,
          indexes the spin sector of the orbital rotation: first alpha, then beta.
          The first axis indexes the terms of the decomposition.
    """
    nocc_a, nocc_b, nvrt_a, nvrt_b = t2_amplitudes.shape
    norb = nocc_a + nvrt_a

    t2_mat = t2_amplitudes.transpose(0, 2, 1, 3).reshape(
        nocc_a * nvrt_a, nocc_b * nvrt_b
    )
    left_vecs, singular_vals, right_vecs = _truncated_svd(t2_mat, tol=tol)

    n_vecs = len(singular_vals)

    one_body_tensors = np.zeros((n_vecs, 2, 2, 2, norb, norb), dtype=complex)
    for left_vec, right_vec, these_one_body_tensors in zip(
        left_vecs.T, right_vecs, one_body_tensors
    ):
        left_mat = np.zeros((norb, norb))
        col, row = zip(*itertools.product(range(nocc_a), range(nocc_a, norb)))
        left_mat[row, col] = left_vec
        right_mat = np.zeros((norb, norb))
        col, row = zip(*itertools.product(range(nocc_b), range(nocc_b, norb)))
        right_mat[row, col] = right_vec
        these_one_body_tensors[0, :, 0] = _quadrature(left_mat, 1)
        these_one_body_tensors[0, 0, 1] = _quadrature(right_mat, 1)
        these_one_body_tensors[0, 1, 1] = _quadrature(-right_mat, 1)
        these_one_body_tensors[1, :, 0] = _quadrature(left_mat, -1)
        these_one_body_tensors[1, 0, 1] = _quadrature(right_mat, -1)
        these_one_body_tensors[1, 1, 1] = _quadrature(-right_mat, -1)
    one_body_tensors = one_body_tensors.reshape((n_vecs, 4, 2, norb, norb))

    eigs, orbital_rotations = np.linalg.eigh(one_body_tensors)
    eigs = np.concatenate([eigs[:, :, 0], eigs[:, :, 1]], axis=-1)
    coeffs = 0.25 * np.array([1, -1, -1, 1]) * singular_vals[:, None]
    big_diag_coulomb_mats = (
        coeffs[:, :, None, None] * eigs[:, :, :, None] * eigs[:, :, None, :]
    )
    mats_aa = big_diag_coulomb_mats[:, :, :norb, :norb]
    mats_ab = big_diag_coulomb_mats[:, :, :norb, norb:]
    mats_bb = big_diag_coulomb_mats[:, :, norb:, norb:]
    diag_coulomb_mats = np.stack([mats_aa, mats_ab, mats_bb], axis=2)

    orbital_rotations = orbital_rotations.reshape(-1, 2, norb, norb)[:max_terms]
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, 3, norb, norb)[:max_terms]

    return diag_coulomb_mats, orbital_rotations


def _quadrature(mat: np.ndarray, sign: int):
    return 0.5 * (1 - sign * 1j) * (mat + sign * 1j * mat.T.conj())
