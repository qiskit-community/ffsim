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

import functools
import math
import warnings
from collections.abc import Callable
from typing import Literal, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.optimize
from opt_einsum import contract

from ffsim.linalg.util import (
    df_tensors_alpha_beta_from_params,
    df_tensors_alpha_beta_from_params_jax,
    df_tensors_alpha_beta_to_params,
    df_tensors_from_params,
    df_tensors_from_params_jax,
    df_tensors_to_params,
)

jax.config.update("jax_enable_x64", True)


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

    where each :math:`v_i` is a vector. :math:`M` must be positive definite.
    No checking is performed to verify whether :math:`M` is positive definite.
    The number of terms :math:`N` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the ``max_vecs`` parameter specifies an optional upper bound
    on :math:`N`. The ``max_vecs`` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    References:
        - `Motta and Zhang, "Ab initio computations of molecular systems by the auxiliary-field quantum Monte Carlo method" (2017)`_

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

    .. _Motta and Zhang, "Ab initio computations of molecular systems by the auxiliary-field quantum Monte Carlo method" (2017): https://arxiv.org/abs/1711.02242
    """  # noqa: E501
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


def _validate_real_t2_amplitudes(t2_amplitudes: np.ndarray) -> None:
    if np.iscomplexobj(t2_amplitudes):
        raise ValueError(
            "Double factorization of complex t2 amplitudes is not supported."
        )


def _validate_multi_stage_args(
    multi_stage_start: int | None, multi_stage_step: int | None
) -> None:
    if multi_stage_start is not None and multi_stage_start < 1:
        raise ValueError(
            f"multi_stage_start must be at least 1. Got {multi_stage_start}."
        )
    if multi_stage_step is not None and multi_stage_step < 1:
        raise ValueError(
            f"multi_stage_step must be at least 1. Got {multi_stage_step}."
        )


def _warn_deprecated_optimize_args(
    func_name: str, compress_func_name: str, deprecated_args: dict[str, bool]
) -> None:
    names = [f"`{name}`" for name, is_set in deprecated_args.items() if is_set]
    if not names:
        return
    if len(names) == 1:
        names_str = f"{names[0]} argument of `{func_name}` is"
    else:
        names_str = (
            f"{', '.join(names[:-1])} and {names[-1]} arguments of `{func_name}` are"
        )
    warnings.warn(
        f"The {names_str} deprecated and will be removed in a future release. To "
        f"optimize the factorization, call `{compress_func_name}` on the output of "
        f"`{func_name}`.",
        DeprecationWarning,
        stacklevel=3,
    )


def _validate_df_tensor_shapes(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_mat_shape: tuple[int, ...],
    orbital_rotation_shape: tuple[int, ...],
) -> None:
    n_terms = len(diag_coulomb_mats)
    if not n_terms:
        raise ValueError("The factorization to compress must have at least one term.")
    if diag_coulomb_mats.shape != (n_terms, *diag_coulomb_mat_shape):
        raise ValueError(
            "Expected diagonal Coulomb matrices to have shape "
            f"{(n_terms, *diag_coulomb_mat_shape)}. Got {diag_coulomb_mats.shape}."
        )
    if orbital_rotations.shape != (n_terms, *orbital_rotation_shape):
        raise ValueError(
            "Expected orbital rotations to have shape "
            f"{(n_terms, *orbital_rotation_shape)}. Got {orbital_rotations.shape}."
        )


@overload
def double_factorized(
    two_body_tensor: np.ndarray,
    *,
    tol: float = ...,
    max_vecs: int | None = ...,
    optimize: bool = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    return_optimize_result: Literal[False] = False,
    cholesky: bool = ...,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def double_factorized(
    two_body_tensor: np.ndarray,
    *,
    tol: float = ...,
    max_vecs: int | None = ...,
    optimize: bool = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    return_optimize_result: Literal[True],
    cholesky: bool = ...,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult | None]: ...
@overload
def double_factorized(
    two_body_tensor: np.ndarray,
    *,
    tol: float = ...,
    max_vecs: int | None = ...,
    optimize: bool = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    return_optimize_result: bool = ...,
    cholesky: bool = ...,
    **optimize_kwargs,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult | None]
): ...
def double_factorized(
    two_body_tensor: np.ndarray,
    *,
    tol: float = 1e-8,
    max_vecs: int | None = None,
    optimize: bool = False,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    return_optimize_result: bool = False,
    cholesky: bool = True,
    **optimize_kwargs,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult | None]
):
    r"""Double-factorized decomposition of a two-body tensor.

    The double-factorized decomposition is a representation of a two-body tensor
    :math:`h_{pqrs}` as

    .. math::

        h_{pqrs} = \sum_{t=0}^{L - 1} \sum_{k\ell}
            Z^{(t)}_{k\ell} U^{(t)}_{pk} U^{(t)}_{qk} U^{(t)}_{r\ell} U^{(t)}_{s\ell},

    where each :math:`Z^{(t)}` is a real symmetric matrix representing a diagonal
    Coulomb operator and each :math:`U^{(t)}` is a unitary matrix representing an
    orbital rotation.

    The number of terms :math:`L` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the ``max_vecs`` parameter specifies an optional upper bound
    on :math:`L`. The ``max_vecs`` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    This routine performs a straightforward "exact" factorization of the two-body
    tensor based on a nested eigenvalue decomposition, and then truncates the terms
    based on the values of ``tol`` and ``max_vecs``. To further optimize the
    resulting tensors to reduce the error in the factorization, pass them to
    :func:`double_factorized_compressed`.

    .. warning::
        The ``optimize``, ``diag_coulomb_indices``, ``return_optimize_result``, and
        ``optimize_kwargs`` arguments are deprecated. Instead, call
        :func:`double_factorized_compressed` on the output of this function.

    References:
        - `Motta et al., "Low rank representations for quantum simulation of electronic structure" (2021)`_
        - `Cohn, Motta, and Parrish, "Quantum Filter Diagonalization with Double-Factorized Hamiltonians" (2021)`_

    Note: Currently, only real-valued two-body tensors are supported.

    Args:
        two_body_tensor: The two-body tensor to decompose.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: An optional limit on the number of terms to keep in the decomposition
            of the two-body tensor. This argument overrides ``tol``.
        optimize: Deprecated. Whether to optimize the tensors returned by the
            decomposition to minimize the error in the factorization.
        diag_coulomb_indices: Deprecated. Allowed indices for nonzero values of the
            diagonal Coulomb matrices when ``optimize`` is set to ``True``.
            See :func:`double_factorized_compressed`.
        return_optimize_result: Deprecated. Whether to also return the
            :class:`~scipy.optimize.OptimizeResult` returned by
            :func:`scipy.optimize.minimize`. If no optimization is performed,
            ``None`` is returned in its place.
        cholesky: Whether to perform the factorization using a modified Cholesky
            decomposition. If False, a full eigenvalue decomposition is used instead,
            which can be much more expensive. This argument is ignored if ``optimize``
            is set to ``True``.
        optimize_kwargs: Deprecated. Keyword arguments to pass to
            :func:`scipy.optimize.minimize` when ``optimize`` is set to ``True``.

    Returns:
        The diagonal Coulomb matrices and the orbital rotations. Each list of matrices
        is collected into a Numpy array, so this method returns a tuple of two Numpy
        arrays, the first containing the diagonal Coulomb matrices and the second
        containing the orbital rotations. Each Numpy array will have shape (L, n, n)
        where L is the number of terms in the decomposition and n is the number of
        orbitals. If ``return_optimize_result`` is set to ``True``, the
        :class:`~scipy.optimize.OptimizeResult` returned by
        :func:`scipy.optimize.minimize` is also returned, or ``None`` if no
        optimization was performed. No optimization is performed if ``optimize`` is
        set to ``False``, or if the two-body tensor has zero orbitals.

    Raises:
        ValueError: The two-body tensor is complex-valued.
        ValueError: ``optimize`` is True, the two-body tensor has at least one
            orbital, and the factorization has no terms.
        ValueError: ``optimize`` is True, the two-body tensor has at least one
            orbital, and diag_coulomb_indices contains lower triangular indices.

    .. _Motta et al., "Low rank representations for quantum simulation of electronic structure" (2021): https://arxiv.org/abs/1808.02625
    .. _Cohn, Motta, and Parrish, "Quantum Filter Diagonalization with Double-Factorized Hamiltonians" (2021): https://arxiv.org/abs/2104.08957
    """  # noqa: E501
    _warn_deprecated_optimize_args(
        "double_factorized",
        "double_factorized_compressed",
        {
            "optimize": optimize,
            "diag_coulomb_indices": diag_coulomb_indices is not None,
            "return_optimize_result": return_optimize_result,
            "optimize_kwargs": bool(optimize_kwargs),
        },
    )
    if np.iscomplexobj(two_body_tensor):
        raise ValueError(
            "Double factorization of complex two-body tensors is not supported."
        )

    norb, _, _, _ = two_body_tensor.shape

    result: scipy.optimize.OptimizeResult | None = None
    if not norb:
        diag_coulomb_mats, orbital_rotations = (
            np.empty((0, 0, 0)),
            np.empty((0, 0, 0)),
        )
    else:
        if max_vecs is None:
            max_vecs = norb * (norb + 1) // 2
        if optimize:
            diag_coulomb_mats, orbital_rotations = _double_factorized_explicit_cholesky(
                two_body_tensor, tol=tol, max_vecs=max_vecs
            )
            diag_coulomb_mats, orbital_rotations, result = double_factorized_compressed(
                two_body_tensor,
                initial_guess=(diag_coulomb_mats, orbital_rotations),
                diag_coulomb_indices=diag_coulomb_indices,
                **optimize_kwargs,
            )
        elif cholesky:
            diag_coulomb_mats, orbital_rotations = _double_factorized_explicit_cholesky(
                two_body_tensor, tol=tol, max_vecs=max_vecs
            )
        else:
            diag_coulomb_mats, orbital_rotations = _double_factorized_explicit_eigh(
                two_body_tensor, tol=tol, max_vecs=max_vecs
            )
    if return_optimize_result:
        return diag_coulomb_mats, orbital_rotations, result
    return diag_coulomb_mats, orbital_rotations


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


@functools.cache
def _make_compressed_value_and_grad(
    n_tensors: int,
    norb: int,
    diag_coulomb_indices: tuple[tuple[int, int], ...] | None,
):
    """Build a jitted value-and-gradient function for the compression objective.

    The result is cached and reused across optimizer iterations and across calls with
    the same static structure ``(n_tensors, norb, diag_coulomb_indices)``, so the loss
    is traced and compiled only once per structure. The target two-body tensor is passed
    as a runtime argument (it does not affect the traced graph), and the gradient is
    taken with respect to the flat variable vector ``x`` only.
    """
    indices = None if diag_coulomb_indices is None else list(diag_coulomb_indices)

    def loss(x: jax.Array, two_body_tensor: jax.Array) -> jax.Array:
        diag_coulomb_mats, orbital_rotations = df_tensors_from_params_jax(
            x, n_tensors, norb, indices, real=True
        )
        reconstructed = jnp.einsum(
            "kpi,kqi,kij,krj,ksj->pqrs",
            orbital_rotations,
            orbital_rotations,
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations,
            optimize=True,
        )
        return 0.5 * jnp.sum((two_body_tensor - reconstructed) ** 2)

    return jax.jit(jax.value_and_grad(loss, argnums=0))


def double_factorized_compressed(
    two_body_tensor: np.ndarray,
    *,
    initial_guess: tuple[np.ndarray, np.ndarray] | None = None,
    max_vecs: int | None = None,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]:
    r"""Compress a double-factorized decomposition of a two-body tensor.

    Optimizes the diagonal Coulomb matrices and orbital rotations of a
    double-factorized decomposition (see :func:`double_factorized`) with
    :func:`scipy.optimize.minimize` to reduce the error in the factorization.
    The optimization starts from the factorization given by ``initial_guess``. If it
    is not specified, the exact factorization computed by :func:`double_factorized`
    with the default tolerance is used. To use a different tolerance, call
    :func:`double_factorized` yourself and pass its output as ``initial_guess``.
    If ``max_vecs`` is specified, then only the first ``max_vecs`` terms of the
    initial guess are kept.

    The diagonal Coulomb matrices returned by the optimization can be
    optionally constrained to have only certain elements allowed to be nonzero.
    This is achieved by passing the ``diag_coulomb_indices`` parameter, which is a
    list of matrix entry indices (integer pairs) specifying where the diagonal Coulomb
    matrices are allowed to be nonzero. Since the diagonal Coulomb matrices are
    symmetric, only upper triangular indices should be given, i.e.,
    pairs :math:`(i, j)` where :math:`i \leq j`.

    Note: Currently, only real-valued two-body tensors are supported.

    Args:
        two_body_tensor: The two-body tensor to decompose.
        initial_guess: The initial guess for the optimization, as a tuple of the
            diagonal Coulomb matrices and the orbital rotations, each a Numpy array
            of shape ``(L, norb, norb)``. Both arrays must be real-valued.
            Typically, this is the output of :func:`double_factorized`. If not
            specified, the exact factorization computed by :func:`double_factorized`
            with the default tolerance is used.
        max_vecs: An optional limit on the number of terms to keep in the
            decomposition. If not specified, all the terms of the initial guess are
            kept.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices. Matrix entries corresponding to indices not in this
            list will be set to zero. This list should contain only upper
            triangular indices, i.e., pairs :math:`(i, j)` where :math:`i \leq j`.
        optimize_kwargs: Keyword arguments to pass to
            :func:`scipy.optimize.minimize`. The default method is
            ``"L-BFGS-B"``.

    Returns:
        - The optimized diagonal Coulomb matrices, as a Numpy array of shape
          ``(L, norb, norb)``.
        - The optimized orbital rotations, as a Numpy array of shape
          ``(L, norb, norb)``.
        - The :class:`~scipy.optimize.OptimizeResult` returned by
          :func:`scipy.optimize.minimize`.

    Raises:
        ValueError: The two-body tensor is complex-valued.
        ValueError: max_vecs is less than 1.
        ValueError: diag_coulomb_indices contains lower triangular indices.
        ValueError: The initial guess has no terms. This includes the case where
            ``initial_guess`` is not specified and the exact factorization has no
            terms.
        ValueError: The shapes of the initial guess tensors are inconsistent.
        ValueError: The initial guess tensors are complex-valued.
    """
    if np.iscomplexobj(two_body_tensor):
        raise ValueError(
            "Double factorization of complex two-body tensors is not supported."
        )
    if max_vecs is not None and max_vecs < 1:
        raise ValueError(f"max_vecs must be at least 1. Got {max_vecs}.")
    _validate_diag_coulomb_indices(diag_coulomb_indices)
    if initial_guess is None:
        # The modified Cholesky decomposition adds vectors one at a time, so this
        # yields the first max_vecs terms of the full factorization without computing
        # the rest
        initial_guess = double_factorized(two_body_tensor, max_vecs=max_vecs)
    diag_coulomb_mats, orbital_rotations = initial_guess
    norb, _, _, _ = two_body_tensor.shape
    _validate_df_tensor_shapes(
        diag_coulomb_mats, orbital_rotations, (norb, norb), (norb, norb)
    )
    if np.iscomplexobj(diag_coulomb_mats) or np.iscomplexobj(orbital_rotations):
        raise ValueError(
            "The initial guess must be real-valued. Got complex diagonal Coulomb "
            "matrices or orbital rotations."
        )
    diag_coulomb_mats, orbital_rotations, result = _compress_double_factorized(
        two_body_tensor,
        diag_coulomb_mats[:max_vecs],
        orbital_rotations[:max_vecs],
        diag_coulomb_indices=diag_coulomb_indices,
        **optimize_kwargs,
    )
    return diag_coulomb_mats, orbital_rotations, result


def _compress_double_factorized(
    two_body_tensor: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    *,
    diag_coulomb_indices: list[tuple[int, int]] | None,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]:
    """Compress a double factorization of a two-body tensor.

    The input tensors are used as the initial guess for the optimization. The
    factorization must have at least one term.
    """
    n_tensors, norb, _ = orbital_rotations.shape
    # The optimization parameterizes special orthogonal orbital rotations, but eigh
    # may return orbital rotations with determinant -1. Each column of an orbital
    # rotation appears twice in the reconstructed tensor, so negating one column fixes
    # the determinant without changing the factorization. Copy first so that the
    # caller's initial guess is not modified.
    orbital_rotations = orbital_rotations.copy()
    orbital_rotations[np.linalg.det(orbital_rotations) < 0, :, 0] *= -1

    # Structure key for the cached jitted objective. Convert to a hashable tuple so it
    # can be used as a cache key, and back to a list inside the builder.
    indices_key = None if diag_coulomb_indices is None else tuple(diag_coulomb_indices)
    two_body_tensor_jax = jnp.asarray(two_body_tensor)

    # Reuse a jitted value-and-gradient function cached by static structure, so the loss
    # is compiled once and reused across all optimizer iterations (and across calls with
    # the same structure) instead of being re-traced eagerly each step.
    value_and_grad = _make_compressed_value_and_grad(n_tensors, norb, indices_key)

    def scipy_func(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(x), two_body_tensor_jax)
        return float(value), np.asarray(grad)

    x0 = df_tensors_to_params(
        diag_coulomb_mats, orbital_rotations, diag_coulomb_indices, real=True
    )
    optimize_kwargs.setdefault("method", "L-BFGS-B")
    result = scipy.optimize.minimize(
        scipy_func,
        x0,
        jac=True,
        **optimize_kwargs,
    )
    diag_coulomb_mats, orbital_rotations = df_tensors_from_params(
        result.x, n_tensors, norb, diag_coulomb_indices, real=True
    )

    return diag_coulomb_mats, orbital_rotations, result


@overload
def double_factorized_t2(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = ...,
    max_terms: int | None = ...,
    optimize: bool = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    regularization: float = ...,
    multi_stage_start: int | None = ...,
    multi_stage_step: int | None = ...,
    return_optimize_result: Literal[False] = False,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def double_factorized_t2(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = ...,
    max_terms: int | None = ...,
    optimize: bool = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    regularization: float = ...,
    multi_stage_start: int | None = ...,
    multi_stage_step: int | None = ...,
    return_optimize_result: Literal[True],
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult | None]: ...
@overload
def double_factorized_t2(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = ...,
    max_terms: int | None = ...,
    optimize: bool = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    regularization: float = ...,
    multi_stage_start: int | None = ...,
    multi_stage_step: int | None = ...,
    return_optimize_result: bool = ...,
    **optimize_kwargs,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult | None]
): ...
def double_factorized_t2(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = 1e-8,
    max_terms: int | None = None,
    optimize: bool = False,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    regularization: float = 0,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    return_optimize_result: bool = False,
    **optimize_kwargs,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult | None]
):
    r"""Double-factorized decomposition of :math:`t_2` amplitudes.

    The double-factorized decomposition of a :math:`t_2` amplitudes tensor
    :math:`t_{ijab}` is

    .. math::

        t_{ijab} = i \sum_{k=0}^{L - 1} \sum_{pq}
            Z^{(k)}_{pq}
            U^{(k)}_{ap} U^{(k)*}_{ip} U^{(k)}_{bq} U^{(k)*}_{jq},

    where each :math:`Z^{(k)}` is a real symmetric matrix representing a diagonal
    Coulomb operator, each :math:`U^{(k)}` is a unitary matrix representing an orbital
    rotation, :math:`i` and :math:`j` run over occupied orbitals, and :math:`a` and
    :math:`b` run over virtual orbitals.

    The number of terms :math:`L` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the ``max_terms`` parameter specifies an optional upper bound
    on :math:`L`. The ``max_terms`` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    This routine performs a straightforward "exact" factorization of the
    :math:`t_2` amplitudes tensor based on a nested eigenvalue decomposition, and then
    truncates the terms based on the values of ``tol`` and ``max_terms``. To further
    optimize the resulting tensors to reduce the error in the factorization, pass the
    untruncated factorization to :func:`double_factorized_t2_compressed`.

    .. warning::
        The ``optimize``, ``diag_coulomb_indices``, ``regularization``,
        ``multi_stage_start``, ``multi_stage_step``, ``return_optimize_result``, and
        ``optimize_kwargs`` arguments are deprecated. Instead, call
        :func:`double_factorized_t2_compressed` on the output of this function.

    The original :math:`t_2` amplitudes tensor can be reconstructed, up to the error
    tolerance, using the following function:

    .. code::

        def reconstruct_t2(
            diag_coulomb_mats: np.ndarray, orbital_rotations: np.ndarray, nocc: int
        ) -> np.ndarray:
            return (
                1j
                * np.einsum(
                    "kpq,kap,kip,kbq,kjq->ijab",
                    diag_coulomb_mats,
                    orbital_rotations,
                    orbital_rotations.conj(),
                    orbital_rotations,
                    orbital_rotations.conj(),
                )[:nocc, :nocc, nocc:, nocc:]
            )

    Note: Currently, only real-valued :math:`t_2` amplitudes are supported.

    Args:
        t2_amplitudes: The doubles amplitudes tensor of shape
            ``(nocc, nocc, nvrt, nvrt)``, where ``nocc`` is the number of occupied
            orbitals and ``nvrt`` is the number of virtual orbitals.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_terms: An optional limit on the number of terms to keep in the decomposition
            of the :math:`t_2` amplitudes tensor. This argument overrides ``tol``.
        optimize: Deprecated. Whether to optimize the tensors returned by the
            decomposition to minimize the error in the factorization.
        diag_coulomb_indices: Deprecated. See :func:`double_factorized_t2_compressed`.
        regularization: Deprecated. See :func:`double_factorized_t2_compressed`.
        multi_stage_start: Deprecated. See :func:`double_factorized_t2_compressed`.
        multi_stage_step: Deprecated. See :func:`double_factorized_t2_compressed`.
        return_optimize_result: Deprecated. Whether to also return the
            :class:`~scipy.optimize.OptimizeResult` returned by
            :func:`scipy.optimize.minimize`. If no optimization is performed,
            ``None`` is returned in its place.
        optimize_kwargs: Deprecated. Keyword arguments to pass to
            :func:`scipy.optimize.minimize` when ``optimize`` is set to ``True``.

    Returns:
        The diagonal Coulomb matrices and the orbital rotations. Each list of matrices
        is collected into a Numpy array, so this method returns a tuple of two Numpy
        arrays, the first containing the diagonal Coulomb matrices and the second
        containing the orbital rotations. Each Numpy array will have shape (L, n, n)
        where L is the number of terms in the decomposition and n is the number of
        orbitals. If ``return_optimize_result`` is set to ``True``, the
        :class:`~scipy.optimize.OptimizeResult` returned by
        :func:`scipy.optimize.minimize` is also returned, or ``None`` if no
        optimization was performed. No optimization is performed if ``optimize`` is
        set to ``False``.

    Raises:
        ValueError: The t2 amplitudes are complex-valued.
        ValueError: max_terms is less than 1.
        ValueError: ``optimize`` is True and multi_stage_start or multi_stage_step
            is less than 1.
        ValueError: ``optimize`` is True and the exact factorization has no terms.
        ValueError: ``optimize`` is True and diag_coulomb_indices contains lower
            triangular indices.
    """
    _warn_deprecated_optimize_args(
        "double_factorized_t2",
        "double_factorized_t2_compressed",
        {
            "optimize": optimize,
            "diag_coulomb_indices": diag_coulomb_indices is not None,
            "regularization": regularization != 0,
            "multi_stage_start": multi_stage_start is not None,
            "multi_stage_step": multi_stage_step is not None,
            "return_optimize_result": return_optimize_result,
            "optimize_kwargs": bool(optimize_kwargs),
        },
    )
    _validate_real_t2_amplitudes(t2_amplitudes)
    if max_terms is not None and max_terms < 1:
        raise ValueError(f"max_terms must be at least 1. Got {max_terms}.")
    if optimize:
        diag_coulomb_mats, orbital_rotations = _double_factorized_t2_explicit(
            t2_amplitudes, tol=tol
        )
        diag_coulomb_mats, orbital_rotations, result = double_factorized_t2_compressed(
            t2_amplitudes,
            initial_guess=(diag_coulomb_mats, orbital_rotations),
            max_terms=max_terms,
            diag_coulomb_indices=diag_coulomb_indices,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
            regularization=regularization,
            **optimize_kwargs,
        )
    else:
        diag_coulomb_mats, orbital_rotations = _double_factorized_t2_explicit(
            t2_amplitudes, tol=tol, max_terms=max_terms
        )
        result = None
    if return_optimize_result:
        return diag_coulomb_mats, orbital_rotations, result
    return diag_coulomb_mats, orbital_rotations


def _double_factorized_t2_explicit(
    t2_amplitudes: np.ndarray, *, tol: float = 1e-8, max_terms: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact double factorization of t2 amplitudes.

    Each eigenvalue of the t2 amplitudes matrix gives rise to 2 terms, and the
    Frobenius norm of each term's diagonal Coulomb matrix is the absolute value of
    its eigenvalue. Terms are sorted in decreasing order of this norm.
    """
    nocc, _, nvrt, _ = t2_amplitudes.shape
    norb = nocc + nvrt

    t2_mat = t2_amplitudes.transpose(0, 2, 1, 3).reshape(nocc * nvrt, nocc * nvrt)
    outer_eigs, outer_vecs = _truncated_eigh(t2_mat, tol=tol)

    n_vecs = len(outer_eigs)

    one_body_tensors = np.zeros((n_vecs, 2, norb, norb), dtype=complex)
    for outer_vec, one_body_tensor in zip(outer_vecs.T, one_body_tensors):
        mat = np.zeros((norb, norb))
        mat[nocc:, :nocc] = outer_vec.reshape(nocc, nvrt).T
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


@functools.cache
def _make_t2_compressed_value_and_grad(
    n_tensors: int,
    norb: int,
    nocc: int,
    diag_coulomb_indices: tuple[tuple[int, int], ...] | None,
):
    """Build a jitted value-and-gradient function for the t2 compression objective.

    The result is cached and reused across optimizer iterations and across calls with
    the same static structure ``(n_tensors, norb, nocc, diag_coulomb_indices)``, so the
    loss is traced and compiled only once per structure. The target amplitudes,
    regularization coefficient, and reference diagonal Coulomb norm are passed as
    runtime arguments (they do not affect the traced graph), and the gradient is taken
    with respect to the flat variable vector ``x`` only.
    """
    indices = None if diag_coulomb_indices is None else list(diag_coulomb_indices)

    def loss(
        x: jax.Array,
        t2_amplitudes: jax.Array,
        regularization: jax.Array,
        init_diag_coulomb_norm: jax.Array,
    ) -> jax.Array:
        diag_coulomb_mats, orbital_rotations = df_tensors_from_params_jax(
            x, n_tensors, norb, indices
        )
        reconstructed = (
            1j
            * jnp.einsum(
                "mpq,map,mip,mbq,mjq->ijab",
                diag_coulomb_mats,
                orbital_rotations,
                orbital_rotations.conj(),
                orbital_rotations,
                orbital_rotations.conj(),
                optimize=True,
            )[:nocc, :nocc, nocc:, nocc:]
        )
        loss_val = 0.5 * jnp.sum(jnp.abs(reconstructed - t2_amplitudes) ** 2)
        diag_coulomb_norm = jnp.sum(jnp.abs(diag_coulomb_mats) ** 2)
        loss_val += regularization * jnp.abs(diag_coulomb_norm - init_diag_coulomb_norm)
        return loss_val

    return jax.jit(jax.value_and_grad(loss, argnums=0))


def double_factorized_t2_compressed(
    t2_amplitudes: np.ndarray,
    *,
    initial_guess: tuple[np.ndarray, np.ndarray] | None = None,
    max_terms: int | None = None,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    regularization: float = 0,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]:
    r"""Compress a double-factorized decomposition of :math:`t_2` amplitudes.

    Optimizes the diagonal Coulomb matrices and orbital rotations of a
    double-factorized decomposition of :math:`t_2` amplitudes
    (see :func:`double_factorized_t2`) with :func:`scipy.optimize.minimize` to reduce
    the error in the factorization. The optimization starts from the factorization
    given by ``initial_guess``. If it is not specified, the exact factorization
    computed by :func:`double_factorized_t2` with the default tolerance is used. To
    use a different tolerance, call :func:`double_factorized_t2` yourself, without
    ``max_terms`` so that the optimization starts from the full factorization, and
    pass its output as ``initial_guess``. If ``max_terms`` is specified,
    then the factorization is truncated to its first ``max_terms`` terms. The
    optimization is always performed, even if no terms are truncated.

    The diagonal Coulomb matrices returned by the optimization can be
    optionally constrained to have only certain elements allowed to be nonzero.
    This is achieved by passing the ``diag_coulomb_indices`` parameter, which is a
    list of matrix entry indices (integer pairs) specifying where the diagonal Coulomb
    matrices are allowed to be nonzero. Since the diagonal Coulomb matrices are
    symmetric, only upper triangular indices should be given, i.e.,
    pairs :math:`(i, j)` where :math:`i \leq j`.

    A "multi-stage" optimization can be requested by passing either
    ``multi_stage_start`` or ``multi_stage_step``. In multi-stage optimization, the
    number of terms is iteratively reduced by ``multi_stage_step``, starting from
    ``multi_stage_start``, until there are only ``max_terms`` remaining. This procedure
    can yield a more accurate factorization at the cost of increased running time. If
    you only pass one of ``multi_stage_start`` or ``multi_stage_step``, then the value
    of the other is chosen automatically: ``multi_stage_start`` defaults to the number
    of terms in the initial guess, and ``multi_stage_step`` defaults to 1. If
    ``max_terms`` is not specified or is at least the number of terms in the initial
    guess, a single stage of optimization is run with all of the terms, and these
    arguments have no effect.

    Note: Currently, only real-valued :math:`t_2` amplitudes are supported.

    Args:
        t2_amplitudes: The doubles amplitudes tensor of shape
            ``(nocc, nocc, nvrt, nvrt)``, where ``nocc`` is the number of occupied
            orbitals and ``nvrt`` is the number of virtual orbitals.
        initial_guess: The initial guess for the optimization, as a tuple of the
            diagonal Coulomb matrices and the orbital rotations, each a Numpy array
            of shape ``(L, norb, norb)``. Typically, this is the untruncated output
            of :func:`double_factorized_t2`. If not specified, the exact
            factorization computed by :func:`double_factorized_t2` with the default
            tolerance is used.
        max_terms: An optional limit on the number of terms to keep in the
            decomposition. If not specified, all the terms of the initial guess are
            kept.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices. Matrix entries corresponding to indices not in this
            list will be set to zero. This list should contain only upper
            triangular indices, i.e., pairs :math:`(i, j)` where :math:`i \leq j`.
        regularization: Parameter for regularizing the norm of the diagonal Coulomb
            matrices. This specifies the coefficient to a term added to the loss
            function equal to the difference between the sums of the squares of the
            Frobenius norms of the diagonal Coulomb matrices from the optimized
            factorization and the initial guess.
        multi_stage_start: The number of terms to start multi-stage optimization.
        multi_stage_step: The number of terms to eliminate in each stage of multi-stage
            optimization.
        optimize_kwargs: Keyword arguments to pass to
            :func:`scipy.optimize.minimize`. The default method is
            ``"L-BFGS-B"``.

    Returns:
        - The optimized diagonal Coulomb matrices, as a Numpy array of shape
          ``(L, norb, norb)``.
        - The optimized orbital rotations, as a Numpy array of shape
          ``(L, norb, norb)``.
        - The :class:`~scipy.optimize.OptimizeResult` returned by
          :func:`scipy.optimize.minimize` in the last stage of the optimization.

    Raises:
        ValueError: The t2 amplitudes are complex-valued.
        ValueError: max_terms is less than 1.
        ValueError: multi_stage_start or multi_stage_step is less than 1.
        ValueError: diag_coulomb_indices contains lower triangular indices.
        ValueError: The initial guess has no terms. This includes the case where
            ``initial_guess`` is not specified and the exact factorization has no
            terms.
        ValueError: The shapes of the initial guess tensors are inconsistent.
    """
    _validate_real_t2_amplitudes(t2_amplitudes)
    if max_terms is not None and max_terms < 1:
        raise ValueError(f"max_terms must be at least 1. Got {max_terms}.")
    _validate_multi_stage_args(multi_stage_start, multi_stage_step)
    _validate_diag_coulomb_indices(diag_coulomb_indices)
    if initial_guess is None:
        initial_guess = _double_factorized_t2_explicit(t2_amplitudes)
    diag_coulomb_mats, orbital_rotations = initial_guess
    nocc, _, nvrt, _ = t2_amplitudes.shape
    norb = nocc + nvrt
    _validate_df_tensor_shapes(
        diag_coulomb_mats, orbital_rotations, (norb, norb), (norb, norb)
    )
    diag_coulomb_mats, orbital_rotations, result = _compress_double_factorized_t2(
        t2_amplitudes,
        diag_coulomb_mats,
        orbital_rotations,
        max_terms=max_terms,
        diag_coulomb_indices=diag_coulomb_indices,
        multi_stage_start=multi_stage_start,
        multi_stage_step=multi_stage_step,
        regularization=regularization,
        **optimize_kwargs,
    )
    return diag_coulomb_mats, orbital_rotations, result


def _compress_double_factorized_t2(
    t2_amplitudes: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    *,
    max_terms: int | None,
    diag_coulomb_indices: list[tuple[int, int]] | None,
    multi_stage_start: int | None,
    multi_stage_step: int | None,
    regularization: float,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]:
    """Compress a double factorization of t2 amplitudes.

    The input tensors are used as the initial guess for the optimization. The
    factorization must have at least one term.
    """
    nocc, _, nvrt, _ = t2_amplitudes.shape
    norb = nocc + nvrt

    # Structure key for the cached jitted objective. Convert to a hashable tuple so it
    # can be used as a cache key, and back to a list inside the builder.
    indices_key = None if diag_coulomb_indices is None else tuple(diag_coulomb_indices)
    return _multi_stage_compress(
        t2_amplitudes,
        diag_coulomb_mats,
        orbital_rotations,
        max_terms=max_terms,
        multi_stage_start=multi_stage_start,
        multi_stage_step=multi_stage_step,
        regularization=regularization,
        make_value_and_grad=lambda n_tensors: _make_t2_compressed_value_and_grad(
            n_tensors, norb, nocc, indices_key
        ),
        to_params=lambda mats, rots: df_tensors_to_params(
            mats, rots, diag_coulomb_indices
        ),
        from_params=lambda x, n_tensors: df_tensors_from_params(
            x, n_tensors, norb, diag_coulomb_indices
        ),
        **optimize_kwargs,
    )


@overload
def double_factorized_t2_alpha_beta(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = ...,
    max_terms: int | None = ...,
    optimize: bool = ...,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = ...,
    regularization: float = ...,
    multi_stage_start: int | None = ...,
    multi_stage_step: int | None = ...,
    return_optimize_result: Literal[False] = False,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def double_factorized_t2_alpha_beta(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = ...,
    max_terms: int | None = ...,
    optimize: bool = ...,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = ...,
    regularization: float = ...,
    multi_stage_start: int | None = ...,
    multi_stage_step: int | None = ...,
    return_optimize_result: Literal[True],
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult | None]: ...
@overload
def double_factorized_t2_alpha_beta(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = ...,
    max_terms: int | None = ...,
    optimize: bool = ...,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = ...,
    regularization: float = ...,
    multi_stage_start: int | None = ...,
    multi_stage_step: int | None = ...,
    return_optimize_result: bool = ...,
    **optimize_kwargs,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult | None]
): ...
def double_factorized_t2_alpha_beta(
    t2_amplitudes: np.ndarray,
    *,
    tol: float = 1e-8,
    max_terms: int | None = None,
    optimize: bool = False,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    regularization: float = 0,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    return_optimize_result: bool = False,
    **optimize_kwargs,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult | None]
):
    r"""Double-factorized decomposition of alpha-beta :math:`t_2` amplitudes.

    Decompose alpha-beta :math:`t_2` amplitudes into diagonal Coulomb matrices with
    orbital rotations. This function returns two arrays:

    - ``diagonal_coulomb_mats``, with shape ``(n_terms, 3, norb, norb)``.
    - ``orbital_rotations``, with shape ``(n_terms, 2, norb, norb)``.

    The value of ``n_terms`` depends on the error tolerance ``tol``. A larger error
    tolerance might yield a smaller value for ``n_terms``. You can also set an optional
    upper bound on ``n_terms`` using the ``max_terms`` argument. To further optimize
    the resulting tensors to reduce the error in the factorization, pass the
    untruncated factorization to :func:`double_factorized_t2_alpha_beta_compressed`.

    .. warning::
        The ``optimize``, ``diag_coulomb_indices``, ``regularization``,
        ``multi_stage_start``, ``multi_stage_step``, ``return_optimize_result``, and
        ``optimize_kwargs`` arguments are deprecated. Instead, call
        :func:`double_factorized_t2_alpha_beta_compressed` on the output of this
        function.

    The original :math:`t_2` amplitudes tensor can be reconstructed, up to the error
    tolerance, using the following function:

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
                1j
                * np.einsum(
                    "kpq,kap,kip,kbq,kjq->ijab",
                    expanded_diag_coulomb_mats,
                    expanded_orbital_rotations,
                    expanded_orbital_rotations.conj(),
                    expanded_orbital_rotations,
                    expanded_orbital_rotations.conj(),
                )[:nocc_a, norb : norb + nocc_b, nocc_a:norb, norb + nocc_b :]
            )

    Note: Currently, only real-valued :math:`t_2` amplitudes are supported.

    Args:
        t2_amplitudes: The doubles amplitudes tensor of shape
            ``(nocc_a, nocc_b, nvrt_a, nvrt_b)``, where ``nocc_a`` is the number of
            occupied spin-alpha orbitals, ``nvrt_a`` is the number of virtual spin-alpha
            orbitals, ``nocc_b`` is the number of occupied spin-beta orbitals, and
            ``nvrt_b`` is the number of virtual spin-beta orbitals.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_terms: An optional limit on the number of terms to keep in the decomposition
            of the :math:`t_2` amplitudes tensor. This argument overrides ``tol``.
        optimize: Deprecated. Whether to optimize the tensors returned by the
            decomposition to minimize the error in the factorization.
        diag_coulomb_indices: Deprecated. See
            :func:`double_factorized_t2_alpha_beta_compressed`.
        regularization: Deprecated. See
            :func:`double_factorized_t2_alpha_beta_compressed`.
        multi_stage_start: Deprecated. See
            :func:`double_factorized_t2_alpha_beta_compressed`.
        multi_stage_step: Deprecated. See
            :func:`double_factorized_t2_alpha_beta_compressed`.
        return_optimize_result: Deprecated. Whether to also return the
            :class:`~scipy.optimize.OptimizeResult` returned by
            :func:`scipy.optimize.minimize`. If no optimization is performed,
            ``None`` is returned in its place.
        optimize_kwargs: Deprecated. Keyword arguments to pass to
            :func:`scipy.optimize.minimize` when ``optimize`` is set to ``True``.

    Returns:
        - The diagonal Coulomb matrices, as a Numpy array of shape
          ``(n_terms, 3, norb, norb)``.
          The last two axes index the rows and columns of
          the matrices, and the third from last axis, which has 3 dimensions, indexes
          the spin interaction type of the matrix: alpha-alpha, alpha-beta, and
          beta-beta (in that order).
          The first axis indexes the terms of the decomposition.
        - The orbital rotations, as a Numpy array of shape
          ``(n_terms, 2, norb, norb)``. The last two axes index the rows and columns of
          the orbital rotations, and the third from last axis, which has 2 dimensions,
          indexes the spin sector of the orbital rotation: first alpha, then beta.
          The first axis indexes the terms of the decomposition.
        - If ``return_optimize_result`` is set to ``True``, the
          :class:`~scipy.optimize.OptimizeResult` returned by
          :func:`scipy.optimize.minimize`, or ``None`` if no optimization was
          performed. No optimization is performed if ``optimize`` is set to ``False``.

    Raises:
        ValueError: The t2 amplitudes are complex-valued.
        ValueError: max_terms is less than 1.
        ValueError: ``optimize`` is True and multi_stage_start or multi_stage_step
            is less than 1.
        ValueError: ``optimize`` is True and the exact factorization has no terms.
        ValueError: ``optimize`` is True and the alpha-alpha or beta-beta
            diag_coulomb_indices contain lower triangular indices.
    """
    _warn_deprecated_optimize_args(
        "double_factorized_t2_alpha_beta",
        "double_factorized_t2_alpha_beta_compressed",
        {
            "optimize": optimize,
            "diag_coulomb_indices": diag_coulomb_indices is not None,
            "regularization": regularization != 0,
            "multi_stage_start": multi_stage_start is not None,
            "multi_stage_step": multi_stage_step is not None,
            "return_optimize_result": return_optimize_result,
            "optimize_kwargs": bool(optimize_kwargs),
        },
    )
    _validate_real_t2_amplitudes(t2_amplitudes)
    if max_terms is not None and max_terms < 1:
        raise ValueError(f"max_terms must be at least 1. Got {max_terms}.")
    if optimize:
        diag_coulomb_mats, orbital_rotations = (
            _double_factorized_t2_alpha_beta_explicit(t2_amplitudes, tol=tol)
        )
        diag_coulomb_mats, orbital_rotations, result = (
            double_factorized_t2_alpha_beta_compressed(
                t2_amplitudes,
                initial_guess=(diag_coulomb_mats, orbital_rotations),
                max_terms=max_terms,
                diag_coulomb_indices=diag_coulomb_indices,
                multi_stage_start=multi_stage_start,
                multi_stage_step=multi_stage_step,
                regularization=regularization,
                **optimize_kwargs,
            )
        )
    else:
        diag_coulomb_mats, orbital_rotations = (
            _double_factorized_t2_alpha_beta_explicit(
                t2_amplitudes, tol=tol, max_terms=max_terms
            )
        )
        result = None
    if return_optimize_result:
        return diag_coulomb_mats, orbital_rotations, result
    return diag_coulomb_mats, orbital_rotations


@functools.cache
def _make_t2_alpha_beta_compressed_value_and_grad(
    n_tensors: int,
    norb: int,
    nocc_a: int,
    nocc_b: int,
    diag_coulomb_indices_key: tuple[
        tuple[tuple[int, int], ...] | None,
        tuple[tuple[int, int], ...] | None,
        tuple[tuple[int, int], ...] | None,
    ]
    | None,
):
    """Build a jitted value-and-grad for the alpha-beta t2 compression objective.

    The result is cached and reused across optimizer iterations and across calls with
    the same static structure, so the loss is traced and compiled only once per
    structure. The target amplitudes, regularization coefficient, and reference
    diagonal Coulomb norm are passed as runtime arguments (they do not affect the
    traced graph), and the gradient is taken with respect to the flat variable vector
    ``x`` only.
    """
    if diag_coulomb_indices_key is None:
        diag_coulomb_indices: (
            tuple[
                list[tuple[int, int]] | None,
                list[tuple[int, int]] | None,
                list[tuple[int, int]] | None,
            ]
            | None
        ) = None
    else:
        diag_coulomb_indices = tuple(  # type: ignore[assignment]
            None if pairs is None else list(pairs) for pairs in diag_coulomb_indices_key
        )

    def loss(
        x: jax.Array,
        t2_amplitudes: jax.Array,
        regularization: jax.Array,
        init_diag_coulomb_norm: jax.Array,
    ) -> jax.Array:
        diag_coulomb_mats, orbital_rotations = df_tensors_alpha_beta_from_params_jax(
            x, n_tensors, norb, diag_coulomb_indices
        )
        mats_aa = diag_coulomb_mats[:, 0]
        mats_ab = diag_coulomb_mats[:, 1]
        mats_bb = diag_coulomb_mats[:, 2]
        top = jnp.concatenate([mats_aa, mats_ab], axis=-1)
        bottom = jnp.concatenate([mats_ab.transpose(0, 2, 1), mats_bb], axis=-1)
        expanded_diag_coulomb_mats = jnp.concatenate([top, bottom], axis=-2)

        rot_a = orbital_rotations[:, 0]
        rot_b = orbital_rotations[:, 1]
        zeros = jnp.zeros((n_tensors, norb, norb), dtype=orbital_rotations.dtype)
        rot_top = jnp.concatenate([rot_a, zeros], axis=-1)
        rot_bottom = jnp.concatenate([zeros, rot_b], axis=-1)
        expanded_orbital_rotations = jnp.concatenate([rot_top, rot_bottom], axis=-2)

        reconstructed = (
            1j
            * jnp.einsum(
                "kpq,kap,kip,kbq,kjq->ijab",
                expanded_diag_coulomb_mats,
                expanded_orbital_rotations,
                expanded_orbital_rotations.conj(),
                expanded_orbital_rotations,
                expanded_orbital_rotations.conj(),
                optimize=True,
            )[:nocc_a, norb : norb + nocc_b, nocc_a:norb, norb + nocc_b :]
        )
        loss_val = 0.5 * jnp.sum(jnp.abs(reconstructed - t2_amplitudes) ** 2)
        diag_coulomb_norm = jnp.sum(jnp.abs(diag_coulomb_mats) ** 2)
        loss_val += regularization * jnp.abs(diag_coulomb_norm - init_diag_coulomb_norm)
        return loss_val

    return jax.jit(jax.value_and_grad(loss, argnums=0))


def double_factorized_t2_alpha_beta_compressed(
    t2_amplitudes: np.ndarray,
    *,
    initial_guess: tuple[np.ndarray, np.ndarray] | None = None,
    max_terms: int | None = None,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    regularization: float = 0,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]:
    r"""Compress a double-factorized decomposition of alpha-beta :math:`t_2` amplitudes.

    Optimizes the diagonal Coulomb matrices and orbital rotations of a
    double-factorized decomposition of alpha-beta :math:`t_2` amplitudes
    (see :func:`double_factorized_t2_alpha_beta`) with :func:`scipy.optimize.minimize`
    to reduce the error in the factorization. The optimization starts from the
    factorization given by ``initial_guess``. If it is not specified, the exact
    factorization computed by :func:`double_factorized_t2_alpha_beta` with the default
    tolerance is used. To use a different tolerance, call
    :func:`double_factorized_t2_alpha_beta` yourself, without ``max_terms`` so that
    the optimization starts from the full factorization, and pass its output as
    ``initial_guess``. If ``max_terms`` is specified,
    then the factorization is truncated to its first ``max_terms`` terms. The
    optimization is always performed, even if no terms are truncated.

    See :func:`double_factorized_t2_compressed` for a description of multi-stage
    optimization.

    Note: Currently, only real-valued :math:`t_2` amplitudes are supported.

    Args:
        t2_amplitudes: The doubles amplitudes tensor of shape
            ``(nocc_a, nocc_b, nvrt_a, nvrt_b)``, where ``nocc_a`` is the number of
            occupied spin-alpha orbitals, ``nvrt_a`` is the number of virtual spin-alpha
            orbitals, ``nocc_b`` is the number of occupied spin-beta orbitals, and
            ``nvrt_b`` is the number of virtual spin-beta orbitals.
        initial_guess: The initial guess for the optimization, as a tuple of the
            diagonal Coulomb matrices, as a Numpy array of shape
            ``(n_terms, 3, norb, norb)``, and the orbital rotations, as a Numpy array
            of shape ``(n_terms, 2, norb, norb)``. Typically, this is the untruncated
            output of :func:`double_factorized_t2_alpha_beta`. If not specified, the
            exact factorization computed by :func:`double_factorized_t2_alpha_beta`
            with the default tolerance is used.
        max_terms: An optional limit on the number of terms to keep in the
            decomposition. If not specified, all the terms of the initial guess are
            kept.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices.
            If specified, should be a tuple of 3 lists, for alpha-alpha, alpha-beta,
            and beta-beta matrices, in that order.
            For the alpha-alpha and beta-beta matrices, each index must be
            upper triangular, that is, of the form :math:`(i, j)` where
            :math:`i \leq j`.
        regularization: Parameter for regularizing the norm of the diagonal Coulomb
            matrices. This specifies the coefficient to a term added to the loss
            function equal to the difference between the sums of the squares of the
            Frobenius norms of the diagonal Coulomb matrices from the optimized
            factorization and the initial guess.
        multi_stage_start: The number of terms to start multi-stage optimization.
        multi_stage_step: The number of terms to eliminate in each stage of multi-stage
            optimization.
        optimize_kwargs: Keyword arguments to pass to
            :func:`scipy.optimize.minimize`. The default method is
            ``"L-BFGS-B"``.

    Returns:
        - The optimized diagonal Coulomb matrices, as a Numpy array of shape
          ``(n_terms, 3, norb, norb)``.
        - The optimized orbital rotations, as a Numpy array of shape
          ``(n_terms, 2, norb, norb)``.
        - The :class:`~scipy.optimize.OptimizeResult` returned by
          :func:`scipy.optimize.minimize` in the last stage of the optimization.

    Raises:
        ValueError: The t2 amplitudes are complex-valued.
        ValueError: max_terms is less than 1.
        ValueError: multi_stage_start or multi_stage_step is less than 1.
        ValueError: The alpha-alpha or beta-beta diag_coulomb_indices contain lower
            triangular indices.
        ValueError: The initial guess has no terms. This includes the case where
            ``initial_guess`` is not specified and the exact factorization has no
            terms.
        ValueError: The shapes of the initial guess tensors are inconsistent.
    """
    _validate_real_t2_amplitudes(t2_amplitudes)
    if max_terms is not None and max_terms < 1:
        raise ValueError(f"max_terms must be at least 1. Got {max_terms}.")
    _validate_multi_stage_args(multi_stage_start, multi_stage_step)
    if diag_coulomb_indices is not None:
        # Alpha-beta indices may be lower triangular because the alpha-beta matrices
        # are not symmetric
        pairs_aa, _, pairs_bb = diag_coulomb_indices
        _validate_diag_coulomb_indices(pairs_aa)
        _validate_diag_coulomb_indices(pairs_bb)
    if initial_guess is None:
        initial_guess = _double_factorized_t2_alpha_beta_explicit(t2_amplitudes)
    diag_coulomb_mats, orbital_rotations = initial_guess
    nocc_a, _, nvrt_a, _ = t2_amplitudes.shape
    norb = nocc_a + nvrt_a
    _validate_df_tensor_shapes(
        diag_coulomb_mats, orbital_rotations, (3, norb, norb), (2, norb, norb)
    )
    diag_coulomb_mats, orbital_rotations, result = (
        _compress_double_factorized_t2_alpha_beta(
            t2_amplitudes,
            diag_coulomb_mats,
            orbital_rotations,
            max_terms=max_terms,
            diag_coulomb_indices=diag_coulomb_indices,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
            regularization=regularization,
            **optimize_kwargs,
        )
    )
    return diag_coulomb_mats, orbital_rotations, result


def _compress_double_factorized_t2_alpha_beta(
    t2_amplitudes: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    *,
    max_terms: int | None,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None,
    multi_stage_start: int | None,
    multi_stage_step: int | None,
    regularization: float,
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]:
    """Compress a double factorization of alpha-beta t2 amplitudes.

    The input tensors are used as the initial guess for the optimization. The
    factorization must have at least one term.
    """
    nocc_a, nocc_b, nvrt_a, _ = t2_amplitudes.shape
    norb = nocc_a + nvrt_a

    # Hashable structure key for the cached jitted objective. Convert the (possibly
    # nested) index lists to tuples so they can serve as a cache key, and back to
    # lists inside the builder.
    if diag_coulomb_indices is None:
        indices_key: tuple | None = None
    else:
        indices_key = tuple(
            None if pairs is None else tuple(pairs) for pairs in diag_coulomb_indices
        )
    return _multi_stage_compress(
        t2_amplitudes,
        diag_coulomb_mats,
        orbital_rotations,
        max_terms=max_terms,
        multi_stage_start=multi_stage_start,
        multi_stage_step=multi_stage_step,
        regularization=regularization,
        make_value_and_grad=lambda n_tensors: (
            _make_t2_alpha_beta_compressed_value_and_grad(
                n_tensors, norb, nocc_a, nocc_b, indices_key
            )
        ),
        to_params=lambda mats, rots: df_tensors_alpha_beta_to_params(
            mats, rots, diag_coulomb_indices
        ),
        from_params=lambda x, n_tensors: df_tensors_alpha_beta_from_params(
            x, n_tensors, norb, diag_coulomb_indices
        ),
        **optimize_kwargs,
    )


def _multi_stage_compress(
    t2_amplitudes: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    *,
    max_terms: int | None,
    multi_stage_start: int | None,
    multi_stage_step: int | None,
    regularization: float,
    make_value_and_grad: Callable[[int], Callable],
    to_params: Callable[[np.ndarray, np.ndarray], np.ndarray],
    from_params: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]],
    **optimize_kwargs,
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]:
    """Compress a double factorization of t2 amplitudes in one or more stages.

    Each stage truncates the current factorization to fewer terms and optimizes it,
    starting from the result of the previous stage. The loss, parameterization, and
    term structure are supplied by the caller. If ``max_terms`` is not specified or is
    at least the number of terms, a single stage is run with all of the terms. The
    factorization must have at least one term.
    """
    init_diag_coulomb_norm = np.sum(np.abs(diag_coulomb_mats) ** 2)
    n_terms_full = len(orbital_rotations)

    if max_terms is None or max_terms >= n_terms_full:
        list_reps = [n_terms_full]
    elif multi_stage_start is None and multi_stage_step is None:
        list_reps = [max_terms]
    else:
        if multi_stage_start is None:
            multi_stage_start = n_terms_full
        multi_stage_start = min(n_terms_full, multi_stage_start)
        if multi_stage_step is None:
            multi_stage_step = 1
        list_reps = list(range(multi_stage_start, max_terms, -multi_stage_step))
        list_reps.append(max_terms)

    t2_jax = jnp.asarray(t2_amplitudes)
    regularization_jax = jnp.asarray(regularization)
    init_diag_coulomb_norm_jax = jnp.asarray(init_diag_coulomb_norm)
    optimize_kwargs.setdefault("method", "L-BFGS-B")

    for n_tensors in list_reps:
        diag_coulomb_mats = diag_coulomb_mats[:n_tensors]
        orbital_rotations = orbital_rotations[:n_tensors]

        # Reuse a jitted value-and-gradient function cached by static structure, so the
        # loss is compiled once and reused across all optimizer iterations (and across
        # calls/stages with the same structure) instead of being re-traced eagerly each
        # step.
        value_and_grad = make_value_and_grad(n_tensors)

        def scipy_func(x: np.ndarray) -> tuple[float, np.ndarray]:
            value, grad = value_and_grad(
                jnp.asarray(x),
                t2_jax,
                regularization_jax,
                init_diag_coulomb_norm_jax,
            )
            return float(value), np.asarray(grad)

        result = scipy.optimize.minimize(
            scipy_func,
            to_params(diag_coulomb_mats, orbital_rotations),
            jac=True,
            **optimize_kwargs,
        )
        diag_coulomb_mats, orbital_rotations = from_params(result.x, n_tensors)

    return diag_coulomb_mats, orbital_rotations, result


def _double_factorized_t2_alpha_beta_explicit(
    t2_amplitudes: np.ndarray, *, tol: float = 1e-8, max_terms: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact double factorization of alpha-beta t2 amplitudes.

    Each singular value of the t2 amplitudes matrix gives rise to 4 terms. The
    Frobenius norm of each term's diagonal Coulomb matrix in the spin-orbital basis,
    whose blocks are the alpha-alpha, alpha-beta, and beta-beta matrices, is its
    singular value. Terms are sorted in decreasing order of this norm.
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
        left_mat[nocc_a:, :nocc_a] = left_vec.reshape(nocc_a, nvrt_a).T
        right_mat = np.zeros((norb, norb))
        right_mat[nocc_b:, :nocc_b] = right_vec.reshape(nocc_b, nvrt_b).T
        these_one_body_tensors[0, :, 0] = _quadrature(left_mat, 1)
        these_one_body_tensors[0, 0, 1] = _quadrature(right_mat, 1)
        these_one_body_tensors[0, 1, 1] = _quadrature(-right_mat, 1)
        these_one_body_tensors[1, :, 0] = _quadrature(left_mat, -1)
        these_one_body_tensors[1, 0, 1] = _quadrature(right_mat, -1)
        these_one_body_tensors[1, 1, 1] = _quadrature(-right_mat, -1)
    one_body_tensors = one_body_tensors.reshape((n_vecs, 4, 2, norb, norb))

    eigs, orbital_rotations = np.linalg.eigh(one_body_tensors)
    eigs = np.concatenate([eigs[:, :, 0], eigs[:, :, 1]], axis=-1)
    coeffs = 0.5 * np.array([1, -1, -1, 1]) * singular_vals[:, None]
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


def reconstruct_t2(
    diag_coulomb_mats: np.ndarray, orbital_rotations: np.ndarray, nocc: int
) -> np.ndarray:
    """Reconstruct t2 amplitudes from double factorization tensors."""
    return (
        1j
        * contract(
            "kpq,kap,kip,kbq,kjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )


def reconstruct_t2_alpha_beta(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    norb: int,
    nocc_a: int,
    nocc_b: int,
) -> np.ndarray:
    """Reconstruct alpha-beta t2 amplitudes from double factorization tensors."""
    n_terms = diag_coulomb_mats.shape[0]
    expanded_diag_coulomb_mats = np.zeros((n_terms, 2 * norb, 2 * norb))
    expanded_orbital_rotations = np.zeros((n_terms, 2 * norb, 2 * norb), dtype=complex)
    for k in range(n_terms):
        (mat_aa, mat_ab, mat_bb) = diag_coulomb_mats[k]
        expanded_diag_coulomb_mats[k] = np.block([[mat_aa, mat_ab], [mat_ab.T, mat_bb]])
        orbital_rotation_a, orbital_rotation_b = orbital_rotations[k]
        expanded_orbital_rotations[k] = scipy.linalg.block_diag(
            orbital_rotation_a, orbital_rotation_b
        )
    return (
        1j
        * contract(
            "kpq,kap,kip,kbq,kjq->ijab",
            expanded_diag_coulomb_mats,
            expanded_orbital_rotations,
            expanded_orbital_rotations.conj(),
            expanded_orbital_rotations,
            expanded_orbital_rotations.conj(),
        )[:nocc_a, norb : norb + nocc_b, nocc_a:norb, norb + nocc_b :]
    )
