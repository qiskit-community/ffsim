# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for performing the compressed double-factorized decomposition."""

from __future__ import annotations

from typing import Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from opt_einsum import contract

from ffsim.linalg import double_factorized_t2
from ffsim.linalg.util import (
    df_tensors_from_params,
    df_tensors_from_params_jax,
    df_tensors_to_params,
)


@overload
def double_factorized_t2_compressed(
    t2: np.ndarray,
    *,
    tol: float = ...,
    n_reps: int | None = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    method: str = ...,
    callback=...,
    options: dict | None = ...,
    multi_stage_optimization: bool = ...,
    begin_reps: int | None = ...,
    step: int = ...,
    return_optimize_result: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def double_factorized_t2_compressed(
    t2: np.ndarray,
    *,
    tol: float = ...,
    n_reps: int | None = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    method: str = ...,
    callback=...,
    options: dict | None = ...,
    multi_stage_optimization: bool = ...,
    begin_reps: int | None = ...,
    step: int = ...,
    return_optimize_result: Literal[True],
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]: ...
def double_factorized_t2_compressed(
    t2: np.ndarray,
    *,
    tol: float = 1e-8,
    n_reps: int | None = None,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    multi_stage_optimization: bool = False,
    begin_reps: int | None = None,
    step: int = 2,
    return_optimize_result: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]
):
    r"""Compressed double-factorized decomposition of t2 amplitudes for LUCJ ansatz.

    The double-factorized decomposition of a t2 amplitudes tensor :math:`t_{ijab}` is

    .. math::

        t_{ijab} = i \sum_{m=1}^L \sum_{k=1}^2 \sum_{pq}
            Z^{(mk)}_{pq}
            U^{(mk)}_{ap} U^{(mk)*}_{ip} U^{(mk)}_{bq} U^{(mk)*}_{jq}

    Here each :math:`Z^{(mk)}` is a real-valued matrix, referred to as a
    "diagonal Coulomb matrix," and each :math:`U^{(mk)}` is a unitary matrix,
    referred to as an "orbital rotation."

    The number of terms :math:`L` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    After decomposition, the goal is to compress the operator down to `n_reps` terms
    while minimizing the difference with the original t2 amplitude with a least-squares
    objective function. This is achieved by first truncating the operator and then
    apply optimizer to minimize the coefficients in the remaining operator.

    Note: Currently, only real-valued t2 amplitudes are supported.

    Args:
        t2: The t2 amplitudes tensor.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        n_reps: The number of ansatz repetitions. If not specified, then it is set
            to the number of terms resulting from the double-factorization of the
            t2 amplitudes. If the specified number of repetitions is larger than the
            number of terms resulting from the double-factorization, then the ansatz
            is padded with additional identity operators up to the specified number
            of repetitions.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices. Matrix entries corresponding to indices not in this
            list will be set to zero. This list should contain only upper
            trianglular indices, i.e., pairs :math:`(i, j)` where :math:`i \leq j`.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        multi_stage_optimization: Iteratively reduce the number of ansatz repetitions
            starting from full configuration if  `begin_reps` is not given. In each
            iteration, the number of repetitions is reduced by `step` until reaching
            `n_reps`.
        begin_reps: The starting point of the multi-stage optimization
        step: The step size for the multi-stage optimization
        return_optimize_result: Whether to also return the `OptimizeResult`_ returned
            by `scipy.optimize.minimize`_.

    Returns:
        - The diagonal Coulomb matrices, as a Numpy array of shape
          `(n_reps, norb, norb)`.
          The first axis indexes the eigenvectors of the decomposition, and he last two
          axes index the rows and columns of the matrices.
        - The orbital rotations, as a Numpy array of shape
          `(n_reps, norb, norb)`.
          The first axis indexes the eigenvectors of the decomposition, and he last two
          axes index the rows and columns of the matrices.
        - If `return_optimize_result` is set to ``True``, the `OptimizeResult`_
          returned by `scipy.optimize.minimize`_ is also returned.

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    nocc, _, nvrt, _ = t2.shape
    norb = nocc + nvrt

    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2, tol=tol)
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)
    n_reps_full, _, _ = orbital_rotations.shape

    if n_reps is None or n_reps_full < n_reps:
        return diag_coulomb_mats, orbital_rotations

    if multi_stage_optimization:
        if begin_reps is None:
            begin_reps = n_reps_full
        begin_reps = min(n_reps_full, begin_reps)
        list_reps = list(range(begin_reps, n_reps, -step))
        list_reps.append(n_reps)
    else:
        list_reps = [n_reps]

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
            diff = reconstructed - t2
            return 0.5 * jnp.sum(jnp.abs(diff) ** 2)

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
