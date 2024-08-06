# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear method for optimization of a variational ansatz."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from pyscf.lib.linalg_helper import safe_eigh
from scipy.optimize import OptimizeResult, minimize
from scipy.sparse.linalg import LinearOperator

from ffsim.optimize._util import (
    WrappedCallable,
    WrappedLinearOperator,
    jacobian_finite_diff,
    orthogonalize_columns,
)


def minimize_linear_method(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    hamiltonian: LinearOperator,
    x0: np.ndarray,
    *,
    maxiter: int = 1000,
    lindep: float = 1e-8,
    epsilon: float = 1e-8,
    ftol: float = 1e-8,
    gtol: float = 1e-5,
    regularization: float = 1e-4,
    variation: float = 0.5,
    optimize_regularization: bool = True,
    optimize_variation: bool = True,
    optimize_kwargs: dict | None = None,
    callback: Callable[[OptimizeResult], Any] | None = None,
) -> OptimizeResult:
    """Minimize the energy of a variational ansatz using the linear method.

    References:

    - `Implementation of the Linear Method for the optimization of Jastrow-Feenberg and Backflow Correlations`_

    Args:
        params_to_vec: Function representing the wavefunction ansatz. It takes as input
            a vector of real-valued parameters and outputs the state vector represented
            by those parameters.
        hamiltonian: The Hamiltonian representing the energy to be minimized.
        x0: Initial guess for the parameters.
        maxiter: Maximum number of optimization iterations to perform.
        lindep: Linear dependency threshold to use when solving the generalized
            eigenvalue problem.
        epsilon: Increment to use for approximating the gradient using
            finite difference.
        ftol: Convergence threshold for the objective function value. The optimization
            stops when `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol`.
        gtol: Convergence threshold for the gradient. The optimization stops when
            `max{|g_i| i = 1, ..., n} <= gtol`.
        regularization: Hyperparameter controlling regularization of the
            energy matrix. Its value must be positive. A larger value results in
            greater regularization.
        variation: Hyperparameter controlling the size of parameter variations
            used in the linear expansion of the wavefunction. Its value must be
            strictly between 0 and 1. A larger value results in larger variations.
        optimize_regularization: Whether to optimize the `regularization` hyperparameter
            in each iteration. Optimizing hyperparameters incurs more function and
            energy evaluations in each iteration, but may improve convergence.
            The optimization is performed using `scipy.optimize.minimize`_.
        optimize_variation: Whether to optimize the `variation` hyperparameter
            in each iteration. Optimizing hyperparameters incurs more function and
            energy evaluations in each iteration, but may improve convergence.
            The optimization is performed using `scipy.optimize.minimize`_.
        optimize_kwargs: Arguments to use when calling `scipy.optimize.minimize`_ to
            optimize hyperparameters. The call is constructed as

            .. code::

                scipy.optimize.minimize(f, x0, **optimize_kwargs)

            If not specified, the default value of `dict(method="L-BFGS-B")` will be
            used.

        callback: A callable called after each iteration. It is called with the
            signature

            .. code::

                callback(intermediate_result: OptimizeResult)

            where ``intermediate_result`` is a `scipy.optimize.OptimizeResult`_
            with attributes ``x``  and ``fun``, the present values of the parameter
            vector and objective function. For all iterations except for the last,
            it also contains the ``jac`` attribute holding the present value of the
            gradient, as well as ``regularization`` and ``variation`` attributes
            holding the present values of the `regularization` and `variation`
            hyperparameters.

    Returns:
        The optimization result represented as a `scipy.optimize.OptimizeResult`_
        object. Note the following definitions of selected attributes:

        - ``x``: The final parameters of the optimization.
        - ``fun``: The energy associated with the final parameters. That is, the
          expectation value of the Hamiltonian with respect to the state vector
          corresponding to the parameters.
        - ``nfev``: The number of times the ``params_to_vec`` function was called.
        - ``nlinop``: The number of times the ``hamiltonian`` linear operator was
          applied to a vector.

    .. _Implementation of the Linear Method for the optimization of Jastrow-Feenberg and Backflow Correlations: https://arxiv.org/abs/1412.0490
    .. _scipy.optimize.OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """  # noqa: E501
    if regularization < 0:
        raise ValueError(f"regularization must be nonnegative. Got {regularization}.")
    if not 0 <= variation <= 1:
        raise ValueError(f"variation must be between 0 and 1. Got {variation}.")
    if maxiter < 1:
        raise ValueError(f"maxiter must be at least 1. Got {maxiter}.")

    if optimize_kwargs is None:
        optimize_kwargs = dict(method="L-BFGS-B")

    params = x0.copy()
    success = False
    message = "Stop: Total number of iterations reached limit."
    intermediate_result = OptimizeResult(
        x=None, fun=None, jac=None, nfev=0, njev=0, nit=0, nlinop=0
    )

    params_to_vec = WrappedCallable(params_to_vec, intermediate_result)
    hamiltonian = WrappedLinearOperator(hamiltonian, intermediate_result)

    for i in range(maxiter):
        vec = params_to_vec(params)
        jac = jacobian_finite_diff(params_to_vec, params, len(vec), epsilon=epsilon)
        jac = orthogonalize_columns(jac, vec)

        energy_mat, overlap_mat = _linear_method_matrices(vec, jac, hamiltonian)
        energy = energy_mat[0, 0]
        grad = 2 * energy_mat[0, 1:]

        previous_energy = intermediate_result.fun
        intermediate_result.njev += 1

        if i > 0 and callback is not None:
            intermediate_result.x = params
            intermediate_result.fun = energy
            intermediate_result.jac = grad
            intermediate_result.energy_mat = energy_mat
            intermediate_result.overlap_mat = overlap_mat
            intermediate_result.regularization = regularization
            intermediate_result.variation = variation
            callback(intermediate_result)

        if (
            previous_energy is not None
            and abs(previous_energy - energy)
            / max(abs(previous_energy), abs(energy), 1)
            <= ftol
        ):
            success = True
            message = "Convergence: Relative reduction of objective function <= ftol."
            break
        if np.max(np.abs(grad)) <= gtol:
            success = True
            message = "Convergence: Norm of gradient <= gtol."
            break

        if optimize_regularization and optimize_variation:

            def f(x: np.ndarray) -> float:
                regularization_param, variation_param = x
                regularization = regularization_param**2
                variation = 0.5 * (1 + math.tanh(variation_param))
                param_update = _get_param_update(
                    energy_mat,
                    overlap_mat,
                    regularization,
                    variation,
                    lindep,
                )
                vec = params_to_vec(params + param_update)
                return np.vdot(vec, hamiltonian @ vec).real

            regularization_param = math.sqrt(regularization)
            variation_param = math.atanh(2 * min(1 - 1e-8, max(1e-8, variation)) - 1)
            result = minimize(
                f,
                x0=[regularization_param, variation_param],
                **optimize_kwargs,
            )
            regularization_param, variation_param = result.x
            regularization = regularization_param**2
            variation = 0.5 * (1 + math.tanh(variation_param))

        elif optimize_regularization:

            def f(x: np.ndarray) -> float:
                (regularization_param,) = x
                regularization = regularization_param**2
                param_update = _get_param_update(
                    energy_mat,
                    overlap_mat,
                    regularization,
                    variation,
                    lindep,
                )
                vec = params_to_vec(params + param_update)
                return np.vdot(vec, hamiltonian @ vec).real

            regularization_param = math.sqrt(regularization)
            result = minimize(
                f,
                x0=[regularization_param],
                **optimize_kwargs,
            )
            (regularization_param,) = result.x
            regularization = regularization_param**2

        elif optimize_variation:

            def f(x: np.ndarray) -> float:
                (variation_param,) = x
                variation = 0.5 * (1 + math.tanh(variation_param))
                param_update = _get_param_update(
                    energy_mat,
                    overlap_mat,
                    regularization,
                    variation,
                    lindep,
                )
                vec = params_to_vec(params + param_update)
                return np.vdot(vec, hamiltonian @ vec).real

            variation_param = math.atanh(2 * min(1 - 1e-8, max(1e-8, variation)) - 1)
            result = minimize(
                f,
                x0=[variation_param],
                **optimize_kwargs,
            )
            (variation_param,) = result.x
            variation = 0.5 * (1 + math.tanh(variation_param))

        param_update = _get_param_update(
            energy_mat,
            overlap_mat,
            regularization,
            variation,
            lindep,
        )
        params = params + param_update
        intermediate_result.nit += 1

    vec = params_to_vec(params)
    energy = np.vdot(vec, hamiltonian @ vec).real

    intermediate_result.x = params
    intermediate_result.fun = energy
    del intermediate_result.jac
    if callback is not None:
        callback(intermediate_result)

    return OptimizeResult(
        x=params,
        success=success,
        message=message,
        fun=energy,
        jac=grad,
        nfev=intermediate_result.nfev,
        njev=intermediate_result.njev,
        nlinop=intermediate_result.nlinop,
        nit=intermediate_result.nit,
    )


def _linear_method_matrices(
    vec: np.ndarray, jac: np.ndarray, hamiltonian: LinearOperator
) -> tuple[np.ndarray, np.ndarray]:
    _, n_params = jac.shape
    energy_mat = np.zeros((n_params + 1, n_params + 1), dtype=complex)
    overlap_mat = np.zeros_like(energy_mat)

    energy_mat[0, 0] = np.vdot(vec, hamiltonian @ vec)
    ham_jac = hamiltonian @ jac
    energy_mat[0, 1:] = vec.conj() @ ham_jac
    energy_mat[1:, 0] = energy_mat[0, 1:].conj()
    energy_mat[1:, 1:] = jac.T.conj() @ ham_jac

    overlap_mat[0, 0] = 1
    overlap_mat[0, 1:] = vec.conj() @ jac
    overlap_mat[1:, 0] = overlap_mat[0, 1:].conj()
    overlap_mat[1:, 1:] = jac.T.conj() @ jac

    return energy_mat.real, overlap_mat.real


def _solve_linear_method_eigensystem(
    energy_mat: np.ndarray,
    overlap_mat: np.ndarray,
    regularization: float,
    lindep: float,
) -> tuple[float, np.ndarray]:
    n_params = energy_mat.shape[0] - 1
    energy_mat_regularized = energy_mat.copy()
    energy_mat_regularized[1:, 1:] += regularization * np.eye(n_params)
    eigs, vecs, _ = safe_eigh(energy_mat_regularized, overlap_mat, lindep)
    eig = eigs[0]
    vec = vecs[:, 0]
    vec /= vec[0]
    vec = vec.real
    return eig, vec


def _get_param_update(
    energy_mat: np.ndarray,
    overlap_mat: np.ndarray,
    regularization: float,
    variation: float,
    lindep: float,
) -> np.ndarray:
    _, param_variations = _solve_linear_method_eigensystem(
        energy_mat, overlap_mat, regularization, lindep=lindep
    )
    average_overlap = np.dot(param_variations, overlap_mat @ param_variations)
    numerator = (1 - variation) * average_overlap
    denominator = (1 - variation) + variation * math.sqrt(1 + average_overlap)
    return param_variations[1:] / (1 + numerator / denominator)
