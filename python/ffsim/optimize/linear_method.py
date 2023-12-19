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

from ffsim.states import one_hot


class _WrappedCallable:
    """Callable wrapper used to count function calls."""

    def __init__(
        self, func: Callable[[np.ndarray], np.ndarray], optimize_result: OptimizeResult
    ):
        self.func = func
        self.optimize_result = optimize_result

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.optimize_result.nfev += 1
        return self.func(x)


class _WrappedLinearOperator:
    """LinearOperator wrapper used to count LinearOperator applications."""

    def __init__(self, linop: LinearOperator, optimize_result: OptimizeResult):
        self.linop = linop
        self.optimize_result = optimize_result

    def __matmul__(self, other: np.ndarray):
        if len(other.shape) == 1:
            self.optimize_result.nlinop += 1
        else:
            _, n = other.shape
            self.optimize_result.nlinop += n
        return self.linop @ other

    def __rmatmul__(self, other: np.ndarray):
        if len(other.shape) == 1:
            self.optimize_result.nlinop += 1
        else:
            n, _ = other.shape
            self.optimize_result.nlinop += n
        return other @ self.linop


def minimize_linear_method(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    hamiltonian: LinearOperator,
    x0: np.ndarray,
    maxiter: int = 1000,
    regularization_param: float = 0.0,
    variation_param: float = 0.0,
    lindep: float = 1e-5,
    pgtol: float = 1e-8,
    scipy_optimize_minimize_args: dict | None = None,
    callback: Callable[[OptimizeResult], Any] | None = None,
) -> OptimizeResult:
    """Minimize the energy of a variational ansatz using the linear method.

    Args:
        params_to_vec: Function representing the wavefunction ansatz. It takes as input
            a vector of real-valued parameters and outputs the state vector represented
            by those parameters.
        hamiltonian: The Hamiltonian representing the energy to be minimized.
        x0: Initial guess for the parameters.
        maxiter: Maximum number of optimization iterations to perform.
        regularization_param: Hyperparameter controlling regularization of the
            energy matrix. A larger value results in greater regularization.
            The value passed here is only the initial value of the hyperparameter,
            which is adjusted during optimization.
        variation_param: Hyperparameter controlling the size of parameter variations
            used in the linear expansion of the wavefunction. A larger value results in
            larger variations.
            The value passed here is only the initial value of the hyperparameter,
            which is adjusted during optimization.
        lindep: Linear dependency threshold to use when solving the generalized
            eigenvalue problem.
        pgtol: Convergence threshold for the norm of the projected gradient.
        scipy_optimize_minimize_args: Arguments to use when calling
            `scipy.optimize.minimize`_ to optimize the hyperparameters. The call is
            constructed as

            .. code::

                scipy.optimize.minimize(f, x0, **scipy_optimize_minimize_args)

        callback: A callable called after each iteration. It is called with the
            signature

            .. code::

                callback(intermediate_result: OptimizeResult)

            where ``intermediate_result`` is a `scipy.optimize.OptimizeResult`_
            with attributes ``x``  and ``fun``, the present values of the parameter
            vector and objective function. For all iterations except for the last,
            it also contains the ``jac`` attribute holding the present value of the
            gradient.

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

    .. _scipy.optimize.OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    if maxiter < 1:
        raise ValueError(f"maxiter must be at least 1. Got {maxiter}.")

    if scipy_optimize_minimize_args is None:
        scipy_optimize_minimize_args = dict(method="L-BFGS-B")

    params = x0.copy()
    converged = False
    intermediate_result = OptimizeResult(
        x=None, fun=None, jac=None, nfev=0, njev=0, nit=0, nlinop=0
    )

    params_to_vec = _WrappedCallable(params_to_vec, intermediate_result)
    hamiltonian = _WrappedLinearOperator(hamiltonian, intermediate_result)

    for i in range(maxiter):
        vec = params_to_vec(params)
        jac = _jac(params_to_vec, params, vec, epsilon=1e-8)

        energy_mat, overlap_mat = _linear_method_matrices(vec, jac, hamiltonian)
        energy = energy_mat[0, 0]
        grad = 2 * energy_mat[0, 1:]
        intermediate_result.njev += 1

        if i > 0 and callback is not None:
            intermediate_result.x = params
            intermediate_result.fun = energy
            intermediate_result.jac = grad
            callback(intermediate_result)

        if np.linalg.norm(grad) < pgtol:
            converged = True
            break

        def f(x: np.ndarray) -> float:
            regularization_param, variation_param = x
            param_update = _get_param_update(
                energy_mat,
                overlap_mat,
                regularization_param,
                variation_param,
                lindep,
            )
            vec = params_to_vec(params + param_update)
            return np.vdot(vec, hamiltonian @ vec).real

        result = minimize(
            f,
            x0=[regularization_param, variation_param],
            **scipy_optimize_minimize_args,
        )
        regularization_param, variation_param = result.x

        param_update = _get_param_update(
            energy_mat,
            overlap_mat,
            regularization_param,
            variation_param,
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

    if converged:
        success = True
        message = "Convergence: Norm of projected gradient <= pgtol."
    else:
        success = False
        message = "Stop: Total number of iterations reached limit."

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

    energy_mat[0, 0] = np.vdot(vec, hamiltonian @ vec).real
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


def _jac(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    vec: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    jac = np.zeros((len(vec), len(theta)), dtype=complex)
    for i in range(len(theta)):
        jac[:, i] = _grad(params_to_vec, theta, i, epsilon)
        jac[:, i] = jac[:, i] - np.vdot(vec, jac[:, i]) * vec
    return jac


def _grad(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    index: int,
    epsilon: float,
) -> np.ndarray:
    unit = one_hot(len(theta), index, dtype=float)
    plus = theta + epsilon * unit
    minus = theta - epsilon * unit
    return (params_to_vec(plus) - params_to_vec(minus)) / (2 * epsilon)


def _get_param_update(
    energy_mat: np.ndarray,
    overlap_mat: np.ndarray,
    regularization_param: float,
    variation_param: float,
    lindep: float,
) -> np.ndarray:
    _, param_variations = _solve_linear_method_eigensystem(
        energy_mat, overlap_mat, regularization_param**2, lindep=lindep
    )
    average_overlap = np.dot(param_variations, overlap_mat @ param_variations)
    variation = 0.5 * (1 + math.tanh(variation_param))
    numerator = (1 - variation) * average_overlap
    denominator = (1 - variation) + variation * math.sqrt(1 + average_overlap)
    return param_variations[1:] / (1 + numerator / denominator)
