# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Stochastic reconfiguration for optimization of a variational ansatz."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import scipy.linalg
from scipy.optimize import OptimizeResult, minimize
from scipy.sparse.linalg import LinearOperator

from ffsim.optimize._util import (
    WrappedCallable,
    WrappedLinearOperator,
    jacobian_finite_diff,
    orthogonalize_columns,
)


def minimize_stochastic_reconfiguration(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    hamiltonian: LinearOperator,
    x0: np.ndarray,
    *,
    maxiter: int = 1000,
    cond: float = 1e-4,
    epsilon: float = 1e-8,
    gtol: float = 1e-5,
    variation: float = 1.0,
    optimize_variation: bool = True,
    optimize_kwargs: dict | None = None,
    callback: Callable[[OptimizeResult], Any] | None = None,
) -> OptimizeResult:
    """Minimize the energy of a variational ansatz using stochastic reconfiguration.

    References:

    - `Generalized Lanczos algorithm for variational quantum Monte Carlo`_

    Args:
        params_to_vec: Function representing the wavefunction ansatz. It takes as input
            a vector of real-valued parameters and outputs the state vector represented
            by those parameters.
        hamiltonian: The Hamiltonian representing the energy to be minimized.
        x0: Initial guess for the parameters.
        maxiter: Maximum number of optimization iterations to perform.
        cond: `cond` argument to pass to `scipy.linalg.lstsq`_, which is called to
            solve for the parameter update.
        epsilon: Increment to use for approximating the gradient using
            finite difference.
        gtol: Convergence threshold for the norm of the projected gradient.
        variation: TODO Hyperparameter controlling the size of parameter variations
            used in the linear expansion of the wavefunction. Its value must be
            positive.
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

    .. _Generalized Lanczos algorithm for variational quantum Monte Carlo: https://doi.org/10.1103/PhysRevB.64.024512
    .. _scipy.optimize.OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    .. _scipy.linalg.lstsq: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """  # noqa: E501
    if variation <= 0:
        raise ValueError(f"variation must be positive. Got {variation}.")
    if maxiter < 1:
        raise ValueError(f"maxiter must be at least 1. Got {maxiter}.")

    if optimize_kwargs is None:
        optimize_kwargs = dict(method="L-BFGS-B")

    variation_param = math.sqrt(variation)
    params = x0.copy()
    converged = False
    intermediate_result = OptimizeResult(
        x=None, fun=None, jac=None, nfev=0, njev=0, nit=0, nlinop=0
    )

    params_to_vec = WrappedCallable(params_to_vec, intermediate_result)
    hamiltonian = WrappedLinearOperator(hamiltonian, intermediate_result)

    for i in range(maxiter):
        vec = params_to_vec(params)
        jac = jacobian_finite_diff(params_to_vec, params, len(vec), epsilon=epsilon)
        jac = orthogonalize_columns(jac, vec)

        energy, grad, overlap_mat = _sr_matrices(vec, jac, hamiltonian)
        intermediate_result.njev += 1

        if i > 0 and callback is not None:
            intermediate_result.x = params
            intermediate_result.fun = energy
            intermediate_result.jac = grad
            intermediate_result.overlap_mat = overlap_mat
            intermediate_result.variation = variation
            callback(intermediate_result)

        if np.linalg.norm(grad) < gtol:
            converged = True
            break

        if optimize_variation:

            def f(x: np.ndarray) -> float:
                (variation_param,) = x
                variation = variation_param**2
                param_update = _get_param_update(
                    grad,
                    overlap_mat,
                    variation,
                    cond=cond,
                )
                vec = params_to_vec(params + param_update)
                return np.vdot(vec, hamiltonian @ vec).real

            result = minimize(
                f,
                x0=[variation_param],
                **optimize_kwargs,
            )
            (variation_param,) = result.x
            variation = variation_param**2

        param_update = _get_param_update(
            grad,
            overlap_mat,
            variation,
            cond=cond,
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
        message = "Convergence: Norm of projected gradient <= gtol."
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


def _sr_matrices(
    vec: np.ndarray, jac: np.ndarray, hamiltonian: LinearOperator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy = np.vdot(vec, hamiltonian @ vec)
    grad = vec.conj() @ (hamiltonian @ jac)
    overlap_mat = jac.T.conj() @ jac
    return energy.real, 2 * grad.real, overlap_mat.real


def _get_param_update(
    grad: np.ndarray, overlap_mat: np.ndarray, variation: float, cond: float
) -> np.ndarray:
    x, _, _, _ = scipy.linalg.lstsq(overlap_mat, -0.5 * variation * grad, cond=cond)
    return x
