# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse.linalg import LinearOperator

from ffsim import linalg


class WrappedCallable:
    """Callable wrapper used to count function calls."""

    def __init__(
        self, func: Callable[[np.ndarray], np.ndarray], optimize_result: OptimizeResult
    ):
        self.func = func
        self.optimize_result = optimize_result

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.optimize_result.nfev += 1
        return self.func(x)


class WrappedLinearOperator:
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


def gradient_finite_diff(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    index: int,
    epsilon: float,
) -> np.ndarray:
    """Return the gradient of one of the components of a function.

    Given a function that maps a vector of "parameters" to an output vector, return
    the gradient of one of the parameter components.

    Args:
        params_to_vec: Function that maps a parameter vector to an output vector.
        theta: The parameters at which to evaluate the gradient.
        index: The index of the parameter to take the gradient of.
        epsilon: Finite difference step size.

    Returns:
        The gradient of the desired parameter component.
    """
    unit = linalg.one_hot(len(theta), index, dtype=float)
    plus = theta + epsilon * unit
    minus = theta - epsilon * unit
    return (params_to_vec(plus) - params_to_vec(minus)) / (2 * epsilon)


def jacobian_finite_diff(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    dim: int,
    epsilon: float,
) -> np.ndarray:
    """Return the Jacobian matrix of a function.

    Given a function that maps a vector of "parameters" to an output vector, return
    the matrix whose :math:$i$-th column contains the gradient of the
    :math:$i$-th component of the function.

    Args:
        params_to_vec: Function that maps a parameter vector to an output vector.
        theta: The parameters at which to evaluate the Jacobian.
        dim: The dimension of an output vector of the function.
        epsilon: Finite difference step size.

    Returns:
        The Jacobian matrix.
    """
    jac = np.zeros((dim, len(theta)), dtype=complex)
    for i in range(len(theta)):
        jac[:, i] = gradient_finite_diff(params_to_vec, theta, i, epsilon)
    return jac


def orthogonalize_columns(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Orthogonalize the columns of a matrix with respect to a vector.

    Given a matrix and a vector, return a new matrix whose columns contain the
    components of the old columns orthogonal to the vector.

    Args:
        mat: The matrix.
        vec: The vector.

    Returns:
        The new matrix with columns orthogonal to the vector.
    """
    coeffs = vec.T.conj() @ mat
    return mat - vec.reshape((-1, 1)) * coeffs.reshape((1, -1))
