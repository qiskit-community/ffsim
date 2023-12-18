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

import math
from typing import Callable

import numpy as np
from pyscf.lib.linalg_helper import safe_eigh
from scipy.optimize import OptimizeResult, minimize
from scipy.sparse.linalg import LinearOperator

from ffsim.states import one_hot


# TODO optionally take energy function as input (maybe instead of Hamiltonian)
# TODO add callback function argument
def minimize_linear_method(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    hamiltonian: LinearOperator,
    maxiter: int = 1000,
    regularization_param: float = 0.0,
    variation_param: float = 0.0,
    lindep: float = 1e-5,
    pgtol: float = 1e-8,
) -> OptimizeResult:
    # TODO get rid of info and instead, show an example of using callback function
    info = {"theta": [], "E": [], "g": []}
    converged = False

    params = x0.copy()

    for i in range(maxiter):
        vec = params_to_vec(params)
        jac = _jac(params_to_vec, params, vec, epsilon=1e-8)

        energy_mat, overlap_mat = _linear_method_matrices(vec, jac, hamiltonian)
        energy, grad = energy_mat[0, 0], 2 * energy_mat[0, 1:]

        info["theta"].append(params)
        info["E"].append(energy)
        info["g"].append(grad)

        print("     E,|g| = ", energy, np.linalg.norm(grad), flush=True)

        def f(x: np.ndarray) -> float:
            regularization_param, variation_param = x
            _, param_update = _get_param_update(
                energy_mat,
                overlap_mat,
                regularization_param,
                variation_param,
                lindep,
            )
            vec = params_to_vec(params + param_update)
            return np.vdot(vec, hamiltonian @ vec).real

        # TODO allow setting options, like maxiter and pgtol
        result = minimize(
            f,
            x0=[regularization_param, variation_param],
            method="L-BFGS-B",
        )
        regularization_param, variation_param = result.x
        energy_linear, param_update = _get_param_update(
            energy_mat,
            overlap_mat,
            regularization_param,
            variation_param,
            lindep,
        )
        params += param_update

        if i > 0:
            if np.linalg.norm(grad) < pgtol:
                converged = True
                break

    # TODO add nfev, success, status, message
    return OptimizeResult(
        x=params,
        fun=energy_linear,
        jac=grad,
        nit=i + 1,
        info=info,
    )


def _linear_method_matrices(
    vec: np.ndarray, jac: np.ndarray, hamiltonian: LinearOperator
) -> tuple[np.ndarray, np.ndarray]:
    _, n_params = jac.shape
    energy_mat = np.zeros((n_params + 1, n_params + 1), dtype=complex)
    overlap_mat = np.zeros_like(energy_mat)

    energy_mat[0, 0] = np.vdot(vec, hamiltonian @ vec).real
    ham_grad = hamiltonian @ jac
    energy_mat[0, 1:] = vec.conj() @ ham_grad
    energy_mat[1:, 0] = energy_mat[0, 1:].conj()
    energy_mat[1:, 1:] = jac.T.conj() @ ham_grad

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
) -> tuple[float, np.ndarray, float]:
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
) -> tuple[float, np.ndarray]:
    energy_linear, param_variations = _solve_linear_method_eigensystem(
        energy_mat, overlap_mat, regularization_param**2, lindep=lindep
    )
    # TODO check this doesn't fail if safe_eigh detects linear dependency
    average_overlap = np.dot(param_variations, overlap_mat @ param_variations)
    variation = 0.5 * (1 + math.tanh(variation_param))
    numerator = (1 - variation) * average_overlap
    denominator = (1 - variation) + variation * math.sqrt(1 + average_overlap)
    return energy_linear, param_variations[1:] / (1 + numerator / denominator)
