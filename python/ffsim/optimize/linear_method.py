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


# TODO use math instead of np where possible
def minimize_linear_method(
    params_to_vec: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    hamiltonian: LinearOperator,
    maxiter: int = 10,
    alpha0: float = 0.0,
    xi0: float = 0.0,
    l0: float = 1e-5,
    pgtol=1e-8,
):
    def energy(x: np.ndarray):
        vec = params_to_vec(x)
        return np.real(np.vdot(vec, hamiltonian @ vec))

    converged = False
    finish = False
    i = 0
    t0 = [alpha0, xi0]
    info = {"theta": [], "E": [], "g": []}
    theta = x0.copy()
    while not finish:
        vec = params_to_vec(theta)
        ham_vec = hamiltonian @ vec
        grad = compute_wfn_gradient(params_to_vec, theta, vec)
        h = apply_H_wfn_gradient(hamiltonian, grad)
        A, B = compute_lm_matrices(vec, ham_vec, grad, h)
        Ec, grad = A[0, 0], 2 * A[0, 1:]
        info["theta"].append(theta)
        info["E"].append(Ec)
        info["g"].append(grad)
        print("     E,|g| = ", Ec, np.linalg.norm(grad), flush=True)

        def f(x):
            alpha, xi = x[0] ** 2, (1.0 + math.tanh(x[1])) / 2.0
            E, dp, Q = solve_lm_eigensystem(A, B, alpha, lindep=l0)
            num = (1 - xi) * Q
            den = (1 - xi) + xi * np.sqrt(1 + Q)
            par = theta + dp[1:] / (1 + num / den)
            vec = params_to_vec(par)
            return np.real(np.vdot(vec, hamiltonian @ vec))

        # TODO allow setting options, like maxiter and pgtol
        res = minimize(f, x0=t0, method="L-BFGS-B")
        t0 = res.x
        alpha, xi = t0[0] ** 2, (1.0 + np.tanh(t0[1])) / 2.0
        print("               alpha,xi,Emin = ", alpha, xi, res.fun)
        E, dp, Q = solve_lm_eigensystem(A, B, alpha)
        num = (1 - xi) * Q
        den = (1 - xi) + xi * np.sqrt(1 + Q)
        theta = theta + dp[1:] / (1 + num / den)
        """
            Emin   = 1e10
            xmin   = (None,None,None)
            for xi in [0.0,0.5,1.0]:
                for alpha in [0.0,0.1,0.2,0.5,1.0]:
                    E,dp,Q = solve_lm_eigensystem(A,B,alpha)
                    num = (1-xi)*Q
                    den = (1-xi)+xi*np.sqrt(1+Q)
                    par = theta+dp[1:]/(1+num/den)
                    E   = self.compute_energy(par)
                    if(E<Emin): 
                        Emin = E
                        xmin = (xi,alpha,par)
            if(Emin<Ec):
                theta = xmin[2]
            """
        if i > 0:
            li = len(info["g"]) - 1
            grad = info["g"][li]
            if np.linalg.norm(grad) < pgtol:
                finish = True
                converged = True
                break
        if i == maxiter - 1:
            finish = True
            converged = False
            break
        i += 1
    print(
        "LM optimization; maxiter,niter,E,converged? ",
        maxiter,
        i,
        min(info["E"]),
        converged,
    )
    # TODO return scipy.optimize.OptimizeResult
    return theta, E, grad, info


def compute_lm_matrices(
    vec: np.ndarray, ham_vec: np.ndarray, grad: np.ndarray, h: np.ndarray
):
    nt = h.shape[1]
    A = np.zeros((nt + 1, nt + 1), dtype=complex)
    A[0, 0] = np.vdot(vec, ham_vec).real
    A[0, 1:] = np.einsum("x,xi->i", np.conj(vec), h)
    A[1:, 0] = np.conj(A[0, 1:])
    A[1:, 1:] = np.einsum("xi,xj->ij", np.conj(grad), h)
    B = np.zeros((nt + 1, nt + 1), dtype=complex)
    B[0, 0] = 1.0
    B[0, 1:] = np.einsum("x,xi->i", np.conj(vec), grad)
    B[1:, 0] = np.conj(B[0, 1:])
    B[1:, 1:] = np.einsum("xi,xj->ij", np.conj(grad), grad)
    return A.real, B.real


def solve_lm_eigensystem(
    A: np.ndarray, B: np.ndarray, alpha: float = 0.0, lindep: float = 1e-5
):
    A_alpha = A.copy()
    nt = A.shape[0] - 1
    A_alpha[1:, 1:] += alpha * np.eye(nt)
    e, c, _ = safe_eigh(A_alpha, B, lindep)
    c = c[:, 0]
    c /= c[0]
    c = c.real
    # TODO is this just np.dot(c, B @ c)?
    Q = np.einsum("i,ij,j->", c, B, c)
    return e[0], c, Q


# TODO use scipy.optimize.approx_fprime instead
def compute_wfn_gradient(params_to_vec, theta: np.ndarray, vec: np.ndarray):
    g = np.zeros((len(vec), len(theta)), dtype=complex)
    for i in range(len(theta)):
        g[:, i] = wfn_deriv(params_to_vec, i, theta)
        g[:, i] = g[:, i] - np.vdot(vec, g[:, i]) * vec
    return g


def wfn_deriv(params_to_vec, i, parameters, eps=1e-8):
    nParams = parameters.size
    plus = parameters + eps * ei(i, nParams)
    minus = parameters - eps * ei(i, nParams)
    return (params_to_vec(plus) - params_to_vec(minus)) / (2 * eps)


def ei(i, n):
    v = np.zeros(n)
    v[i] = 1
    return v


def apply_H_wfn_gradient(hamiltonian: LinearOperator, g):
    _, ntheta = g.shape
    h = np.zeros(g.shape, dtype=complex)
    for i in range(ntheta):
        h[:, i] = hamiltonian @ g[:, i]
    return h
