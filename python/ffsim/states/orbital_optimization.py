# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import cmath
import itertools

import numpy as np
import scipy.linalg
import scipy.optimize

from ffsim.hamiltonians import MolecularHamiltonian
from ffsim.states.rdm import ReducedDensityMatrix


def generator_to_params(mat: np.ndarray):
    norb, _ = mat.shape
    return mat[np.tril_indices(norb, k=-1)]


def params_to_generator(params: np.ndarray, norb: int):
    mat = np.zeros((norb, norb))
    i = np.tril_indices(norb, k=-1)
    mat[i] = params
    mat -= mat.T
    return mat


def optimize_orbitals(
    rdm: ReducedDensityMatrix,
    hamiltonian: MolecularHamiltonian,
    *,
    initial_orbital_rotation: np.ndarray | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    return_optimize_result: bool = False,
) -> np.ndarray | tuple[np.ndarray, scipy.optimize.OptimizeResult]:
    """Find orbitals that minimize the energy of a pair of one- and two-RDMs."""
    norb = hamiltonian.norb
    rho1 = rdm.one_rdm
    rho2 = rdm.two_rdm
    h1 = hamiltonian.one_body_tensor
    h2 = hamiltonian.two_body_tensor

    def fun(x: np.ndarray):
        # Conjugate orbital rotation to match ffsim.MolecularHamiltonian's convention
        orbital_rotation = scipy.linalg.expm(params_to_generator(x, norb)).T.conj()
        return rdm.expectation(hamiltonian.rotated(orbital_rotation))

    def jac_u(x: np.ndarray):
        generator = params_to_generator(x, norb)
        eigs, vecs = scipy.linalg.eigh(-1j * generator)
        vecs_conj = vecs.T.conj()
        t_mat = np.zeros((norb, norb), dtype=complex)
        for (m, eig_m), (n, eig_n) in itertools.product(enumerate(eigs), repeat=2):
            if cmath.isclose(eig_m, eig_n, abs_tol=1e-6):
                t_mat[m, n] = cmath.exp(1j * eig_m)
            else:
                t_mat[m, n] = (
                    -1j
                    * (cmath.exp(1j * eig_m) - cmath.exp(1j * eig_n))
                    / (eig_m - eig_n)
                )
        grad = np.zeros((norb, norb, len(x)), dtype=complex)
        for m, (p, r) in enumerate(zip(*np.tril_indices(norb, k=-1))):
            grad[:, :, m] = np.einsum(
                "Am,m,mn,n,nB->AB",
                vecs,
                vecs_conj[:, p],
                t_mat,
                vecs[r, :],
                vecs_conj,
            ) - np.einsum(
                "Am,m,mn,n,nB->AB",
                vecs,
                vecs_conj[:, r],
                t_mat,
                vecs[p, :],
                vecs_conj,
            )
        return grad.real

    def jac(x: np.ndarray):
        orbital_rotation = scipy.linalg.expm(params_to_generator(x, norb))
        grad_u = jac_u(x)
        h1tilde = np.einsum(
            "pr,pAX,rB->ABX", h1, grad_u, orbital_rotation, optimize=True
        )
        h1tilde += np.einsum(
            "pr,pA,rBX->ABX", h1, orbital_rotation, grad_u, optimize=True
        )
        h2tilde = np.einsum(
            "prqs,pAX,rB,qC,sD->ABCDX",
            h2,
            grad_u,
            orbital_rotation,
            orbital_rotation,
            orbital_rotation,
            optimize=True,
        )
        h2tilde += np.einsum(
            "prqs,pA,rBX,qC,sD->ABCDX",
            h2,
            orbital_rotation,
            grad_u,
            orbital_rotation,
            orbital_rotation,
            optimize=True,
        )
        h2tilde += np.einsum(
            "prqs,pA,rB,qCX,sD->ABCDX",
            h2,
            orbital_rotation,
            orbital_rotation,
            grad_u,
            orbital_rotation,
            optimize=True,
        )
        h2tilde += np.einsum(
            "prqs,pA,rB,qC,sDX->ABCDX",
            h2,
            orbital_rotation,
            orbital_rotation,
            orbital_rotation,
            grad_u,
            optimize=True,
        )
        grad = np.einsum("prX,pr->X", h1tilde, rho1) + 0.5 * np.einsum(
            "prqsX,prqs->X", h2tilde, rho2, optimize=True
        )
        return grad

    if initial_orbital_rotation is None:
        initial_orbital_rotation = np.eye(norb)

    result = scipy.optimize.minimize(
        fun,
        generator_to_params(scipy.linalg.logm(initial_orbital_rotation)),
        method=method,
        jac=jac,
        callback=callback,
        options=options,
    )

    orbital_rotation = scipy.linalg.expm(params_to_generator(result.x, norb))

    if return_optimize_result:
        return orbital_rotation, result
    return orbital_rotation
