# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import scipy.linalg
import scipy.optimize


def M2V(M):
    return M[np.tril_indices(M.shape[0], k=-1)]


def V2M(V, n):
    M = np.zeros((n, n))
    i = np.tril_indices(n, k=-1)
    M[i] = V
    M -= M.T
    return M


def optimize_orbitals(
    rho1: np.ndarray,
    rho2: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    k0: np.ndarray | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
):
    """Find orbitals that minimize the energy of a pair of one- and two-RDMs."""
    norb, _ = h1.shape

    def fun(x: np.ndarray):
        U = scipy.linalg.expm(V2M(x, norb))
        h1tilde = np.einsum("pr,pA,rB->AB", h1, U, U, optimize=True)
        h2tilde = np.einsum("prqs,pA,rB,qC,sD->ABCD", h2, U, U, U, U, optimize=True)
        return np.einsum("pr,pr->", h1tilde, rho1) + 0.5 * np.einsum(
            "prqs,prqs->", h2tilde, rho2
        )

    def grad_U(k: np.ndarray):
        i = np.tril_indices(norb, k=-1)
        kmat = V2M(k, norb)
        l, V = scipy.linalg.eigh(kmat / 1j)
        T = np.zeros((norb, norb), dtype=complex)
        for m in range(norb):
            for n in range(norb):
                if np.abs(l[m] - l[n]) < 1e-6:
                    T[m, n] = np.exp(1j * l[m])
                else:
                    T[m, n] = (
                        -1j * (np.exp(1j * l[m]) - np.exp(1j * l[n])) / (l[m] - l[n])
                    )
        J = np.zeros((norb, norb, len(k)), dtype=complex)
        for m in range(len(k)):
            p, r = i[0][m], i[1][m]
            J[:, :, m] = np.einsum(
                "Am,m,mn,n,nB->AB", V, V.T.conj()[:, p], T, V[r, :], V.T.conj()
            ) - np.einsum(
                "Am,m,mn,n,nB->AB", V, V.T.conj()[:, r], T, V[p, :], V.T.conj()
            )
        return J.real

    def jac(x: np.ndarray):
        Ek = fun(x)
        J = grad_U(x)
        U = scipy.linalg.expm(V2M(x, norb))
        h1tilde = np.einsum("pr,pAX,rB->ABX", h1, J, U, optimize=True)
        h1tilde += np.einsum("pr,pA,rBX->ABX", h1, U, J, optimize=True)
        h2tilde = np.einsum("prqs,pAX,rB,qC,sD->ABCDX", h2, J, U, U, U, optimize=True)
        h2tilde += np.einsum("prqs,pA,rBX,qC,sD->ABCDX", h2, U, J, U, U, optimize=True)
        h2tilde += np.einsum("prqs,pA,rB,qCX,sD->ABCDX", h2, U, U, J, U, optimize=True)
        h2tilde += np.einsum("prqs,pA,rB,qC,sDX->ABCDX", h2, U, U, U, J, optimize=True)
        G = np.einsum("prX,pr->X", h1tilde, rho1) + 0.5 * np.einsum(
            "prqsX,prqs->X", h2tilde, rho2, optimize=True
        )
        print("E(k), G(k) = ", Ek, np.abs(G).max())
        return G

    if k0 is None:
        k0 = np.zeros((norb, norb))

    print("Initial energy ", fun(M2V(k0)))
    result = scipy.optimize.minimize(
        fun, M2V(k0), method=method, jac=jac, callback=callback, options=options
    )
    print("Final energy   ", fun(result.x))

    # Conjugate the orbital rotation to match ffsim.MolecularHamiltonian's convention
    return scipy.linalg.expm(V2M(result.x, norb)).T.conj()
