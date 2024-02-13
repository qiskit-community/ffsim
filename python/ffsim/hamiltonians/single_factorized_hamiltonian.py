# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ffsim.contract.one_body import one_body_linop
from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from ffsim.linalg.double_factorized_decomposition import (
    _truncated_eigh,
    modified_cholesky,
)
from ffsim.states import dim


@dataclasses.dataclass(frozen=True)
class SingleFactorizedHamiltonian:
    r"""A Hamiltonian in the single-factorized representation.

    The single-factorized form of the molecular Hamiltonian is

    .. math::

        H = \sum_{\sigma, pq} \kappa_{pq} a^\dagger_{\sigma, p} a_{\sigma, q}
        + \frac12 \sum_{t=1}^L \left(\mathcal{M}^{(t)}\right)^2
        + \text{constant}'.

    Here each :math:`\mathcal{M}^{(t)}` is a one-body operator:

    .. math::

        \mathcal{M}^{(t)} =
        \sum_{\sigma, pq} M^{(t)}_{pq} a^\dagger_{\sigma, p} a_{\sigma, q}

    where each :math:`M^{(t)}` is a Hermitian matrix.

    Attributes:
        one_body_tensor (np.ndarray): The one-body tensor :math:`\kappa`.
        one_body_squares (np.ndarray): The one-body tensors :math:`M^{(t)}` whose
            squares are summed in the Hamiltonian.
        constant (float): The constant.
    """

    one_body_tensor: np.ndarray
    one_body_squares: np.ndarray
    constant: float = 0.0

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    @staticmethod
    def from_molecular_hamiltonian(
        hamiltonian: MolecularHamiltonian,
        *,
        tol: float = 1e-8,
        max_vecs: int | None = None,
        cholesky: bool = True,
    ) -> SingleFactorizedHamiltonian:
        r"""Initialize a SingleFactorizedHamiltonian from a MolecularHamiltonian.

        The number of terms in the decomposition depends on the allowed
        error threshold. A larger error threshold leads to a smaller number of terms.
        Furthermore, the `max_vecs` parameter specifies an optional upper bound
        on the number of terms.

        Note: Currently, only real-valued two-body tensors are supported.

        Args:
            hamiltonian: The Hamiltonian whose single-factorized representation to
                compute.
            tol: Tolerance for error in the decomposition.
                The error is defined as the maximum absolute difference between
                an element of the original tensor and the corresponding element of
                the reconstructed tensor.
            max_vecs: An optional limit on the number of terms to keep in the
                decomposition of the two-body tensor. This argument overrides ``tol``.
            cholesky: Whether to perform the factorization using a modified Cholesky
                decomposition. If False, a full eigenvalue decomposition is used
                instead, which can be much more expensive.

        Returns:
            The single-factorized Hamiltonian.
        """
        one_body_tensor = hamiltonian.one_body_tensor - 0.5 * np.einsum(
            "prqr", hamiltonian.two_body_tensor
        )

        norb = hamiltonian.norb
        reshaped_tensor = np.reshape(
            hamiltonian.two_body_tensor, (norb**2, norb**2)
        )

        if cholesky:
            cholesky_vecs = modified_cholesky(
                reshaped_tensor, tol=tol, max_vecs=max_vecs
            )
            one_body_squares = cholesky_vecs.T.reshape((-1, norb, norb))
        else:
            eigs, vecs = _truncated_eigh(reshaped_tensor, tol=tol, max_vecs=max_vecs)
            vecs *= np.sqrt(eigs)
            one_body_squares = vecs.T.reshape((-1, norb, norb))

        return SingleFactorizedHamiltonian(
            one_body_tensor=one_body_tensor,
            one_body_squares=one_body_squares,
            constant=hamiltonian.constant,
        )

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        dim_ = dim(norb, nelec)
        one_body_tensor_linop = one_body_linop(self.one_body_tensor, norb, nelec)
        one_body_square_linops = [
            0.5 * one_body_linop(one_body, norb, nelec) ** 2
            for one_body in self.one_body_squares
        ]

        def matvec(vec: np.ndarray):
            result = self.constant * vec
            result += one_body_tensor_linop @ vec
            for linop in one_body_square_linops:
                result += linop @ vec
            return result

        return LinearOperator(
            shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
        )
