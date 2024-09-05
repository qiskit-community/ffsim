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
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

from ffsim.contract.diag_coulomb import diag_coulomb_linop
from ffsim.contract.num_op_sum import num_op_sum_linop
from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from ffsim.linalg import double_factorized
from ffsim.operators import FermionOperator
from ffsim.protocols import fermion_operator
from ffsim.states import dim


@dataclasses.dataclass(frozen=True)
class DoubleFactorizedHamiltonian:
    r"""A Hamiltonian in the double-factorized representation.

    The double-factorized form of the molecular Hamiltonian is

    .. math::

        H = \sum_{\sigma, pq} \kappa_{pq} a^\dagger_{\sigma, p} a_{\sigma, q}
        + \frac12 \sum_t \sum_{\sigma\tau, ij}
        Z^{(t)}_{ij} n^{(t)}_{\sigma, i} n^{(t)}_{\tau, j}
        + \text{constant}'.

    where

    .. math::

        n^{(t)}_{\sigma, i} = \sum_{pq} U^{(t)}_{pi}
        a^\dagger_{\sigma, p} a_{\sigma, q} U^{(t)}_{qi}.

    Here each :math:`U^{(t)}` is a unitary matrix and each :math:`Z^{(t)}`
    is a real symmetric matrix.

    **"Z" representation**

    The "Z" representation of the double factorization is an alternative
    representation that sometimes yields simpler quantum circuits.

    Under the Jordan-Wigner transformation, the number operators take the form

    .. math::

        n^{(t)}_{\sigma, i} = \frac{(1 - z^{(t)}_{\sigma, i})}{2}

    where :math:`z^{(t)}_{\sigma, i}` is the Pauli Z operator in the rotated basis.
    The "Z" representation is obtained by rewriting the two-body part in terms
    of these Pauli Z operators and updating the one-body term as appropriate:

    .. math::

        H = \sum_{\sigma, pq} \kappa'_{pq} a^\dagger_{\sigma, p} a_{\sigma, q}
        + \frac18 \sum_t \sum_{\sigma\tau, ij}^*
        Z^{(t)}_{ij} z^{(t)}_{\sigma, i} z^{(t)}_{\tau, j}
        + \text{constant}''

    where the asterisk denotes summation over indices :math:`\sigma\tau, ij`
    where :math:`\sigma \neq \tau` or :math:`i \neq j`.

    References:
        - `Low rank representations for quantum simulation of electronic structure`_
        - `Quantum Filter Diagonalization with Double-Factorized Hamiltonians`_

    Attributes:
        one_body_tensor (np.ndarray): The one-body tensor :math:`\kappa`.
        diag_coulomb_mats (np.ndarray): The diagonal Coulomb matrices.
        orbital_rotations (np.ndarray): The orbital rotations.
        constant (float): The constant.
        z_representation (bool): Whether the Hamiltonian is in the "Z" representation
            rather than the "number" representation.

    .. _Low rank representations for quantum simulation of electronic structure: https://arxiv.org/abs/1808.02625
    .. _Quantum Filter Diagonalization with Double-Factorized Hamiltonians: https://arxiv.org/abs/2104.08957
    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    one_body_tensor: np.ndarray
    diag_coulomb_mats: np.ndarray
    orbital_rotations: np.ndarray
    constant: float = 0.0
    z_representation: bool = False

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    def to_z_representation(self) -> "DoubleFactorizedHamiltonian":
        """Return the Hamiltonian in the "Z" representation."""
        if self.z_representation:
            return self

        one_body_correction, constant_correction = _df_z_representation(
            self.diag_coulomb_mats, self.orbital_rotations
        )
        return DoubleFactorizedHamiltonian(
            one_body_tensor=self.one_body_tensor + one_body_correction,
            diag_coulomb_mats=self.diag_coulomb_mats,
            orbital_rotations=self.orbital_rotations,
            constant=self.constant + constant_correction,
            z_representation=True,
        )

    def to_number_representation(self) -> "DoubleFactorizedHamiltonian":
        """Return the Hamiltonian in the "number" representation."""
        if not self.z_representation:
            return self

        one_body_correction, constant_correction = _df_z_representation(
            self.diag_coulomb_mats, self.orbital_rotations
        )
        return DoubleFactorizedHamiltonian(
            one_body_tensor=self.one_body_tensor - one_body_correction,
            diag_coulomb_mats=self.diag_coulomb_mats,
            orbital_rotations=self.orbital_rotations,
            constant=self.constant - constant_correction,
            z_representation=False,
        )

    @staticmethod
    def from_molecular_hamiltonian(
        hamiltonian: MolecularHamiltonian,
        *,
        z_representation: bool = False,
        tol: float = 1e-8,
        max_vecs: int | None = None,
        optimize: bool = False,
        method: str = "L-BFGS-B",
        callback=None,
        options: dict | None = None,
        diag_coulomb_indices: list[tuple[int, int]] | None = None,
        cholesky: bool = True,
    ) -> DoubleFactorizedHamiltonian:
        r"""Initialize a DoubleFactorizedHamiltonian from a MolecularHamiltonian.

        This function takes as input a :class:`MolecularHamiltonian`, which stores a
        one-body tensor, two-body tensor, and constant. It performs a double-factorized
        decomposition of the two-body tensor and computes a new one-body tensor
        and constant, and returns a :class:`DoubleFactorizedHamiltonian` storing the
        results.

        See :class:`DoubleFactorizedHamiltonian` for a description of the
        `z_representation` argument. See :func:`ffsim.linalg.double_factorized` for a
        description of the rest of the arguments.

        Args:
            hamiltonian: The Hamiltonian whose double-factorized representation to
                compute.
            z_representation: Whether to use the "Z" representation of the
                decomposition.
            tol: Tolerance for error in the decomposition.
                The error is defined as the maximum absolute difference between
                an element of the original tensor and the corresponding element of
                the reconstructed tensor.
            max_vecs: An optional limit on the number of terms to keep in the
                decomposition of the two-body tensor. This argument overrides ``tol``.
            optimize: Whether to optimize the tensors returned by the decomposition.
            method: The optimization method. See the documentation of
                `scipy.optimize.minimize`_ for possible values.
            callback: Callback function for the optimization. See the documentation of
                `scipy.optimize.minimize`_ for usage.
            options: Options for the optimization. See the documentation of
                `scipy.optimize.minimize`_ for usage.
            diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
                Coulomb matrices. Matrix entries corresponding to indices not in this
                list will be set to zero. This list should contain only upper
                trianglular indices, i.e., pairs :math:`(i, j)` where :math:`i \leq j`.
                Passing a list with lower triangular indices will raise an error.
                This parameter is only used if `optimize` is set to True.
            cholesky: Whether to perform the factorization using a modified Cholesky
                decomposition. If False, a full eigenvalue decomposition is used
                instead, which can be much more expensive. This argument is ignored if
                `optimize` is set to True.

        Returns:
            The double-factorized Hamiltonian.
        """
        one_body_tensor = hamiltonian.one_body_tensor - 0.5 * np.einsum(
            "prqr", hamiltonian.two_body_tensor
        )

        diag_coulomb_mats, orbital_rotations = double_factorized(
            hamiltonian.two_body_tensor,
            tol=tol,
            max_vecs=max_vecs,
            optimize=optimize,
            method=method,
            callback=callback,
            options=options,
            diag_coulomb_indices=diag_coulomb_indices,
            cholesky=cholesky,
        )
        df_hamiltonian = DoubleFactorizedHamiltonian(
            one_body_tensor=one_body_tensor,
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            constant=hamiltonian.constant,
        )

        if z_representation:
            df_hamiltonian = df_hamiltonian.to_z_representation()

        return df_hamiltonian

    def to_molecular_hamiltonian(self):
        """Convert the DoubleFactorizedHamiltonian to a MolecularHamiltonian."""
        df_hamiltonian = self.to_number_representation()
        two_body_tensor = np.einsum(
            "kij,kpi,kqi,krj,ksj->pqrs",
            df_hamiltonian.diag_coulomb_mats,
            df_hamiltonian.orbital_rotations,
            df_hamiltonian.orbital_rotations.conj(),
            df_hamiltonian.orbital_rotations,
            df_hamiltonian.orbital_rotations.conj(),
        )
        one_body_tensor = df_hamiltonian.one_body_tensor + 0.5 * np.einsum(
            "prqr", two_body_tensor
        )
        return MolecularHamiltonian(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            constant=df_hamiltonian.constant,
        )

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        dim_ = dim(norb, nelec)
        eigs, vecs = scipy.linalg.eigh(self.one_body_tensor)
        num_linop = num_op_sum_linop(eigs, norb, nelec, orbital_rotation=vecs)
        diag_coulomb_linops = [
            diag_coulomb_linop(
                diag_coulomb_mat,
                norb,
                nelec,
                orbital_rotation=orbital_rotation,
                z_representation=self.z_representation,
            )
            for diag_coulomb_mat, orbital_rotation in zip(
                self.diag_coulomb_mats, self.orbital_rotations
            )
        ]

        def matvec(vec: np.ndarray):
            vec = vec.astype(complex, copy=False)
            result = self.constant * vec
            result += num_linop @ vec
            for linop in diag_coulomb_linops:
                result += linop @ vec
            return result

        return LinearOperator(
            shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
        )

    def _diag_(self, norb: int, nelec: tuple[int, int]) -> np.ndarray:
        """Return the diagonal entries of the Hamiltonian."""
        return self.to_molecular_hamiltonian()._diag_(norb, nelec)

    def _fermion_operator_(self) -> FermionOperator:
        """Return a FermionOperator representing the object."""
        return fermion_operator(self.to_molecular_hamiltonian())


def _df_z_representation(
    diag_coulomb_mats: np.ndarray, orbital_rotations: np.ndarray
) -> tuple[np.ndarray, float]:
    one_body_correction = 0.5 * (
        np.einsum(
            "tij,tpi,tqi->pq",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
        )
        + np.einsum(
            "tij,tpj,tqj->pq",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
        )
    )
    constant_correction = 0.25 * np.einsum("ijj->", diag_coulomb_mats) - 0.5 * np.sum(
        diag_coulomb_mats
    )
    return one_body_correction, constant_correction
