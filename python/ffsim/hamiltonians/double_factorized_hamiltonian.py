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

from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from ffsim.linalg import double_factorized


@dataclasses.dataclass
class DoubleFactorizedHamiltonian:
    r"""A Hamiltonian in the double-factorized form of the low rank decomposition.

    The double-factorized form of the molecular Hamiltonian is

    .. math::

        H = \sum_{pq, \sigma} \kappa_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \frac12 \sum_t \sum_{ij, \sigma\tau}
        Z^{(t)}_{ij} n^{(t)}_{i, \sigma} n^{(t)}_{j, \tau}
        + \text{constant}'.

    where

    .. math::

        n^{(t)}_{i, \sigma} = \sum_{pq} U^{(t)}_{pi}
        a^\dagger_{p, \sigma} a^\dagger_{q, \sigma} U^{(t)}_{qi}.

    Here each :math:`U^{(t)}` is a unitary matrix and each :math:`Z^{(t)}`
    is a real symmetric matrix.

    **"Z" representation**

    The "Z" representation of the double factorization is an alternative
    representation that sometimes yields simpler quantum circuits.

    Under the Jordan-Wigner transformation, the number operators take the form

    .. math::

        n^{(t)}_{i, \sigma} = \frac{(1 - z^{(t)}_{i, \sigma})}{2}

    where :math:`z^{(t)}_{i, \sigma}` is the Pauli Z operator in the rotated basis.
    The "Z" representation is obtained by rewriting the two-body part in terms
    of these Pauli Z operators and updating the one-body term as appropriate:

    .. math::

        H = \sum_{pq, \sigma} \kappa'_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \frac18 \sum_t \sum_{ij, \sigma\tau}^*
        Z^{(t)}_{ij} z^{(t)}_{i, \sigma} z^{(t)}_{j, \tau}
        + \text{constant}''

    where the asterisk denotes summation over indices :math:`ij, \sigma\tau`
    where :math:`i \neq j` or :math:`\sigma \neq \tau`.

    Attributes:
        one_body_tensor: The one-body tensor :math:`\kappa`.
        diag_coulomb_mats: The diagonal Coulomb matrices.
        orbital_rotations: The orbital rotations.
        constant: The constant.
        z_representation: Whether the Hamiltonian is in the "Z" representation rather
            than the "number" representation.
    """

    one_body_tensor: np.ndarray
    diag_coulomb_mats: np.ndarray
    orbital_rotations: np.ndarray
    constant: float = 0.0
    z_representation: bool = False

    @property
    def norb(self):
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


def double_factorized_hamiltonian(
    hamiltonian: MolecularHamiltonian,
    *,
    z_representation: bool = False,
    tol: float = 1e-8,
    max_vecs: int | None = None,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    options: dict | None = None,
    diag_coulomb_mask: np.ndarray | None = None,
    cholesky: bool = True,
) -> DoubleFactorizedHamiltonian:
    r"""Double-factorized decomposition of a molecular Hamiltonian.

    The double-factorized decomposition acts on a Hamiltonian of the form

    .. math::

        H = \sum_{pq, \sigma} h_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
            + \frac12 \sum_{pqrs, \sigma \tau} h_{pqrs}
            a^\dagger_{p, \sigma} a^\dagger_{r, \tau} a_{s, \tau} a_{q, \sigma}
            + \text{constant}.

    The Hamiltonian is decomposed into the double-factorized form

    .. math::

        H = \sum_{pq, \sigma} \kappa_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \frac12 \sum_t \sum_{ij, \sigma\tau}
        Z^{(t)}_{ij} n^{(t)}_{i, \sigma} n^{(t)}_{j, \tau}
        + \text{constant}'.

    where

    .. math::

        n^{(t)}_{i, \sigma} = \sum_{pq} U^{(t)}_{pi}
        a^\dagger_{p, \sigma} a^\dagger_{q, \sigma} U^{(t)}_{qi}.

    Here :math:`U^{(t)}_{ij}` and :math:`Z^{(t)}_{ij}` are tensors that are output by
    the decomposition, and :math:`\kappa_{pq}` is an updated one-body tensor.
    Each matrix :math:`U^{(t)}` is guaranteed to be unitary so that the
    :math:`n^{(t)}_{i, \sigma}` are number operators in a rotated basis, and
    each :math:`Z^{(t)}` is a real symmetric matrix.
    The number of terms :math:`t` in the decomposition depends on the allowed
    error threshold. A larger error threshold leads to a smaller number of terms.
    Furthermore, the `max_rank` parameter specifies an optional upper bound
    on :math:`t`.

    The default behavior of this routine is to perform a straightforward
    "exact" factorization of the two-body tensor based on a nested
    eigenvalue decomposition. Additionally, one can choose to optimize the
    coefficients stored in the tensor to achieve a "compressed" factorization.
    This option is enabled by setting the `optimize` parameter to `True`.
    The optimization attempts to minimize a least-squares objective function
    quantifying the error in the low rank decomposition.
    It uses `scipy.optimize.minimize`, passing both the objective function
    and its gradient. The core tensors returned by the optimization can be optionally
    constrained to have only certain elements allowed to be nonzero. This is achieved by
    passing the `diag_coulomb_mask` parameter, which is an :math:`N \times N` matrix of
    boolean values where :math:`N` is the number of orbitals. The nonzero elements of
    this matrix indicate where the core tensors are allowed to be nonzero. Only the
    upper triangular part of the matrix is used because the core tensors are symmetric.

    **"Z" representation**

    The "Z" representation of the double factorization is an alternative
    representation that sometimes yields simpler quantum circuits.

    Under the Jordan-Wigner transformation, the number operators take the form

    .. math::

        n^{(t)}_{i, \sigma} = \frac{(1 - z^{(t)}_{i, \sigma})}{2}

    where :math:`z^{(t)}_{i, \sigma}` is the Pauli Z operator in the rotated basis.
    The "Z" representation is obtained by rewriting the two-body part in terms
    of these Pauli Z operators and updating the one-body term as appropriate:

    .. math::

        H = \sum_{pq, \sigma} \kappa'_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
        + \frac18 \sum_t \sum_{ij, \sigma\tau}^*
        Z^{(t)}_{ij} z^{(t)}_{i, \sigma} z^{(t)}_{j, \tau}
        + \text{constant}''

    where the asterisk denotes summation over indices :math:`ij, \sigma\tau`
    where :math:`i \neq j` or :math:`\sigma \neq \tau`.

    Note: Currently, only real-valued two-body tensors are supported.

    Args:
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        z_representation: Whether to use the "Z" representation of the
            low rank decomposition.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: An optional limit on the number of terms to keep in the decomposition
            of the two-body tensor. This argument overrides ``tol``.
        optimize: Whether to optimize the tensors returned by the decomposition.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        diag_coulomb_mask: Diagonal Coulomb matrix mask to use in the optimization.
            This is a matrix of boolean values where the nonzero elements indicate where
            the diagonal coulomb matrices returned by optimization are allowed to be
            nonzero. This parameter is only used if `optimize` is set to `True`, and
            only the upper triangular part of the matrix is used.
        cholesky: Whether to perform the factorization using a modified Cholesky
            decomposition. If False, a full eigenvalue decomposition is used instead,
            which can be much more expensive. This argument is ignored if ``optimize``
            is set to True.

    Returns:
        The double-factorized Hamiltonian.

    References:
        - `arXiv:1808.02625`_
        - `arXiv:2104.08957`_

    .. _arXiv:1808.02625: https://arxiv.org/abs/1808.02625
    .. _arXiv:2104.08957: https://arxiv.org/abs/2104.08957
    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
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
        options=options,
        diag_coulomb_mask=diag_coulomb_mask,
        cholesky=cholesky,
    )
    df_hamiltonian = DoubleFactorizedHamiltonian(
        one_body_tensor=one_body_tensor,
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
    )

    if z_representation:
        df_hamiltonian = df_hamiltonian.to_z_representation()

    return df_hamiltonian
