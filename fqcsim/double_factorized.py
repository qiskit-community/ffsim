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

from fqcsim.linalg import double_factorized


@dataclasses.dataclass
class DoubleFactorizedHamiltonian:
    """A Hamiltonian in the double-factorized form of the low rank decomposition.

    Attributes:
        one_body_tensor: The one-body tensor.
        core_tensors: The core tensors.
        leaf_tensors: The leaf tensors.
        constant: The constant.
        z_representation: Whether the Hamiltonian is in the "Z" representation rather
            than the "number" representation.
    """

    one_body_tensor: np.ndarray
    core_tensors: np.ndarray
    leaf_tensors: np.ndarray
    constant: float = 0.0
    z_representation: bool = False

    @property
    def norb(self):
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    @property
    def two_body_tensor(self):
        """The two-body tensor."""
        return np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            self.leaf_tensors,
            self.leaf_tensors,
            self.core_tensors,
            self.leaf_tensors,
            self.leaf_tensors,
        )

    def to_z_representation(self) -> "DoubleFactorizedHamiltonian":
        """Return the Hamiltonian in the "Z" representation."""
        if self.z_representation:
            return self

        one_body_correction, constant_correction = _df_z_representation(
            self.core_tensors, self.leaf_tensors
        )
        return DoubleFactorizedHamiltonian(
            one_body_tensor=self.one_body_tensor + one_body_correction,
            core_tensors=self.core_tensors,
            leaf_tensors=self.leaf_tensors,
            constant=self.constant + constant_correction,
            z_representation=True,
        )

    def to_number_representation(self) -> "DoubleFactorizedHamiltonian":
        """Return the Hamiltonian in the "number" representation."""
        if not self.z_representation:
            return self

        one_body_correction, constant_correction = _df_z_representation(
            self.core_tensors, self.leaf_tensors
        )
        return DoubleFactorizedHamiltonian(
            one_body_tensor=self.one_body_tensor - one_body_correction,
            core_tensors=self.core_tensors,
            leaf_tensors=self.leaf_tensors,
            constant=self.constant - constant_correction,
            z_representation=False,
        )


def _df_z_representation(
    core_tensors: np.ndarray, leaf_tensors: np.ndarray
) -> tuple[np.ndarray, float]:
    one_body_correction = 0.5 * (
        np.einsum("tij,tpi,tqi->pq", core_tensors, leaf_tensors, leaf_tensors.conj())
        + np.einsum("tij,tpj,tqj->pq", core_tensors, leaf_tensors, leaf_tensors.conj())
    )
    constant_correction = 0.25 * np.einsum("ijj->", core_tensors) - 0.5 * np.sum(
        core_tensors
    )
    return one_body_correction, constant_correction


def double_factorized_decomposition(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    *,
    error_threshold: float = 1e-8,
    max_rank: int | None = None,
    z_representation: bool = False,
) -> DoubleFactorizedHamiltonian:
    r"""Double factorized decomposition of a molecular Hamiltonian."""
    one_body_tensor = one_body_tensor - 0.5 * np.einsum("prqr", two_body_tensor)

    core_tensors, leaf_tensors = double_factorized(
        two_body_tensor, max_rank=max_rank, error_threshold=error_threshold
    )
    df_hamiltonian = DoubleFactorizedHamiltonian(
        one_body_tensor=one_body_tensor,
        core_tensors=core_tensors,
        leaf_tensors=leaf_tensors,
    )

    if z_representation:
        df_hamiltonian = df_hamiltonian.to_z_representation()

    return df_hamiltonian
