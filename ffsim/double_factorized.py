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
from functools import cached_property

import numpy as np
from opt_einsum import contract

from ffsim.linalg import double_factorized


@dataclasses.dataclass
class DoubleFactorizedHamiltonian:
    """A Hamiltonian in the double-factorized form of the low rank decomposition.

    Attributes:
        one_body_tensor: The one-body tensor.
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

    @cached_property
    def two_body_tensor(self):
        """The two-body tensor."""
        return contract(
            "tpk,tqk,tkl,trl,tsl->pqrs",
            self.orbital_rotations,
            self.orbital_rotations,
            self.diag_coulomb_mats,
            self.orbital_rotations,
            self.orbital_rotations,
            optimize="greedy",
        )

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


def double_factorized_decomposition(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    *,
    error_threshold: float = 1e-8,
    max_vecs: int | None = None,
    z_representation: bool = False,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    options: dict | None = None,
    diag_coulomb_mask: np.ndarray | None = None,
    cholesky: bool = True,
) -> DoubleFactorizedHamiltonian:
    """Double factorized decomposition of a molecular Hamiltonian."""
    one_body_tensor = one_body_tensor - 0.5 * np.einsum("prqr", two_body_tensor)

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor,
        error_threshold=error_threshold,
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
