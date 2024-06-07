# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Givens rotation ansatz."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.linalg.blas import drot
from scipy.linalg.lapack import zrot
from typing_extensions import deprecated

from ffsim import linalg
from ffsim.gates import apply_orbital_rotation


@dataclass(frozen=True)
@deprecated("GivensAnsatzOperator is deprecated. Use GivensAnsatzOp instead.")
class GivensAnsatzOperator:
    """A Givens rotation ansatz operator.

    The Givens rotation ansatz consists of a sequence of `Givens rotations`_.

    Note that this ansatz does not implement any interactions between spin alpha and
    spin beta orbitals.

    Attributes:
        norb (int): The number of spatial orbitals.
        interaction_pairs (list[tuple[int, int]]): The orbital pairs to apply the Givens
            rotations to.
        thetas (np.ndarray): The angles for the Givens rotations.

    .. _Givens rotations: ffsim.html#ffsim.apply_givens_rotation
    """

    norb: int
    interaction_pairs: list[tuple[int, int]]
    thetas: np.ndarray

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        return apply_orbital_rotation(
            vec, self.to_orbital_rotation(), norb=norb, nelec=nelec, copy=copy
        )

    def to_parameters(self) -> np.ndarray:
        """Convert the operator to a real-valued parameter vector."""
        num_params = len(self.thetas)
        params = np.zeros(num_params)
        params[: len(self.thetas)] = self.thetas
        return params

    @staticmethod
    def from_parameters(
        params: np.ndarray, norb: int, interaction_pairs: list[tuple[int, int]]
    ) -> GivensAnsatzOperator:
        """Initialize the operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            interaction_pairs: The orbital pairs to apply the Givens rotation gates to.
        """
        return GivensAnsatzOperator(
            norb=norb, interaction_pairs=interaction_pairs, thetas=params
        )

    def to_orbital_rotation(self) -> np.ndarray:
        orbital_rotation = np.eye(self.norb)
        for (i, j), theta in zip(self.interaction_pairs[::-1], self.thetas[::-1]):
            orbital_rotation[:, j], orbital_rotation[:, i] = drot(
                orbital_rotation[:, j],
                orbital_rotation[:, i],
                math.cos(theta),
                math.sin(theta),
            )
        return orbital_rotation


@dataclass(frozen=True)
class GivensAnsatzOp:
    """A Givens rotation ansatz operator.

    The Givens rotation ansatz consists of a sequence of `Givens rotations`_ followed
    by a layer of single-orbital phase gates.

    Note that this ansatz does not implement any interactions between spin alpha and
    spin beta orbitals.

    Attributes:
        norb (int): The number of spatial orbitals.
        interaction_pairs (list[tuple[int, int]]): The orbital pairs to apply the Givens
            rotations to.
        thetas (np.ndarray): The angles for the Givens rotations.
        phis (np.ndarray): The phase angles for the Givens rotations.
        phase_angles (np.ndarray): The phase angles for the layer of single-orbital
            phase gates.

    .. _Givens rotations: ffsim.html#ffsim.apply_givens_rotation
    """

    norb: int
    interaction_pairs: list[tuple[int, int]]
    thetas: np.ndarray
    # TODO make phis optional
    phis: np.ndarray
    phase_angles: np.ndarray | None = None

    def __post_init__(self):
        if not len(self.interaction_pairs) == len(self.thetas) == len(self.phis):
            raise ValueError(
                "The number of interaction pairs, the length of thetas, and "
                "the length of phis must all be equal. "
                f"Got {len(self.interaction_pairs)}, "
                f"{len(self.thetas)}, and {len(self.phis)}."
            )
        if self.phase_angles is not None and len(self.phase_angles) != self.norb:
            raise ValueError(
                "The number of phase angles must equal the number of orbitals. "
                f"Got {len(self.phase_angles)} and {self.norb}."
            )

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        return apply_orbital_rotation(
            vec, self.to_orbital_rotation(), norb=norb, nelec=nelec, copy=copy
        )

    def to_parameters(self) -> np.ndarray:
        """Convert the operator to a real-valued parameter vector."""
        if self.phase_angles is not None:
            return np.concatenate([self.thetas, self.phis, self.phase_angles])
        return np.concatenate([self.thetas, self.phis])

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        norb: int,
        interaction_pairs: list[tuple[int, int]],
        with_phase_layer: bool = False,
    ) -> GivensAnsatzOp:
        """Initialize the operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            interaction_pairs: The orbital pairs to apply the Givens rotation gates to.
            with_phase_layer: Whether to include a layer of single-orbital phase gates.
        """
        n_params = 2 * len(interaction_pairs) + with_phase_layer * norb
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        thetas = params[: len(interaction_pairs)]
        phis = params[len(interaction_pairs) : 2 * len(interaction_pairs)]
        phase_angles = None
        if with_phase_layer:
            phase_angles = params[2 * len(interaction_pairs) :]
        return GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=thetas,
            phis=phis,
            phase_angles=phase_angles,
        )

    @staticmethod
    def from_orbital_rotation(orbital_rotation: np.ndarray) -> GivensAnsatzOp:
        """Initialize the operator from an orbital rotation.

        Args:
            orbital_rotation: The orbital rotation.
        """
        norb, _ = orbital_rotation.shape
        givens_rotations, phases = linalg.givens_decomposition(orbital_rotation)
        interaction_pairs = []
        thetas = []
        phis = []
        for c, s, i, j in givens_rotations:
            interaction_pairs.append((i, j))
            r, phi = cmath.polar(s)
            thetas.append(math.atan2(r, c))
            phis.append(phi)
        return GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=np.array(thetas),
            phis=np.array(phis),
            phase_angles=np.angle(phases),
        )

    def to_orbital_rotation(self) -> np.ndarray:
        """Convert the Givens ansatz operator to an orbital rotation."""
        if self.phase_angles is None:
            orbital_rotation = np.eye(self.norb, dtype=complex)
        else:
            orbital_rotation = np.diag(np.exp(1j * self.phase_angles))
        for (i, j), theta, phi in zip(
            self.interaction_pairs[::-1], self.thetas[::-1], self.phis[::-1]
        ):
            orbital_rotation[:, j], orbital_rotation[:, i] = zrot(
                orbital_rotation[:, j],
                orbital_rotation[:, i],
                math.cos(theta),
                cmath.rect(math.sin(theta), -phi),
            )
        return orbital_rotation

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, GivensAnsatzOp):
            if self.norb != other.norb:
                return False
            if self.interaction_pairs != other.interaction_pairs:
                return False
            if not np.allclose(self.thetas, other.thetas, rtol=rtol, atol=atol):
                return False
            if not np.allclose(self.phis, other.phis, rtol=rtol, atol=atol):
                return False
            if (self.phase_angles is None) != (other.phase_angles is None):
                return False
            if self.phase_angles is not None:
                return np.allclose(
                    cast(np.ndarray, self.phase_angles),
                    cast(np.ndarray, other.phase_angles),
                    rtol=rtol,
                    atol=atol,
                )
            return True
        return NotImplemented
