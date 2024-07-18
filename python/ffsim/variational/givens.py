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
import itertools
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

    .. warning::
        This class is deprecated. Use :class:`ffsim.GivensAnsatzOp` instead.

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
        phis (np.ndarray | None): The optional phase angles for the Givens rotations.
        phase_angles (np.ndarray | None): The optional phase angles for the layer of
            single-orbital phase gates.

    .. _Givens rotations: ffsim.html#ffsim.apply_givens_rotation
    """

    norb: int
    interaction_pairs: list[tuple[int, int]]
    thetas: np.ndarray
    phis: np.ndarray | None
    phase_angles: np.ndarray | None

    def __post_init__(self):
        if len(self.thetas) != len(self.interaction_pairs):
            raise ValueError(
                "The number of thetas must equal the number of interaction pairs. "
                f"Got {len(self.phis)} and {len(self.interaction_pairs)}."
            )
        if self.phis is not None and len(self.phis) != len(self.interaction_pairs):
            raise ValueError(
                "The number of phis must equal the number of interaction pairs. "
                f"Got {len(self.phis)} and {len(self.interaction_pairs)}."
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

    @staticmethod
    def n_params(
        norb: int,
        interaction_pairs: list[tuple[int, int]],
        with_phis: bool = True,
        with_phase_angles: bool = True,
    ) -> int:
        """Return the number of parameters of an ansatz with given settings.

        Args:
            norb: The number of spatial orbitals.
            interaction_pairs: The orbital pairs to apply the Givens rotation gates to.
            with_phis: Whether to include complex phases for the Givens rotations.
            with_phase_angles: Whether to include a layer of single-orbital phase gates.
        """
        return (1 + with_phis) * len(interaction_pairs) + with_phase_angles * norb

    def to_parameters(self) -> np.ndarray:
        """Convert the operator to a real-valued parameter vector."""
        if self.phis is not None and self.phase_angles is not None:
            return np.concatenate([self.thetas, self.phis, self.phase_angles])
        if self.phis is not None:
            return np.concatenate([self.thetas, self.phis])
        if self.phase_angles is not None:
            return np.concatenate([self.thetas, self.phase_angles])
        return self.thetas

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        norb: int,
        interaction_pairs: list[tuple[int, int]],
        with_phis: bool = True,
        with_phase_angles: bool = True,
    ) -> GivensAnsatzOp:
        """Initialize the operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            interaction_pairs: The orbital pairs to apply the Givens rotation gates to.
            with_phis: Whether to include complex phases for the Givens rotations.
            with_phase_angles: Whether to include a layer of single-orbital phase gates.
        """
        n_params = (1 + with_phis) * len(interaction_pairs) + with_phase_angles * norb
        if len(params) != n_params:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {n_params} but got {len(params)}."
            )
        thetas = params[: len(interaction_pairs)]
        phis = None
        phase_angles = None
        if with_phis and with_phase_angles:
            phis = params[len(interaction_pairs) : 2 * len(interaction_pairs)]
            phase_angles = params[2 * len(interaction_pairs) :]
        elif with_phis:
            phis = params[len(interaction_pairs) :]
            phase_angles = None
        elif with_phase_angles:
            phis = None
            phase_angles = params[len(interaction_pairs) :]
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
        interaction_pairs, thetas, phis = _brickwork_givens_rotations(
            interaction_pairs, thetas, phis, norb=norb
        )
        return GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=np.array(thetas),
            phis=np.array(phis),
            phase_angles=np.angle(phases),
        )

    def to_orbital_rotation(self) -> np.ndarray:
        """Convert the Givens ansatz operator to an orbital rotation."""
        phis = self.phis
        phase_angles = self.phase_angles
        if phis is None:
            phis = np.zeros(len(self.interaction_pairs))
        if phase_angles is None:
            phase_angles = np.zeros(self.norb)
        orbital_rotation = np.diag(np.exp(1j * phase_angles))
        for (i, j), theta, phi in zip(
            self.interaction_pairs[::-1], self.thetas[::-1], phis[::-1]
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
            if (self.phis is None) != (other.phis is None):
                return False
            if (self.phase_angles is None) != (other.phase_angles is None):
                return False
            if not np.allclose(self.thetas, other.thetas, rtol=rtol, atol=atol):
                return False
            if self.phis is not None and not np.allclose(
                cast(np.ndarray, self.phis),
                cast(np.ndarray, other.phis),
                rtol=rtol,
                atol=atol,
            ):
                return False
            if self.phase_angles is not None and not np.allclose(
                cast(np.ndarray, self.phase_angles),
                cast(np.ndarray, other.phase_angles),
                rtol=rtol,
                atol=atol,
            ):
                return False
            return True
        return NotImplemented


def _brickwork_givens_rotations(
    interaction_pairs: list[tuple[int, int]],
    thetas: list[float],
    phis: list[float],
    norb: int,
) -> tuple[list[tuple[int, int]], list[float], list[float]]:
    """Expand a sparse Givens rotation decomposition to a full brickwork pattern."""
    # Construct a brickwork pattern of Givens rotations with angles set to zero
    q, r = divmod(norb, 2)
    even_layers = [
        [((i, i + 1), 0.0, 0.0) for i in range(0, norb - 1, 2)] for _ in range(q + r)
    ]
    odd_layers = [
        [((i, i + 1), 0.0, 0.0) for i in range(1, norb - 1, 2)] for _ in range(q)
    ]
    # even_layer_index[i] is the index of the last even layer acting on orbital i
    even_layer_index = [-1] * norb
    # odd_layer_index[i] is the index of the last odd layer acting on orbital i
    odd_layer_index = [-1] * norb
    for (i, j), theta, phi in zip(interaction_pairs, thetas, phis):
        if i > j:
            # Enforce i < j
            i, j = j, i
            theta = -theta
            phi = -phi
        if i % 2 == 0:
            # Even layer
            # Get the index of the even layer this Givens rotation should go in
            index = max(odd_layer_index[i], odd_layer_index[j]) + 1
            # Add the Givens rotation in the appropriate place
            even_layers[index][i // 2] = ((i, j), theta, phi)
            # Update the even layer index
            even_layer_index[i] = index
            even_layer_index[j] = index
        else:
            # Odd layer
            # Get the index of the odd layer this Givens rotation should go in
            index = max(even_layer_index[i], even_layer_index[j])
            # Add the Givens rotation in the appropriate place
            odd_layers[index][i // 2] = ((i, j), theta, phi)
            # Update the odd layer index
            odd_layer_index[i] = index
            odd_layer_index[j] = index
    # Construct the new Givens rotation decomposition and return
    new_interaction_pairs = []
    new_thetas = []
    new_phis = []
    for even_layer, odd_layer in itertools.zip_longest(
        even_layers, odd_layers, fillvalue=()
    ):
        for layer in [even_layer, odd_layer]:
            for pair, theta, phi in layer:
                new_interaction_pairs.append(pair)
                new_thetas.append(theta)
                new_phis.append(phi)
    return new_interaction_pairs, new_thetas, new_phis
