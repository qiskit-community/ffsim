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

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg.blas import drot

from ffsim.gates import apply_givens_rotation


@dataclass(frozen=True)
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
    # TODO add phis for complex phases

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        if copy:
            vec = vec.copy()
        for target_orbs, theta in zip(
            itertools.cycle(self.interaction_pairs), self.thetas
        ):
            vec = apply_givens_rotation(
                vec, theta, target_orbs=target_orbs, norb=norb, nelec=nelec, copy=False
            )
        return vec

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
            interaction_pairs: The orbital pairs to apply the hop gates to.
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
