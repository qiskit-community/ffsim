# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hop gate ansatz."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ffsim.gates import apply_hop_gate, apply_orbital_rotation
from ffsim.variational.util import (
    orbital_rotation_from_parameters,
    orbital_rotation_to_parameters,
)


@dataclass(frozen=True)
class HopGateAnsatzOperator:
    r"""A hop gate ansatz operator."""

    norb: int
    interaction_pairs: list[tuple[int, int]]
    thetas: np.ndarray
    final_orbital_rotation: np.ndarray | None = None

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply the operator to a vector."""
        if copy:
            vec = vec.copy()
        for target_orbs, theta in zip(
            itertools.cycle(self.interaction_pairs), self.thetas
        ):
            vec = apply_hop_gate(
                vec, theta, target_orbs=target_orbs, norb=norb, nelec=nelec, copy=False
            )
        if self.final_orbital_rotation is not None:
            vec = apply_orbital_rotation(
                vec,
                mat=self.final_orbital_rotation,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
        return vec

    def to_parameters(self) -> np.ndarray:
        """Convert the hop gate ansatz operator to a real-valued parameter vector."""
        num_params = len(self.thetas)
        if self.final_orbital_rotation is not None:
            num_params += self.norb**2
        params = np.zeros(num_params)
        params[: len(self.thetas)] = self.thetas
        if self.final_orbital_rotation is not None:
            params[len(self.thetas) :] = orbital_rotation_to_parameters(
                self.final_orbital_rotation
            )
        return params

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        norb: int,
        interaction_pairs: list[tuple[int, int]],
        with_final_orbital_rotation: bool = False,
    ) -> HopGateAnsatzOperator:
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = orbital_rotation_from_parameters(
                params[-(norb**2) :], norb
            )
            params = params[: -(norb**2)]
        return HopGateAnsatzOperator(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=params,
            final_orbital_rotation=final_orbital_rotation,
        )
