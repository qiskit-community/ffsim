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

from ffsim.gates import apply_hop_gate


@dataclass
class HopGateAnsatzOperator:
    r"""A hop gate ansatz operator."""

    interaction_pairs: list[tuple[int, int]]
    thetas: np.ndarray

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
        return vec
