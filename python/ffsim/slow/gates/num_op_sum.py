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

import numpy as np


def apply_num_op_sum_evolution_in_place_slow(
    vec: np.ndarray, phases: np.ndarray, occupations: np.ndarray
) -> None:
    """Apply time evolution by a sum of number operators in-place."""
    for row, orbs in zip(vec, occupations):
        phase = 1
        for orb in orbs:
            phase *= phases[orb]
        row *= phase
