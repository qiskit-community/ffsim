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


def contract_num_op_sum_spin_into_buffer_slow(
    vec: np.ndarray, coeffs: np.ndarray, occupations: np.ndarray, out: np.ndarray
) -> None:
    for source_row, target_row, orbs in zip(vec, out, occupations):
        coeff = 0
        for orb in orbs:
            coeff += coeffs[orb]
        target_row += coeff * source_row
