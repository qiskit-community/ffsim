# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ProductStateSum NamedTuple."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ProductStateSum(NamedTuple):
    """A linear combination of product states.

    Given a ProductStateSum ``prod_state_sum``, the full state vector can be
    reconstructed as

    .. code-block:: python

        sum(
            coeff * np.kron(vec_a, vec_b)
            for coeff, (vec_a, vec_b) in zip(
                prod_state_sum.coeffs, prod_state_sum.states
            )
        )
    """

    coeffs: np.ndarray
    states: list[tuple[np.ndarray, np.ndarray]]
