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

from typing import NamedTuple

import numpy as np


class ProductStateSum(NamedTuple):
    """A linear combination of product states.

    Attributes:
        coeffs: The coefficients of the linear combination.
        states: The product states appearing in the linear combination. Each product
            state is represented as a pair of state vectors, with the first state vector
            storing the spin alpha state and the second storing the spin beta state.
    """

    coeffs: np.ndarray
    states: list[tuple[np.ndarray, np.ndarray]]
