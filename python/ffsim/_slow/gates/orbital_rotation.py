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
from scipy.linalg.lapack import zrot


def apply_givens_rotation_in_place_slow(
    vec: np.ndarray,
    c: float,
    s: complex,
    slice1: np.ndarray,
    slice2: np.ndarray,
) -> None:
    """Apply a Givens rotation to slices of a state vector."""
    for i, j in zip(slice1, slice2):
        vec[i], vec[j] = zrot(vec[i], vec[j], c, s)
