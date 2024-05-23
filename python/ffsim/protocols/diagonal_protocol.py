# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Diagonal protocol."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SupportsDiagonal(Protocol):
    """A linear operator whose diagonal entries can be returned."""

    def _diag_(self, norb: int, nelec: tuple[int, int]) -> np.ndarray:
        """Return the diagonal entries of the linear operator.

        Args:
            norb: The number of spatial orbitals.
            nelec: The number of alpha and beta electrons.

        Returns:
            The diagonal entries of the linear operator.
        """


def diag(obj: Any, norb: int, nelec: tuple[int, int]) -> float:
    """Return the diagonal entries of the linear operator."""
    method = getattr(obj, "_diag_", None)
    if method is not None:
        return method(norb=norb, nelec=nelec)
    raise TypeError(
        f"Could not compute diagonal entries of object of type {type(obj)}.\n"
        "The object did not have a _diag_ method that returned its diagonal entries."
    )
