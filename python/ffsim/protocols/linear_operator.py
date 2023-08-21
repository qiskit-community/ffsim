# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear operator protocol."""

from __future__ import annotations

from typing import Any, Protocol

from scipy.sparse.linalg import LinearOperator


class SupportsLinearOperator(Protocol):
    """An object that can be converted to a SciPy LinearOperator."""

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object.

        Args:
            norb: The number of spatial orbitals.
            nelec: The number of alpha and beta electrons.

        Returns:
            A Scipy LinearOperator representing the object.
        """


def linear_operator(obj: Any, norb: int, nelec: tuple[int, int]) -> LinearOperator:
    """Return a SciPy LinearOperator representing the object."""
    method = getattr(obj, "_linear_operator_", None)
    if method is not None:
        return method(norb=norb, nelec=nelec)
    raise TypeError(f"Object of type {type(obj)} has no _linear_operator_ method.")
