# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Trace protocol."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SupportsTrace(Protocol):
    """A linear operator whose trace can be computed."""

    def _trace_(self, norb: int, nelec: tuple[int, int]) -> float:
        """Return the trace of the linear operator.

        Args:
            norb: The number of spatial orbitals.
            nelec: The number of alpha and beta electrons.

        Returns:
            The trace of the linear operator.
        """


def trace(obj: Any, norb: int, nelec: tuple[int, int]) -> float:
    """Return the trace of the linear operator."""
    method = getattr(obj, "_trace_", None)
    if method is not None:
        return method(norb=norb, nelec=nelec)
    method = getattr(obj, "_diag_", None)
    if method is not None:
        return np.sum(method(norb=norb, nelec=nelec))
    raise TypeError(
        f"Could not compute trace of object of type {type(obj)}.\n"
        "The object did not have a _trace_ method that returned the trace "
        "or a _diag_ method that returned its diagonal entries."
    )
