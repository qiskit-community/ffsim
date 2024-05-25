# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Enumeration for indicating alpha, beta, or both spins."""

from __future__ import annotations

from enum import Flag, auto
from typing import TypeVar


class Spin(Flag):
    """Enumeration for indicating alpha, beta, or both spins."""

    ALPHA = auto()
    """Use this to indicate spin alpha."""

    BETA = auto()
    """Use this to indicate spin beta."""

    ALPHA_AND_BETA = ALPHA | BETA
    """Use this to indicate both spin alpha and spin beta."""


T = TypeVar("T")


def pair_for_spin(obj: T, spin: Spin) -> tuple[T | None, T | None]:
    """Create a pair of objects for a given spin setting.

    Given an object, return a pair where each element of the pair is either the object
    or `None`, depending on the specified `spin`. The first element of the pair
    corresponds to spin alpha and the second corresponds to spin beta.
    """
    if spin is Spin.ALPHA:
        return (obj, None)
    if spin is Spin.BETA:
        return (None, obj)
    return (obj, obj)
