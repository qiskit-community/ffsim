# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Approximate equality protocol."""

from __future__ import annotations

from typing import Any, Protocol


class SupportsApproximateEquality(Protocol):
    """An object that can be compared approximately."""

    def _approx_eq_(self, other: Any, rtol: float, atol: float) -> bool:
        """Return whether the object is approximately equal to another object.

        See the documentation of `np.isclose`_ for the interpretation of the tolerance
        parameters ``rtol`` and ``atol``.

        Args:
            other: The object to compare to.
            rtol: Relative numerical tolerance.
            atol: Absolute numerical tolerance.

        Returns:
            True if the objects are approximately equal up to the specified tolerance,
            and False otherwise.

        .. _np.isclose: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
        """


def approx_eq(obj: Any, other: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Return whether two objects are approximately equal.

    See the documentation of `np.isclose`_ for the interpretation of the tolerance
    parameters ``rtol`` and ``atol``.

    Args:
        obj: The first object.
        other: The object to compare to.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Returns:
        True if the objects are approximately equal up to the specified tolerance,
        and False otherwise.

    .. _np.isclose: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    """
    method = getattr(obj, "_approx_eq_", None)
    if method is not None:
        result = method(other, rtol=rtol, atol=atol)
        if result is not NotImplemented:
            return result

    method = getattr(other, "_approx_eq_", None)
    if method is not None:
        result = method(obj, rtol=rtol, atol=atol)
        if result is not NotImplemented:
            return result

    return obj == other
