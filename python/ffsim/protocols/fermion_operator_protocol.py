# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""FermionOperator protocol."""

from __future__ import annotations

from typing import Any, Protocol

from ffsim.operators import FermionOperator


class SupportsFermionOperator(Protocol):
    """An object that can be converted to a FermionOperator."""

    def _fermion_operator_(self) -> FermionOperator:
        """Return a FermionOperator representing the object.

        Returns:
            A FermionOperator representing the object.
        """


def fermion_operator(obj: Any) -> FermionOperator:
    """Return a FermionOperator representing the object.

    Args:
        obj: The object to convert to a LinearOperator.

    Returns:
        A FermionOperator representing the object.
    """
    method = getattr(obj, "_fermion_operator_", None)
    if method is not None:
        return method()

    raise TypeError(f"Object of type {type(obj)} has no _fermion_operator_ method.")
