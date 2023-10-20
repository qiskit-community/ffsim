# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Apply unitary protocol."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SupportsApplyUnitary(Protocol):
    """An object that can apply a unitary transformation to a vector."""

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Apply a unitary transformation to a vector.

        Args:
            vec: The vector to apply the unitary transformation to.
            norb: The number of spatial orbitals.
            nelec: The number of alpha and beta electrons.
            copy: Whether to copy the vector before operating on it.
                - If ``copy=True`` then this function always returns a newly allocated
                vector and the original vector is left untouched.
                - If ``copy=False`` then this function may still return a newly
                allocated vector, but the original vector may have its data overwritten.
                It is also possible that the original vector is returned,
                modified in-place.

        Returns:
            The transformed vector.
        """


def apply_unitary(
    vec: np.ndarray, obj: Any, norb: int, nelec: tuple[int, int], copy: bool = True
) -> np.ndarray:
    """Apply a unitary transformation to a vector.

    Args:
        vec: The vector to apply the unitary transformation to.
        obj: The object with a unitary effect.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly
            allocated vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.

    Returns:
        The transformed vector.
    """
    method = getattr(obj, "_apply_unitary_", None)
    if method is not None:
        return method(vec, norb=norb, nelec=nelec, copy=copy)

    raise TypeError(f"Object of type {type(obj)} has no _apply_unitary_ method.")
