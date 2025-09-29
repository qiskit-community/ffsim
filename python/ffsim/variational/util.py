# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Variational ansatz utilities."""

from __future__ import annotations

import numpy as np
import scipy.linalg


def validate_interaction_pairs(
    interaction_pairs: list[tuple[int, int]] | None, ordered: bool
) -> None:
    if interaction_pairs is None:
        return
    if len(set(interaction_pairs)) != len(interaction_pairs):
        raise ValueError(
            f"Duplicate interaction pairs encountered: {interaction_pairs}."
        )
    if not ordered:
        for i, j in interaction_pairs:
            if i > j:
                raise ValueError(
                    "When specifying spinless, alpha-alpha or beta-beta "
                    "interaction pairs, you must provide only upper triangular pairs. "
                    f"Got {(i, j)}, which is a lower triangular pair."
                )


def orbital_rotation_from_t1_amplitudes(t1: np.ndarray) -> np.ndarray:
    """Construct an orbital rotation from t1 amplitudes.

    The orbital rotation is constructed as exp(t1 - t1†).

    Args:
        t1: The t1 amplitudes.

    Returns:
        The orbital rotation.
    """
    nocc, nvrt = t1.shape
    norb = nocc + nvrt
    generator = np.zeros((norb, norb), dtype=t1.dtype)
    generator[:nocc, nocc:] = -t1.conj()
    generator[nocc:, :nocc] = t1.T
    return scipy.linalg.expm(generator)


def interaction_pairs_spin_balanced(
    connectivity: str, norb: int
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]:
    """Returns alpha-alpha and alpha-beta diagonal Coulomb interaction pairs."""
    if connectivity == "all-to-all":
        pairs_aa = None
        pairs_ab = None
    elif connectivity == "square":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb)]
    elif connectivity == "hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 2 == 0]
    elif connectivity == "heavy-hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 4 == 0]
    else:
        raise ValueError(f"Invalid connectivity: {connectivity}")
    return pairs_aa, pairs_ab
