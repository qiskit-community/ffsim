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
    interaction_pairs: list[tuple[int, int]] | None, ordered: bool, *, norb: int
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
    for i, j in interaction_pairs:
        if not (0 <= i < norb and 0 <= j < norb):
            raise ValueError(
                f"Interaction pair {(i, j)} contains an orbital index outside the "
                f"range [0, {norb})."
            )


def validate_t2_pair_symmetric(
    t2: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-8
) -> None:
    """Check that restricted t2 amplitudes have the pair symmetry.

    Restricted t2 amplitudes satisfy ``t2[i, j, a, b] == t2[j, i, b, a]``.

    Args:
        t2: The t2 amplitudes.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Raises:
        ValueError: The t2 amplitudes do not have the pair symmetry.
    """
    if not np.allclose(t2, t2.transpose(1, 0, 3, 2), rtol=rtol, atol=atol):
        raise ValueError(
            "t2 amplitudes were not symmetric under exchange of the "
            "occupied-virtual pairs (i, a) and (j, b), that is, "
            "t2[i, j, a, b] == t2[j, i, b, a]."
        )


def validate_t2_antisymmetric(
    t2: np.ndarray, *, name: str = "t2", rtol: float = 1e-5, atol: float = 1e-8
) -> None:
    """Check that spinless or same-spin t2 amplitudes are antisymmetric.

    Spinless and same-spin t2 amplitudes are antisymmetric in the occupied indices
    and in the virtual indices.

    Args:
        t2: The t2 amplitudes.
        name: The name of the amplitudes to use in the error message.
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.

    Raises:
        ValueError: The t2 amplitudes are not antisymmetric.
    """
    if not np.allclose(
        t2, -t2.transpose(1, 0, 2, 3), rtol=rtol, atol=atol
    ) or not np.allclose(t2, -t2.transpose(0, 1, 3, 2), rtol=rtol, atol=atol):
        raise ValueError(
            f"{name} amplitudes were not antisymmetric in the occupied indices and in "
            "the virtual indices. Restricted t2 amplitudes, such as those computed "
            "by PySCF's RCCSD, can be converted to antisymmetric amplitudes with "
            "t2 - t2.transpose(1, 0, 2, 3)."
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
