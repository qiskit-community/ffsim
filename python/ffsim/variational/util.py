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


def orbital_rotation_to_parameters(
    orbital_rotation: np.ndarray, real: bool = False
) -> np.ndarray:
    """Convert an orbital rotation to parameters.

    Converts an orbital rotation to a real-valued parameter vector. The parameter vector
    contains non-redundant real and imaginary parts of the elements of the matrix
    logarithm of the orbital rotation matrix.

    Args:
        orbital_rotation: The orbital rotation.
        real: Whether to take only the real part of the matrix logarithm of the orbital
            rotation matrix, and discard the imaginary part.

    Returns:
        The list of real numbers parameterizing the orbital rotation.
    """
    norb, _ = orbital_rotation.shape
    triu_indices = np.triu_indices(norb, k=1)
    n_triu = norb * (norb - 1) // 2
    mat = scipy.linalg.logm(orbital_rotation)
    params = np.zeros(n_triu if real else norb**2)
    # real part
    params[:n_triu] = mat[triu_indices].real
    # imaginary part
    if not real:
        triu_indices = np.triu_indices(norb)
        params[n_triu:] = mat[triu_indices].imag
    return params


def orbital_rotation_from_parameters(
    params: np.ndarray, norb: int, real: bool = False
) -> np.ndarray:
    """Construct an orbital rotation from parameters.

    Converts a real-valued parameter vector to an orbital rotation. The parameter vector
    contains non-redundant real and imaginary parts of the elements of the matrix
    logarithm of the orbital rotation matrix.

    Args:
        params: The real-valued parameters.
        norb: The number of spatial orbitals, which gives the width and height of the
            orbital rotation matrix.
        real: Whether the parameter vector describes a real-valued orbital rotation.

    Returns:
        The orbital rotation.
    """
    generator = np.zeros((norb, norb), dtype=float if real else complex)
    n_triu = norb * (norb - 1) // 2
    if not real:
        # imaginary part
        rows, cols = np.triu_indices(norb)
        vals = 1j * params[n_triu:]
        generator[rows, cols] = vals
        generator[cols, rows] = vals
    # real part
    vals = params[:n_triu]
    rows, cols = np.triu_indices(norb, k=1)
    generator[rows, cols] += vals
    generator[cols, rows] -= vals
    return scipy.linalg.expm(generator)


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
    generator[:nocc, nocc:] = t1
    generator[nocc:, :nocc] = -t1.T.conj()
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
