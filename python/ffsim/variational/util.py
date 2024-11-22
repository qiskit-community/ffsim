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

import itertools

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
        real: Whether to construct a parameter vector for a real-valued
            orbital rotation. If True, the orbital rotation must have a real-valued
            data type.

    Returns:
        The list of real numbers parameterizing the orbital rotation.
    """
    if real and np.iscomplexobj(orbital_rotation):
        raise TypeError(
            "real was set to True, but the orbital rotation has a complex data type. "
            "Try passing an orbital rotation with a real-valued data type, or else "
            "set real=False."
        )
    norb, _ = orbital_rotation.shape
    triu_indices_no_diag = list(itertools.combinations(range(norb), 2))
    mat = scipy.linalg.logm(orbital_rotation)
    params = np.zeros(norb * (norb - 1) // 2 if real else norb**2)
    # real part
    params[: len(triu_indices_no_diag)] = mat[tuple(zip(*triu_indices_no_diag))].real
    # imaginary part
    if not real:
        triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
        params[len(triu_indices_no_diag) :] = mat[tuple(zip(*triu_indices))].imag
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
    triu_indices_no_diag = list(itertools.combinations(range(norb), 2))
    generator = np.zeros((norb, norb), dtype=float if real else complex)
    if not real:
        # imaginary part
        triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
        vals = 1j * params[len(triu_indices_no_diag) :]
        rows, cols = zip(*triu_indices)
        generator[rows, cols] = vals
        generator[cols, rows] = vals
    # real part
    vals = params[: len(triu_indices_no_diag)]
    rows, cols = zip(*triu_indices_no_diag)
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
