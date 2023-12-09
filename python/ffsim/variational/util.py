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


def orbital_rotation_to_parameters(orbital_rotation: np.ndarray) -> np.ndarray:
    """Convert an orbital rotation to parameters.

    Converts an orbital rotation to a real-valued parameter vector. The parameter vector
    contains non-redundant real and imaginary parts of the elements of the matrix
    logarithm of the orbital rotation matrix.

    Args:
        orbital_rotation: The orbital rotation.

    Returns:
        The list of real numbers parameterizing the orbital rotation.
    """
    norb, _ = orbital_rotation.shape
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    triu_indices_no_diag = list(itertools.combinations(range(norb), 2))
    mat = scipy.linalg.logm(orbital_rotation)
    params = np.zeros(norb**2)
    # real part
    params[: len(triu_indices_no_diag)] = mat[tuple(zip(*triu_indices_no_diag))].real
    # imaginary part
    params[len(triu_indices_no_diag) :] = mat[tuple(zip(*triu_indices))].imag
    return params


def orbital_rotation_from_parameters(params: np.ndarray, norb: int) -> np.ndarray:
    """Construct an orbital rotation from parameters.

    Converts a real-valued parameter vector to an orbital rotation. The parameter vector
    contains non-redundant real and imaginary parts of the elements of the matrix
    logarithm of the orbital rotation matrix.

    Args:
        params: The real-valued parameters.
        norb: The number of spatial orbitals, which gives the width and height of the
            orbital rotation matrix.

    Returns:
        The orbital rotation.
    """
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    triu_indices_no_diag = list(itertools.combinations(range(norb), 2))
    generator = np.zeros((norb, norb), dtype=complex)
    # imaginary part
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
