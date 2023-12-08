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
    norb, _ = orbital_rotation.shape
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    triu_indices_no_diag = list(itertools.combinations(range(norb), 2))
    mat = scipy.linalg.logm(orbital_rotation)
    params = np.zeros(norb**2)
    # imaginary part
    params[: len(triu_indices)] = mat[tuple(zip(*triu_indices))].imag
    # real part
    params[len(triu_indices) :] = mat[tuple(zip(*triu_indices_no_diag))].real
    return params


def orbital_rotation_from_parameters(params: np.ndarray, norb: int) -> np.ndarray:
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    triu_indices_no_diag = list(itertools.combinations(range(norb), 2))
    generator = np.zeros((norb, norb), dtype=complex)
    # imaginary part
    rows, cols = zip(*triu_indices)
    vals = 1j * params[: len(triu_indices)]
    generator[rows, cols] = vals
    generator[cols, rows] = vals
    # real part
    rows, cols = zip(*triu_indices_no_diag)
    vals = params[len(triu_indices) :]
    generator[rows, cols] += vals
    generator[cols, rows] -= vals
    return scipy.linalg.expm(generator)
