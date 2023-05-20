# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for givens decomposition utilities."""

from __future__ import annotations

import numpy as np

from ffsim.linalg import (
    apply_matrix_to_slices,
    givens_decomposition,
)
from ffsim.random import random_unitary


def test_givens_decomposition():
    dim = 5
    mat = random_unitary(dim)
    givens_rotations, phase_shifts = givens_decomposition(mat)
    reconstructed = np.eye(dim, dtype=complex)
    for i, phase_shift in enumerate(phase_shifts):
        reconstructed[i] *= phase_shift
    for givens_mat, (i, j) in givens_rotations[::-1]:
        reconstructed = apply_matrix_to_slices(
            reconstructed, givens_mat.conj(), ((Ellipsis, j), (Ellipsis, i))
        )
    np.testing.assert_allclose(reconstructed, mat, atol=1e-8)
