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

from pathlib import Path

import numpy as np
from scipy.linalg.lapack import zrot

import ffsim
from ffsim.linalg import givens_decomposition


def test_givens_decomposition():
    dim = 5
    rng = np.random.default_rng()
    for _ in range(5):
        mat = ffsim.random.random_unitary(dim, seed=rng)
        givens_rotations, phase_shifts = givens_decomposition(mat)
        reconstructed = np.eye(dim, dtype=complex)
        for i, phase_shift in enumerate(phase_shifts):
            reconstructed[i] *= phase_shift
        for (c, s), (i, j) in givens_rotations[::-1]:
            reconstructed = reconstructed.T.copy()
            reconstructed[j], reconstructed[i] = zrot(
                reconstructed[j], reconstructed[i], c, s.conjugate()
            )
            reconstructed = reconstructed.T

        np.testing.assert_allclose(reconstructed, mat)


def test_givens_decomposition_no_side_effects():
    """Test that the Givens decomposition doesn't modify the original matrix."""
    norb = 8
    rng = np.random.default_rng()
    for _ in range(5):
        mat = ffsim.random.random_unitary(norb, seed=rng)
        original_mat = mat.copy()
        _ = givens_decomposition(mat)

        assert ffsim.linalg.is_unitary(original_mat)
        assert ffsim.linalg.is_unitary(mat)
        np.testing.assert_allclose(mat, original_mat, atol=1e-12)


def test_givens_decomposition_no_side_effects_special_case():
    """Test that the Givens decomposition doesn't modify the original matrix."""
    datadir = Path(__file__).parent.parent / "test_data"
    filepath = datadir / "orbital_rotation-0.npy"

    with open(filepath, "rb") as f:
        mat = np.load(f)
    assert ffsim.linalg.is_unitary(mat, atol=1e-12)

    original_mat = mat.copy()
    _ = givens_decomposition(mat)

    assert ffsim.linalg.is_unitary(original_mat)
    assert ffsim.linalg.is_unitary(mat)
    np.testing.assert_allclose(mat, original_mat, atol=1e-12)
