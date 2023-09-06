# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for linear algebra utilities."""

from __future__ import annotations

import itertools

import numpy as np

import ffsim


def test_random_two_body_tensor_symmetry():
    """Test random two-body tensor symmetry."""
    n_orbitals = 5
    two_body_tensor = ffsim.random.random_two_body_tensor_real(n_orbitals)
    for p, q, r, s in itertools.product(range(n_orbitals), repeat=4):
        val = two_body_tensor[p, q, r, s]
        np.testing.assert_allclose(two_body_tensor[r, s, p, q], val)
        np.testing.assert_allclose(two_body_tensor[q, p, s, r], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[s, r, q, p], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[q, p, r, s], val)
        np.testing.assert_allclose(two_body_tensor[s, r, p, q], val)
        np.testing.assert_allclose(two_body_tensor[p, q, s, r], val)
        np.testing.assert_allclose(two_body_tensor[r, s, q, p], val)


def test_random_t2_amplitudes_symmetry():
    """Test random t2 amplitudes symmetry."""
    # TODO t2 amplitudes from pySCF don't satisfy these symmetries
    norb = 5
    nocc = 3
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc)
    for i, j in itertools.product(range(nocc), repeat=2):
        for a, b in itertools.product(range(norb - nocc), repeat=2):
            val = t2[i, j, a, b]
            np.testing.assert_allclose(t2[i, j, b, a], -val)
            np.testing.assert_allclose(t2[j, i, a, b], -val)
            np.testing.assert_allclose(t2[j, i, b, a], val)


def test_random_unitary():
    """Test random unitary."""
    dim = 5
    mat = ffsim.random.random_unitary(dim)
    assert ffsim.linalg.is_unitary(mat)


def test_random_orthogonal():
    """Test random orthogonal."""
    dim = 5
    mat = ffsim.random.random_orthogonal(dim)
    assert ffsim.linalg.is_orthogonal(mat)
    assert mat.dtype == float

    mat = ffsim.random.random_orthogonal(dim, dtype=complex)
    assert ffsim.linalg.is_orthogonal(mat)
    assert mat.dtype == complex


def test_random_special_orthogonal():
    """Test random special orthogonal."""
    dim = 5
    mat = ffsim.random.random_special_orthogonal(dim)
    assert ffsim.linalg.is_special_orthogonal(mat)
    assert mat.dtype == float

    mat = ffsim.random.random_special_orthogonal(dim, dtype=np.float32)
    assert ffsim.linalg.is_special_orthogonal(mat, atol=1e-5)
    assert mat.dtype == np.float32


def test_random_real_symmetric_matrix():
    """Test random real symmetric matrix."""
    dim = 5
    mat = ffsim.random.random_real_symmetric_matrix(dim)
    assert ffsim.linalg.is_real_symmetric(mat)
    np.testing.assert_allclose(np.linalg.matrix_rank(mat), dim)

    rank = 3
    mat = ffsim.random.random_real_symmetric_matrix(dim, rank=rank)
    assert ffsim.linalg.is_real_symmetric(mat)
    np.testing.assert_allclose(np.linalg.matrix_rank(mat), rank)


def test_random_antihermitian_matrix():
    """Test random anti-Hermitian matrix."""
    dim = 5
    mat = ffsim.random.random_antihermitian(dim)
    assert ffsim.linalg.is_antihermitian(mat)
