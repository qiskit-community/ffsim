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
import scipy.sparse

import ffsim


def test_lup():
    dim = 5
    rng = np.random.default_rng()
    mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    ell, u, p = ffsim.linalg.lup(mat)
    np.testing.assert_allclose(ell @ u @ p, mat)
    np.testing.assert_allclose(np.diagonal(ell), np.ones(dim))


def test_reduced_matrix():
    big_dim = 20
    small_dim = 5
    rng = np.random.default_rng()
    mat = scipy.sparse.random(big_dim, big_dim, random_state=rng)
    vecs = [
        rng.standard_normal(big_dim) + 1j * rng.standard_normal(big_dim)
        for _ in range(small_dim)
    ]
    reduced_mat = ffsim.linalg.reduced_matrix(mat, vecs)
    for i, j in itertools.product(range(small_dim), repeat=2):
        actual = reduced_mat[i, j]
        expected = np.vdot(vecs[i], mat @ vecs[j])
        np.testing.assert_allclose(actual, expected)


def test_match_global_phase():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = 1j * a
    c, d = ffsim.linalg.match_global_phase(a, b)
    np.testing.assert_allclose(c, d)

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = 2j * a
    c, d = ffsim.linalg.match_global_phase(a, b)
    np.testing.assert_allclose(2 * c, d)
