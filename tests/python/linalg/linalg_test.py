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
import scipy.linalg
import scipy.sparse

import ffsim

RNG = np.random.default_rng(21131415246441775736488838123610662270)


def test_lup():
    dim = 5
    mat = RNG.standard_normal((dim, dim)) + 1j * RNG.standard_normal((dim, dim))
    ell, u, p = ffsim.linalg.lup(mat)
    np.testing.assert_allclose(ell @ u @ p, mat)
    np.testing.assert_allclose(np.diagonal(ell), np.ones(dim))


def test_reduced_matrix():
    big_dim = 20
    small_dim = 5
    mat = scipy.sparse.random(big_dim, big_dim, random_state=RNG)
    vecs = [
        RNG.standard_normal(big_dim) + 1j * RNG.standard_normal(big_dim)
        for _ in range(small_dim)
    ]
    reduced_mat = ffsim.linalg.reduced_matrix(mat, vecs)
    for i, j in itertools.product(range(small_dim), repeat=2):
        actual = reduced_mat[i, j]
        expected = np.vdot(vecs[i], mat @ vecs[j])
        np.testing.assert_allclose(actual, expected)


def test_logm_unitary():
    for dim in range(10):
        mat = ffsim.random.random_unitary(dim, seed=RNG)
        log = ffsim.linalg.logm_unitary(mat)
        assert ffsim.linalg.is_antihermitian(log)
        np.testing.assert_allclose(scipy.linalg.expm(log), mat)
        if dim:
            # agrees with the general-purpose matrix logarithm
            np.testing.assert_allclose(log, scipy.linalg.logm(mat), atol=1e-12)


def test_logm_unitary_orthogonal():
    for dim in range(10):
        mat = ffsim.random.random_orthogonal(dim, seed=RNG)
        log = ffsim.linalg.logm_unitary(mat)
        assert ffsim.linalg.is_antihermitian(log)
        np.testing.assert_allclose(scipy.linalg.expm(log), mat, atol=1e-12)
        # The logarithm is not compared against scipy.linalg.logm here. An orthogonal
        # matrix with determinant -1 has an eigenvalue of -1, whose principal logarithm
        # can be either of the equally valid values i*pi and -i*pi, and the two
        # functions do not always make the same choice.


def test_match_global_phase():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = 1j * a
    c, d = ffsim.linalg.match_global_phase(a, b)
    np.testing.assert_allclose(c, d)

    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = 2j * a
    c, d = ffsim.linalg.match_global_phase(a, b)
    np.testing.assert_allclose(2 * c, d)
