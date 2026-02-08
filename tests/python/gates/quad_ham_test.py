# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for quadratic Hamiltonian evolution."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg

import ffsim

RNG = np.random.default_rng(221925319548051244210434403650365857976)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(1, 5)))
def test_apply_quad_ham_evolution_spinless(norb: int, nelec: int):
    """Test applying time evolution of a quadratic Hamiltonian, spin symmetric."""
    mat = ffsim.random.random_hermitian(norb, seed=RNG)
    time = RNG.standard_normal()
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    # Use apply_quad_ham_evolution
    result = ffsim.apply_quad_ham_evolution(vec, mat, time, norb=norb, nelec=nelec)
    # Use apply_num_op_sum_evolution
    eigs, vecs = scipy.linalg.eigh(mat)
    expected = ffsim.apply_num_op_sum_evolution(
        vec, eigs, time, norb, nelec, orbital_rotation=vecs
    )
    # Check that the results match
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_apply_quad_ham_evolution_spinful_symm(norb: int, nelec: tuple[int, int]):
    """Test applying time evolution of a quadratic Hamiltonian, spin symmetric."""
    mat = ffsim.random.random_hermitian(norb, seed=RNG)
    time = RNG.standard_normal()
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    # Use apply_quad_ham_evolution
    result = ffsim.apply_quad_ham_evolution(vec, mat, time, norb=norb, nelec=nelec)
    # Use apply_num_op_sum_evolution
    eigs, vecs = scipy.linalg.eigh(mat)
    expected = ffsim.apply_num_op_sum_evolution(
        vec, eigs, time, norb, nelec, orbital_rotation=vecs
    )
    # Check that the results match
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 5)))
def test_apply_quad_ham_evolution_spinful_asymm(norb: int, nelec: tuple[int, int]):
    """Test applying time evolution of a quadratic Hamiltonian, spin asymmetric."""
    mat_a = ffsim.random.random_hermitian(norb, seed=RNG)
    mat_b = ffsim.random.random_hermitian(norb, seed=RNG)
    time = RNG.standard_normal()
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    # Both spin sectors
    result = ffsim.apply_quad_ham_evolution(
        vec, (mat_a, mat_b), time, norb=norb, nelec=nelec
    )
    eigs_a, vecs_a = scipy.linalg.eigh(mat_a)
    eigs_b, vecs_b = scipy.linalg.eigh(mat_b)
    expected = ffsim.apply_num_op_sum_evolution(
        vec, (eigs_a, eigs_b), time, norb, nelec, orbital_rotation=(vecs_a, vecs_b)
    )
    np.testing.assert_allclose(result, expected)

    # Only spin alpha
    result = ffsim.apply_quad_ham_evolution(
        vec, (mat_a, None), time, norb=norb, nelec=nelec
    )
    eigs_a, vecs_a = scipy.linalg.eigh(mat_a)
    eigs_b, vecs_b = scipy.linalg.eigh(mat_b)
    expected = ffsim.apply_num_op_sum_evolution(
        vec, (eigs_a, None), time, norb, nelec, orbital_rotation=(vecs_a, None)
    )
    np.testing.assert_allclose(result, expected)

    # Only spin beta
    result = ffsim.apply_quad_ham_evolution(
        vec, (None, mat_b), time, norb=norb, nelec=nelec
    )
    eigs_a, vecs_a = scipy.linalg.eigh(mat_a)
    eigs_b, vecs_b = scipy.linalg.eigh(mat_b)
    expected = ffsim.apply_num_op_sum_evolution(
        vec, (None, eigs_b), time, norb, nelec, orbital_rotation=(None, vecs_b)
    )
    np.testing.assert_allclose(result, expected)
