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
import pyscf
import pyscf.cc
import pytest

import ffsim

rng = np.random.default_rng(263516597386892983361450709003146582085)


def assert_t2_has_correct_symmetry(t2: np.ndarray):
    nocc, _, nvrt, _ = t2.shape
    for i, j, a, b in itertools.product(
        range(nocc), range(nocc), range(nvrt), range(nvrt)
    ):
        np.testing.assert_allclose(t2[i, j, a, b], t2[j, i, b, a])


def test_assert_t2_has_correct_symmetry():
    """Test that t2 amplitudes from a real molecule passes our symmetry test."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["Be", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    scf = pyscf.scf.ROHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()
    t2aa, _, t2bb = ccsd.t2
    assert_t2_has_correct_symmetry(t2aa)
    assert_t2_has_correct_symmetry(t2bb)


def test_random_t2_amplitudes_symmetry():
    """Test random t2 amplitudes symmetry."""
    norb = 5
    nocc = 3
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=rng)
    assert_t2_has_correct_symmetry(t2)


def test_random_two_body_tensor_symmetry_real():
    """Test random real two-body tensor symmetry."""
    n_orbitals = 5
    two_body_tensor = ffsim.random.random_two_body_tensor(
        n_orbitals, seed=rng, dtype=float
    )
    assert np.issubdtype(two_body_tensor.dtype, np.floating)
    for i, j, k, ell in itertools.product(range(n_orbitals), repeat=4):
        val = two_body_tensor[i, j, k, ell]
        np.testing.assert_allclose(two_body_tensor[k, ell, i, j], val)
        np.testing.assert_allclose(two_body_tensor[j, i, ell, k], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[ell, k, j, i], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[j, i, k, ell], val)
        np.testing.assert_allclose(two_body_tensor[ell, k, i, j], val)
        np.testing.assert_allclose(two_body_tensor[i, j, ell, k], val)
        np.testing.assert_allclose(two_body_tensor[k, ell, j, i], val)


def test_random_two_body_tensor_symmetry():
    """Test random two-body tensor symmetry."""
    n_orbitals = 5
    two_body_tensor = ffsim.random.random_two_body_tensor(n_orbitals, seed=rng)
    for i, j, k, ell in itertools.product(range(n_orbitals), repeat=4):
        val = two_body_tensor[i, j, k, ell]
        np.testing.assert_allclose(two_body_tensor[k, ell, i, j], val)
        np.testing.assert_allclose(two_body_tensor[j, i, ell, k], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[ell, k, j, i], val.conjugate())


@pytest.mark.parametrize("dim", range(10))
def test_random_unitary(dim: int):
    """Test random unitary."""
    mat = ffsim.random.random_unitary(dim, seed=rng)
    assert mat.dtype == complex
    assert ffsim.linalg.is_unitary(mat)


@pytest.mark.parametrize("dim", range(10))
def test_random_orthogonal(dim: int):
    """Test random orthogonal."""
    mat = ffsim.random.random_orthogonal(dim, seed=rng)
    assert mat.dtype == float
    assert ffsim.linalg.is_orthogonal(mat)

    mat = ffsim.random.random_orthogonal(dim, seed=rng, dtype=complex)
    assert mat.dtype == complex
    assert ffsim.linalg.is_orthogonal(mat)


@pytest.mark.parametrize("dim", range(10))
def test_random_special_orthogonal(dim: int):
    """Test random special orthogonal."""
    mat = ffsim.random.random_special_orthogonal(dim, seed=rng)
    assert mat.dtype == float
    assert ffsim.linalg.is_special_orthogonal(mat)

    mat = ffsim.random.random_special_orthogonal(dim, seed=rng, dtype=np.float32)
    assert mat.dtype == np.float32
    assert ffsim.linalg.is_special_orthogonal(mat, atol=1e-5)


def test_random_real_symmetric_matrix():
    """Test random real symmetric matrix."""
    dim = 5
    mat = ffsim.random.random_real_symmetric_matrix(dim, seed=rng)
    assert ffsim.linalg.is_real_symmetric(mat)
    np.testing.assert_allclose(np.linalg.matrix_rank(mat), dim)

    rank = 3
    mat = ffsim.random.random_real_symmetric_matrix(dim, rank=rank, seed=rng)
    assert ffsim.linalg.is_real_symmetric(mat)
    np.testing.assert_allclose(np.linalg.matrix_rank(mat), rank)


@pytest.mark.parametrize("dim", range(10))
def test_random_antihermitian_matrix(dim: int):
    """Test random anti-Hermitian matrix."""
    mat = ffsim.random.random_antihermitian(dim, seed=rng)
    assert ffsim.linalg.is_antihermitian(mat)


@pytest.mark.parametrize("dim", range(1, 10))
def test_random_state_vector(dim: int):
    """Test random state vector."""
    vec = ffsim.random.random_state_vector(dim, seed=rng)
    assert vec.dtype == complex
    np.testing.assert_allclose(np.linalg.norm(vec), 1)

    vec = ffsim.random.random_state_vector(dim, seed=rng, dtype=float)
    assert vec.dtype == float
    np.testing.assert_allclose(np.linalg.norm(vec), 1)


@pytest.mark.parametrize("dim", range(1, 10))
def test_random_density_matrix(dim: int):
    """Test random density matrix."""
    mat = ffsim.random.random_density_matrix(dim, seed=rng)
    assert mat.dtype == complex
    assert ffsim.linalg.is_hermitian(mat)
    eigs, _ = np.linalg.eigh(mat)
    assert all(eigs >= 0)
    np.testing.assert_allclose(np.trace(mat), 1)

    mat = ffsim.random.random_density_matrix(dim, seed=rng, dtype=float)
    assert mat.dtype == float
    assert ffsim.linalg.is_hermitian(mat)
    eigs, _ = np.linalg.eigh(mat)
    assert all(eigs >= 0)
    np.testing.assert_allclose(np.trace(mat), 1)


def test_random_diagonal_coulomb_hamiltonian():
    """Test random diagonal Coulomb Hamiltonian."""
    norb = 5

    dc_ham = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=rng)
    assert dc_ham.one_body_tensor.dtype == complex

    dc_ham = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=rng, real=True)
    assert dc_ham.one_body_tensor.dtype == float


def test_raise_errors():
    """Test errors are raised as expected."""
    with pytest.raises(ValueError, match="Dimension"):
        _ = ffsim.random.random_state_vector(0, seed=rng)

    with pytest.raises(ValueError, match="Dimension"):
        _ = ffsim.random.random_density_matrix(0, seed=rng)
