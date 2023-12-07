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
from pyscf import cc, gto, scf

import ffsim


def assert_t2_has_correct_symmetry(t2: np.ndarray):
    nocc, _, nvrt, _ = t2.shape
    norb = nocc + nvrt
    pairs = itertools.product(range(nocc), range(nocc, norb))
    for (i, a), (j, b) in itertools.product(pairs, repeat=2):
        np.testing.assert_allclose(
            t2[i, j, a - nocc, b - nocc], t2[j, i, b - nocc, a - nocc]
        )


def test_assert_t2_has_correct_symmetry():
    """Test that t2 amplitudes from a real molecule passes our symmetry test."""
    # Build a stretched ethene molecule
    bond_distance = 2.678
    a = 0.5 * bond_distance
    b = a + 0.5626
    c = 0.9289
    mol = gto.Mole()
    mol.build(
        atom=[
            ["C", (0, 0, a)],
            ["C", (0, 0, -a)],
            ["H", (0, c, b)],
            ["H", (0, -c, b)],
            ["H", (0, c, -b)],
            ["H", (0, -c, -b)],
        ],
        basis="sto-6g",
        symmetry="d2h",
    )
    hartree_fock = scf.RHF(mol)
    hartree_fock.kernel()
    # Define active space
    active_space = range(mol.nelectron // 2 - 2, mol.nelectron // 2 + 2)
    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = cc.CCSD(
        hartree_fock,
        frozen=[i for i in range(mol.nao_nr()) if i not in active_space],
    )
    _, _, t2 = ccsd.kernel()
    assert_t2_has_correct_symmetry(t2)


def test_random_t2_amplitudes_symmetry():
    """Test random t2 amplitudes symmetry."""
    norb = 5
    nocc = 3
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc)
    assert_t2_has_correct_symmetry(t2)


def test_random_two_body_tensor_symmetry_real():
    """Test random real two-body tensor symmetry."""
    n_orbitals = 5
    two_body_tensor = ffsim.random.random_two_body_tensor(n_orbitals, dtype=float)
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
    two_body_tensor = ffsim.random.random_two_body_tensor(n_orbitals)
    for i, j, k, ell in itertools.product(range(n_orbitals), repeat=4):
        val = two_body_tensor[i, j, k, ell]
        np.testing.assert_allclose(two_body_tensor[k, ell, i, j], val)
        np.testing.assert_allclose(two_body_tensor[j, i, ell, k], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[ell, k, j, i], val.conjugate())


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
