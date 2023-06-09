# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for double factorization utilities."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import ao2mo, gto, mcscf, scf

from ffsim.linalg import (
    double_factorized,
    modified_cholesky,
)
from ffsim.linalg.double_factorized import optimal_diag_coulomb_mats
from ffsim.random import random_two_body_tensor_real, random_unitary


@pytest.mark.parametrize("dim", [4, 5])
def test_modified_cholesky(dim: int):
    """Test modified Cholesky decomposition on a random tensor."""
    rng = np.random.default_rng()
    # construct a random positive definite matrix
    unitary = np.array(random_unitary(dim, seed=rng))
    eigs = rng.uniform(size=dim)
    mat = unitary @ np.diag(eigs) @ unitary.T.conj()
    cholesky_vecs = modified_cholesky(mat)
    reconstructed = np.einsum("ji,ki->jk", cholesky_vecs, cholesky_vecs.conj())
    np.testing.assert_allclose(reconstructed, mat, atol=1e-8)


@pytest.mark.parametrize(
    "dim, cholesky", [(4, True), (4, False), (5, True), (5, False)]
)
def test_double_factorized_random(dim: int, cholesky: bool):
    """Test low rank two-body decomposition on a random tensor."""
    two_body_tensor = random_two_body_tensor_real(dim)
    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, cholesky=cholesky
    )
    reconstructed = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)


@pytest.mark.parametrize("cholesky", [True, False])
def test_double_factorized_tol_max_vecs(cholesky: bool):
    """Test low rank decomposition error threshold and max vecs."""
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[["Li", (0, 0, 0)], ["H", (1.6, 0, 0)]],
        basis="sto-3g",
    )
    hartree_fock = scf.RHF(mol)
    hartree_fock.kernel()
    norb, _ = hartree_fock.mo_coeff.shape
    mc = mcscf.CASCI(hartree_fock, norb, mol.nelec)
    two_body_tensor = ao2mo.restore(1, mc.get_h2cas(), mc.ncas)

    # test max_vecs
    max_vecs = 20
    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=max_vecs, cholesky=cholesky
    )
    reconstructed = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    assert len(orbital_rotations) == max_vecs
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-5)

    # test error threshold
    tol = 1e-4
    diag_coulomb_mats, orbital_rotations = double_factorized(two_body_tensor, tol=tol)
    reconstructed = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    assert len(orbital_rotations) <= 18
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=tol)

    # test error threshold and max vecs
    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, tol=tol, max_vecs=max_vecs
    )
    reconstructed = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    assert len(orbital_rotations) <= 18
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=tol)


def test_optimal_diag_coulomb_mats_exact():
    """Test optimal diag Coulomb matrices on exact decomposition."""
    dim = 5
    two_body_tensor = random_two_body_tensor_real(dim)

    _, orbital_rotations = double_factorized(two_body_tensor)
    diag_coulomb_mats_optimal = optimal_diag_coulomb_mats(
        two_body_tensor, orbital_rotations
    )
    reconstructed = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats_optimal,
        orbital_rotations,
        orbital_rotations,
    )
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)


def test_optimal_diag_coulomb_mats_approximate():
    """Test optimal diag Coulomb matrices on approximate decomposition."""
    dim = 5
    two_body_tensor = random_two_body_tensor_real(dim)

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=3
    )
    diag_coulomb_mats_optimal = optimal_diag_coulomb_mats(
        two_body_tensor, orbital_rotations
    )
    reconstructed = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats_optimal,
        orbital_rotations,
        orbital_rotations,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimal = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimal < error


def test_double_factorized_compressed():
    """Test compressed double factorization"""
    # TODO test on simple molecule like ethylene
    dim = 2
    two_body_tensor = random_two_body_tensor_real(dim)

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_mats_optimized, orbital_rotations_optimized = double_factorized(
        two_body_tensor, max_vecs=2, optimize=True
    )
    reconstructed = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < error


def test_double_factorized_compressed_constrained():
    """Test constrained compressed double factorization"""
    # TODO test on simple molecule like ethylene
    dim = 3
    two_body_tensor = random_two_body_tensor_real(dim)

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_mask = np.sum(
        [np.diag(np.ones(dim - abs(k)), k=k) for k in range(-1, 2)],
        axis=0,
        dtype=bool,
    )
    diag_coulomb_mats_optimized, orbital_rotations_optimized = double_factorized(
        two_body_tensor, max_vecs=2, optimize=True, diag_coulomb_mask=diag_coulomb_mask
    )
    np.testing.assert_allclose(
        diag_coulomb_mats_optimized, diag_coulomb_mats_optimized * diag_coulomb_mask
    )
    reconstructed = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < error
