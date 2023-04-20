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

import numpy as np
import pytest
from pyscf import ao2mo, gto, mcscf, scf

from ffsim.linalg import (
    apply_matrix_to_slices,
    double_factorized,
    givens_decomposition,
    lup,
    modified_cholesky,
)
from ffsim.random_utils import random_two_body_tensor_real, random_unitary


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


def test_lup():
    dim = 5
    rng = np.random.default_rng()
    mat = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    ell, u, p = lup(mat)
    np.testing.assert_allclose(ell @ u @ p, mat)
    np.testing.assert_allclose(np.diagonal(ell), np.ones(dim))


@pytest.mark.parametrize("dim", [4, 5])
def test_modified_cholesky(dim: int):
    """Test modified Cholesky decomposition on a random tensor."""
    rng = np.random.default_rng(4640)
    # construct a random positive definite matrix
    unitary = np.array(random_unitary(dim, seed=rng))
    eigs = rng.uniform(size=dim)
    mat = unitary @ np.diag(eigs) @ unitary.T.conj()
    cholesky_vecs = modified_cholesky(mat)
    reconstructed = np.einsum("ji,ki->jk", cholesky_vecs, cholesky_vecs.conj())
    np.testing.assert_allclose(reconstructed, mat, atol=1e-8)


@pytest.mark.parametrize("dim", [4, 5])
def test_double_factorized_random(dim: int):
    """Test low rank two-body decomposition on a random tensor."""
    two_body_tensor = random_two_body_tensor_real(dim, seed=25257)
    diag_coulomb_mats, orbital_rotations = double_factorized(two_body_tensor)
    reconstructed = np.einsum(
        "tpk,tqk,tkl,trl,tsl->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)


def test_double_factorized_error_threshold_max_vecs():
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
        two_body_tensor, max_vecs=max_vecs
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
    error_threshold = 1e-4
    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, error_threshold=error_threshold
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
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=error_threshold)

    # test error threshold and max vecs
    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, error_threshold=error_threshold, max_vecs=max_vecs
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
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=error_threshold)
