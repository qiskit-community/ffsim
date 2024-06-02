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

import itertools

import numpy as np
import pytest
import scipy.linalg
from opt_einsum import contract
from pyscf import ao2mo, cc, gto, mcscf, scf

import ffsim
from ffsim.linalg import (
    double_factorized,
    modified_cholesky,
)
from ffsim.linalg.double_factorized_decomposition import (
    double_factorized_t2,
    optimal_diag_coulomb_mats,
)
from ffsim.random import random_t2_amplitudes, random_unitary


def reconstruct_t2(
    diag_coulomb_mats: np.ndarray, orbital_rotations: np.ndarray, nocc: int
) -> np.ndarray:
    return (
        1j
        * contract(
            "mkpq,mkap,mkip,mkbq,mkjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )


def reconstruct_t2_alpha_beta(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    norb: int,
    nocc_a: int,
    nocc_b: int,
) -> np.ndarray:
    n_vecs = diag_coulomb_mats.shape[0]
    expanded_diag_coulomb_mats = np.zeros((n_vecs, 4, 2 * norb, 2 * norb))
    expanded_orbital_rotations = np.zeros(
        (n_vecs, 4, 2 * norb, 2 * norb), dtype=complex
    )
    for m, k in itertools.product(range(n_vecs), range(4)):
        (mat_aa, mat_ab, mat_bb) = diag_coulomb_mats[m, k]
        expanded_diag_coulomb_mats[m, k] = np.block(
            [[mat_aa, mat_ab], [mat_ab.T, mat_bb]]
        )
        orbital_rotation_a, orbital_rotation_b = orbital_rotations[m, k]
        expanded_orbital_rotations[m, k] = scipy.linalg.block_diag(
            orbital_rotation_a, orbital_rotation_b
        )
    return (
        2j
        * contract(
            "mkpq,mkap,mkip,mkbq,mkjq->ijab",
            expanded_diag_coulomb_mats,
            expanded_orbital_rotations,
            expanded_orbital_rotations.conj(),
            expanded_orbital_rotations,
            expanded_orbital_rotations.conj(),
        )[:nocc_a, norb : norb + nocc_b, nocc_a:norb, norb + nocc_b :]
    )


@pytest.mark.parametrize("dim", range(6))
def test_modified_cholesky(dim: int):
    """Test modified Cholesky decomposition on a random tensor."""
    rng = np.random.default_rng(4088)
    # construct a random positive definite matrix
    unitary = np.array(random_unitary(dim, seed=rng))
    eigs = rng.uniform(size=dim)
    mat = unitary @ np.diag(eigs) @ unitary.T.conj()
    cholesky_vecs = modified_cholesky(mat)
    reconstructed = np.einsum("ji,ki->jk", cholesky_vecs, cholesky_vecs.conj())
    np.testing.assert_allclose(reconstructed, mat, atol=1e-8)


@pytest.mark.parametrize("dim, cholesky", itertools.product(range(6), [False, True]))
def test_double_factorized_random(dim: int, cholesky: bool):
    """Test double-factorized decomposition on a random tensor."""
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=9825, dtype=float)
    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, cholesky=cholesky
    )
    reconstructed = np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)


@pytest.mark.parametrize("cholesky", [True, False])
def test_double_factorized_tol_max_vecs(cholesky: bool):
    """Test double-factorized decomposition error threshold and max vecs."""
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[["Li", (0, 0, 0)], ["H", (1.6, 0, 0)]],
        basis="sto-3g",
    )
    hartree_fock = scf.RHF(mol)
    hartree_fock.kernel()
    norb = hartree_fock.mol.nao_nr()
    mc = mcscf.CASCI(hartree_fock, norb, mol.nelec)
    two_body_tensor = ao2mo.restore(1, mc.get_h2cas(), mc.ncas)

    # test max_vecs
    max_vecs = 20
    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=max_vecs, cholesky=cholesky
    )
    reconstructed = np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
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
        "kpi,kqi,kij,krj,ksj->pqrs",
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
        "kpi,kqi,kij,krj,ksj->pqrs",
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
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=8386, dtype=float)

    _, orbital_rotations = double_factorized(two_body_tensor)
    diag_coulomb_mats_optimal = optimal_diag_coulomb_mats(
        two_body_tensor, orbital_rotations
    )
    reconstructed = np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
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
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=3718, dtype=float)

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=3
    )
    diag_coulomb_mats_optimal = optimal_diag_coulomb_mats(
        two_body_tensor, orbital_rotations
    )
    reconstructed = np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
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
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=8364, dtype=float)

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_mats_optimized, orbital_rotations_optimized = double_factorized(
        two_body_tensor, max_vecs=2, optimize=True
    )
    reconstructed = np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
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
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=2927, dtype=float)

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_indices = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]

    diag_coulomb_mats_optimized, orbital_rotations_optimized = double_factorized(
        two_body_tensor,
        max_vecs=2,
        optimize=True,
        diag_coulomb_indices=diag_coulomb_indices,
    )

    diag_coulomb_mask = np.zeros((3, 3), dtype=bool)
    rows, cols = zip(*diag_coulomb_indices)
    diag_coulomb_mask[rows, cols] = True
    diag_coulomb_mask[cols, rows] = True
    np.testing.assert_allclose(
        diag_coulomb_mats_optimized, diag_coulomb_mats_optimized * diag_coulomb_mask
    )
    reconstructed = np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < error


@pytest.mark.parametrize("norb, nocc", [(4, 2), (5, 2), (5, 3)])
def test_double_factorized_t2_amplitudes_random(norb: int, nocc: int):
    """Test double factorization of random t2 amplitudes."""
    t2 = random_t2_amplitudes(norb, nocc, dtype=float)
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2)
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    np.testing.assert_allclose(reconstructed, t2, atol=1e-8)
    np.testing.assert_allclose(
        diag_coulomb_mats[:, 0], -diag_coulomb_mats[:, 1], atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[:, 0], orbital_rotations[:, 1].conj(), atol=1e-8
    )


def test_double_factorized_t2_tol_max_vecs():
    """Test double-factorized decomposition error threshold and max vecs."""
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[["Li", (0, 0, 0)], ["H", (1.6, 0, 0)]],
        basis="sto-3g",
    )
    hartree_fock = scf.RHF(mol)
    hartree_fock.kernel()

    ccsd = cc.CCSD(hartree_fock)
    _, _, t2 = ccsd.kernel()
    nocc, _, _, _ = t2.shape

    # test max_vecs
    max_vecs = 8
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(
        t2,
        max_vecs=max_vecs,
    )
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    assert len(orbital_rotations) == max_vecs
    np.testing.assert_allclose(reconstructed, t2, atol=1e-5)

    # test error threshold
    tol = 1e-3
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2, tol=tol)
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    assert len(orbital_rotations) <= 7
    np.testing.assert_allclose(reconstructed, t2, atol=tol)

    # test error threshold and max vecs
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(
        t2, tol=tol, max_vecs=max_vecs
    )
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    assert len(orbital_rotations) <= 7
    np.testing.assert_allclose(reconstructed, t2, atol=tol)


def test_double_factorized_t2_alpha_beta_random():
    """Test double factorization of opposite-spin t2 amplitudes with random tensor."""
    rng = np.random.default_rng()
    shape = (3, 6, 7, 4)
    t2ab = rng.standard_normal(shape)
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab
    )
    nocc_a, nocc_b, nvrt_a, _ = t2ab.shape
    norb = nocc_a + nvrt_a
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    np.testing.assert_allclose(reconstructed, t2ab, atol=1e-8)

    np.testing.assert_allclose(
        diag_coulomb_mats[:, 0], -diag_coulomb_mats[:, 1], atol=1e-8
    )
    np.testing.assert_allclose(
        diag_coulomb_mats[:, 0], -diag_coulomb_mats[:, 2], atol=1e-8
    )
    np.testing.assert_allclose(
        diag_coulomb_mats[:, 0], diag_coulomb_mats[:, 3], atol=1e-8
    )

    np.testing.assert_allclose(
        orbital_rotations[:, 0, 0], orbital_rotations[:, 1, 0], atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[:, 0, 0], orbital_rotations[:, 2, 0].conj(), atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[:, 0, 0], orbital_rotations[:, 3, 0].conj(), atol=1e-8
    )
    # TODO add the rest of the relations


def test_double_factorized_t2_alpha_beta_tol_max_vecs():
    """Test double-factorized decomposition alpha-beta error threshold and max vecs."""
    mol = gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["O", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    hartree_fock = scf.ROHF(mol).run()

    ccsd = cc.CCSD(hartree_fock).run()
    _, t2ab, _ = ccsd.t2
    nocc_a, nocc_b, nvrt_a, _ = t2ab.shape
    norb = nocc_a + nvrt_a

    # test max_vecs
    max_vecs = 25
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, max_vecs=max_vecs
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    assert len(diag_coulomb_mats) == max_vecs
    np.testing.assert_allclose(reconstructed, t2ab, atol=1e-4)

    # test error threshold
    tol = 1e-3
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, tol=tol
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    assert len(diag_coulomb_mats) <= 23
    np.testing.assert_allclose(reconstructed, t2ab, atol=tol)

    # test error threshold and max vecs
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, tol=tol, max_vecs=max_vecs
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    assert len(orbital_rotations) <= 23
    np.testing.assert_allclose(reconstructed, t2ab, atol=tol)
