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
import pyscf
import pyscf.cc
import pyscf.mcscf
import pytest
import scipy.linalg
from opt_einsum import contract

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
            "mpq,map,mip,mbq,mjq->ijab",
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
    n_terms = diag_coulomb_mats.shape[0]
    expanded_diag_coulomb_mats = np.zeros((n_terms, 2 * norb, 2 * norb))
    expanded_orbital_rotations = np.zeros((n_terms, 2 * norb, 2 * norb), dtype=complex)
    for m in range(n_terms):
        (mat_aa, mat_ab, mat_bb) = diag_coulomb_mats[m]
        expanded_diag_coulomb_mats[m] = np.block([[mat_aa, mat_ab], [mat_ab.T, mat_bb]])
        orbital_rotation_a, orbital_rotation_b = orbital_rotations[m]
        expanded_orbital_rotations[m] = scipy.linalg.block_diag(
            orbital_rotation_a, orbital_rotation_b
        )
    return (
        2j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
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
    reconstructed = contract("ji,ki->jk", cholesky_vecs, cholesky_vecs.conj())
    np.testing.assert_allclose(reconstructed, mat, atol=1e-8)


@pytest.mark.parametrize("dim, cholesky", itertools.product(range(6), [False, True]))
def test_double_factorized_random(dim: int, cholesky: bool):
    """Test double-factorized decomposition on a random tensor."""
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=9825, dtype=float)
    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, cholesky=cholesky
    )
    reconstructed = contract(
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
    mol = pyscf.gto.Mole()
    mol.build(
        verbose=0,
        atom=[["Li", (0, 0, 0)], ["H", (1.6, 0, 0)]],
        basis="sto-3g",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock.kernel()
    norb = hartree_fock.mol.nao_nr()
    mc = pyscf.mcscf.CASCI(hartree_fock, norb, mol.nelec)
    two_body_tensor = pyscf.ao2mo.restore(1, mc.get_h2cas(), mc.ncas)

    # test max_vecs
    max_vecs = 20
    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=max_vecs, cholesky=cholesky
    )
    reconstructed = contract(
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
    reconstructed = contract(
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
    reconstructed = contract(
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
    reconstructed = contract(
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
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
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


def test_double_factorized_compressed_n2_unconstrained():
    """Test compressed double factorization on N2."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    two_body_tensor = mol_data.hamiltonian.two_body_tensor

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_mats_optimized, orbital_rotations_optimized, result = (
        double_factorized(
            two_body_tensor,
            max_vecs=2,
            optimize=True,
            options=dict(maxiter=100),
            return_optimize_result=True,
        )
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < 0.5 * error
    assert np.isrealobj(orbital_rotations_optimized)
    assert np.isrealobj(diag_coulomb_mats_optimized)
    assert result.nit <= 100
    assert result.nfev <= 120
    assert result.njev <= 120


def test_double_factorized_compressed_n2_constrained():
    """Test constrained compressed double factorization on N2."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 4
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    two_body_tensor = mol_data.hamiltonian.two_body_tensor

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_indices = [(p, p) for p in range(norb)]
    diag_coulomb_indices.extend([(p, p + 1) for p in range(norb - 1)])
    diag_coulomb_indices.extend([(p, p + 2) for p in range(norb - 2)])
    diag_coulomb_mats_optimized, orbital_rotations_optimized, result = (
        double_factorized(
            two_body_tensor,
            max_vecs=4,
            optimize=True,
            options=dict(maxiter=100),
            diag_coulomb_indices=diag_coulomb_indices,
            return_optimize_result=True,
        )
    )

    diag_coulomb_mask = np.zeros((norb, norb), dtype=bool)
    rows, cols = zip(*diag_coulomb_indices)
    diag_coulomb_mask[rows, cols] = True
    diag_coulomb_mask[cols, rows] = True
    np.testing.assert_allclose(
        diag_coulomb_mats_optimized, diag_coulomb_mats_optimized * diag_coulomb_mask
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < 0.5 * error
    assert np.isrealobj(orbital_rotations_optimized)
    assert np.isrealobj(diag_coulomb_mats_optimized)
    assert result.nit <= 100
    assert result.nfev <= 120
    assert result.njev <= 120


def test_double_factorized_compressed_random():
    """Test compressed double factorization on random tensor."""
    dim = 2
    two_body_tensor = ffsim.random.random_two_body_tensor(dim, seed=8364, dtype=float)

    diag_coulomb_mats, orbital_rotations = double_factorized(
        two_body_tensor, max_vecs=2
    )
    diag_coulomb_mats_optimized, orbital_rotations_optimized = double_factorized(
        two_body_tensor, max_vecs=2, optimize=True
    )
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < 0.1 * error


def test_double_factorized_compressed_random_constrained():
    """Test constrained compressed double factorization"""
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
    reconstructed = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )
    reconstructed_optimal = contract(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations_optimized,
        orbital_rotations_optimized,
        diag_coulomb_mats_optimized,
        orbital_rotations_optimized,
        orbital_rotations_optimized,
    )
    error = np.sum((reconstructed - two_body_tensor) ** 2)
    error_optimized = np.sum((reconstructed_optimal - two_body_tensor) ** 2)
    assert error_optimized < 0.3 * error


@pytest.mark.parametrize("norb, nocc", [(4, 2), (5, 2), (5, 3)])
def test_double_factorized_t2_amplitudes_random(norb: int, nocc: int):
    """Test double factorization of random t2 amplitudes."""
    t2 = random_t2_amplitudes(norb, nocc, dtype=float)
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2)
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    np.testing.assert_allclose(reconstructed, t2, atol=1e-8)
    n_reps, _, _ = diag_coulomb_mats.shape
    even_index = list(range(0, n_reps, 2))
    odd_index = list(range(1, n_reps, 2))
    np.testing.assert_allclose(
        diag_coulomb_mats[even_index], -diag_coulomb_mats[odd_index], atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[even_index], orbital_rotations[odd_index].conj(), atol=1e-8
    )


def test_double_factorized_t2_tol_max_terms():
    """Test double-factorized decomposition error threshold and max terms."""
    mol = pyscf.gto.Mole()
    mol.build(
        verbose=0,
        atom=[["Li", (0, 0, 0)], ["H", (1.6, 0, 0)]],
        basis="sto-3g",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock.kernel()

    ccsd = pyscf.cc.CCSD(hartree_fock)
    _, _, t2 = ccsd.kernel()
    nocc, _, _, _ = t2.shape

    # test max_vecs
    max_terms = 16
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(
        t2,
        max_terms=max_terms,
    )
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    assert len(orbital_rotations) == max_terms
    np.testing.assert_allclose(reconstructed, t2, atol=1e-5)

    # test error threshold
    tol = 1e-3
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2, tol=tol)
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    assert len(orbital_rotations) <= 14
    np.testing.assert_allclose(reconstructed, t2, atol=tol)

    # test error threshold and max vecs
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(
        t2, tol=tol, max_terms=max_terms
    )
    reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
    assert len(orbital_rotations) <= 14
    np.testing.assert_allclose(reconstructed, t2, atol=tol)


def test_double_factorized_t2_optimize_max_terms_n2_small():
    """Test compressed double factorization for smaller N2."""
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )

    # Define active space
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular data and Hamiltonian
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb, _ = mol_data.norb, mol_data.nelec

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    nocc, _, _, _ = ccsd.t2.shape

    # Perform compressed factorization
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    diag_coulomb_indices = pairs_aa + pairs_ab
    max_terms = 1
    diag_coulomb_mats_optimized, orbital_rotations_optimized, result = (
        double_factorized_t2(
            ccsd.t2,
            optimize=True,
            max_terms=max_terms,
            diag_coulomb_indices=diag_coulomb_indices,
            method="L-BFGS-B",
            options=dict(maxiter=25),
            multi_stage_start=8,
            multi_stage_step=4,
            return_optimize_result=True,
        )
    )
    reconstructed_optimized = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats_optimized,
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error_optimized = np.sum(np.abs(reconstructed_optimized - ccsd.t2) ** 2)

    # Perform uncompressed factorization
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(
        ccsd.t2, max_terms=max_terms
    )
    reconstructed = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error = np.sum(np.abs(reconstructed - ccsd.t2) ** 2)

    # Check results
    assert error_optimized < 0.5 * error
    assert diag_coulomb_mats_optimized.shape == (max_terms, norb, norb)
    assert orbital_rotations_optimized.shape == (max_terms, norb, norb)
    assert result.nit <= 25
    assert result.nfev <= 35
    assert result.njev <= 35


def test_double_factorized_t2_optimize_max_terms_n2_large():
    """Test compressed double factorization for larger N2."""
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="6-31g",
        symmetry="Dooh",
    )

    # Define active space
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular data and Hamiltonian
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb, _ = mol_data.norb, mol_data.nelec

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    nocc, _, _, _ = ccsd.t2.shape

    # Perform compressed factorization
    max_terms = 2
    diag_coulomb_mats_optimized, orbital_rotations_optimized = double_factorized_t2(
        ccsd.t2,
        max_terms=max_terms,
        optimize=True,
        method="L-BFGS-B",
        options=dict(maxiter=150),
        regularization=1e-4,
    )
    optimized_diag_coulomb_norm = np.sum(np.abs(diag_coulomb_mats_optimized) ** 2)
    reconstructed_optimized = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats_optimized,
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error_optimized = np.sum(np.abs(reconstructed_optimized - ccsd.t2) ** 2)

    # Perform uncompressed factorization
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(
        ccsd.t2, max_terms=max_terms
    )
    init_diag_coulomb_norm = np.sum(np.abs(diag_coulomb_mats) ** 2)
    reconstructed = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error = np.sum(np.abs(reconstructed - ccsd.t2) ** 2)

    # Check results
    assert error_optimized < 0.5 * error
    np.testing.assert_allclose(
        optimized_diag_coulomb_norm, init_diag_coulomb_norm, atol=2
    )
    assert diag_coulomb_mats_optimized.shape == (max_terms, norb, norb)
    assert orbital_rotations_optimized.shape == (max_terms, norb, norb)


def test_double_factorized_t2_optimize_max_terms_random():
    """Test compressed double factorization with random t2"""
    norb = 4
    nocc = 2
    t2 = ffsim.random.random_t2_amplitudes(norb=norb, nocc=nocc, seed=8856, dtype=float)

    # Perform compressed factorization
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    diag_coulomb_indices = pairs_aa + pairs_ab
    max_terms = 2
    diag_coulomb_mats_optimized, orbital_rotations_optimized = double_factorized_t2(
        t2,
        max_terms=max_terms,
        optimize=True,
        diag_coulomb_indices=diag_coulomb_indices,
        method="L-BFGS-B",
        options=dict(maxiter=25),
        multi_stage_start=3,
        return_optimize_result=False,
    )
    reconstructed_optimized = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats_optimized,
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
            orbital_rotations_optimized,
            orbital_rotations_optimized.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error_optimized = np.sum(np.abs(reconstructed_optimized - t2) ** 2)

    # Perform uncompressed factorization
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2, max_terms=max_terms)
    reconstructed = (
        1j
        * contract(
            "mpq,map,mip,mbq,mjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            orbital_rotations.conj(),
            orbital_rotations,
            orbital_rotations.conj(),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error = np.sum(np.abs(reconstructed - t2) ** 2)

    # Check results
    assert diag_coulomb_mats_optimized.shape == (max_terms, norb, norb)
    assert orbital_rotations_optimized.shape == (max_terms, norb, norb)
    assert error_optimized < 0.5 * error


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

    n_reps = len(diag_coulomb_mats)
    index_0 = list(range(0, n_reps, 4))
    index_1 = list(range(1, n_reps, 4))
    index_2 = list(range(2, n_reps, 4))
    index_3 = list(range(3, n_reps, 4))

    np.testing.assert_allclose(
        diag_coulomb_mats[index_0], -diag_coulomb_mats[index_1], atol=1e-8
    )
    np.testing.assert_allclose(
        diag_coulomb_mats[index_0], -diag_coulomb_mats[index_2], atol=1e-8
    )
    np.testing.assert_allclose(
        diag_coulomb_mats[index_0], diag_coulomb_mats[index_3], atol=1e-8
    )

    np.testing.assert_allclose(
        orbital_rotations[index_0, 0], orbital_rotations[index_1, 0], atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[index_0, 0], orbital_rotations[index_2, 0].conj(), atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[index_0, 0], orbital_rotations[index_3, 0].conj(), atol=1e-8
    )
    # TODO add the rest of the relations


def test_double_factorized_t2_alpha_beta_tol_max_vecs():
    """Test double-factorized decomposition alpha-beta error threshold and max terms."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["O", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    hartree_fock = pyscf.scf.ROHF(mol).run()

    ccsd = pyscf.cc.CCSD(hartree_fock).run()
    _, t2ab, _ = ccsd.t2
    nocc_a, nocc_b, nvrt_a, _ = t2ab.shape
    norb = nocc_a + nvrt_a

    # test max_vecs
    max_terms = 100
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, max_terms=max_terms
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    assert len(diag_coulomb_mats) == max_terms
    np.testing.assert_allclose(reconstructed, t2ab, atol=1e-4)

    # test error threshold
    tol = 1e-3
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, tol=tol
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    assert len(diag_coulomb_mats) <= 92
    np.testing.assert_allclose(reconstructed, t2ab, atol=tol)

    # test error threshold and max vecs
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2_alpha_beta(
        t2ab, tol=tol, max_terms=max_terms
    )
    reconstructed = reconstruct_t2_alpha_beta(
        diag_coulomb_mats, orbital_rotations, norb=norb, nocc_a=nocc_a, nocc_b=nocc_b
    )
    assert len(orbital_rotations) <= 92
    np.testing.assert_allclose(reconstructed, t2ab, atol=tol)
