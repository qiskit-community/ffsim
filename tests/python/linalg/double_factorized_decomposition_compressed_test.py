# (C) Copyright IBM 2025.
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
import pyscf
import pyscf.cc
from opt_einsum import contract

import ffsim
from ffsim.linalg import double_factorized_t2
from ffsim.linalg.double_factorized_decomposition_compressed import (
    double_factorized_t2_compressed,
)


def test_double_factorized_compressed_n2_small():
    """Test compressed double factorization"""
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
    n_reps = 1
    diag_coulomb_mats_optimized, orbital_rotations_optimized, result = (
        double_factorized_t2_compressed(
            ccsd.t2,
            n_reps=n_reps,
            diag_coulomb_indices=diag_coulomb_indices,
            method="L-BFGS-B",
            options=dict(maxiter=25),
            multi_stage_optimization=True,
            begin_reps=8,
            step=4,
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
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(ccsd.t2)
    _, _, norb, _ = orbital_rotations.shape
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
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
    assert diag_coulomb_mats_optimized.shape == (n_reps, norb, norb)
    assert orbital_rotations_optimized.shape == (n_reps, norb, norb)
    assert result.nit <= 25
    assert result.nfev <= 35
    assert result.njev <= 35


def test_double_factorized_compressed_n2_large():
    """Test compressed double factorization"""
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
    n_reps = 2
    diag_coulomb_mats_optimized, orbital_rotations_optimized = (
        double_factorized_t2_compressed(
            ccsd.t2,
            n_reps=n_reps,
            method="L-BFGS-B",
            options=dict(maxiter=150),
            multi_stage_optimization=False,
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
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(ccsd.t2)
    _, _, norb, _ = orbital_rotations.shape
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
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
    assert diag_coulomb_mats_optimized.shape == (n_reps, norb, norb)
    assert orbital_rotations_optimized.shape == (n_reps, norb, norb)


def test_double_factorized_compressed_random():
    """Test compressed double factorization with random t2"""
    norb = 4
    nocc = 2
    t2 = ffsim.random.random_t2_amplitudes(norb=norb, nocc=nocc, seed=8856, dtype=float)

    # Perform compressed factorization
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    diag_coulomb_indices = pairs_aa + pairs_ab
    n_reps = 2
    diag_coulomb_mats_optimized, orbital_rotations_optimized = (
        double_factorized_t2_compressed(
            t2,
            n_reps=n_reps,
            diag_coulomb_indices=diag_coulomb_indices,
            method="L-BFGS-B",
            options=dict(maxiter=25),
            multi_stage_optimization=True,
            begin_reps=5,
            step=4,
            return_optimize_result=False,
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
    error_optimized = np.sum(np.abs(reconstructed_optimized - t2) ** 2)

    # Perform uncompressed factorization
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2)
    _, _, norb, _ = orbital_rotations.shape
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
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
    assert diag_coulomb_mats_optimized.shape == (n_reps, norb, norb)
    assert orbital_rotations_optimized.shape == (n_reps, norb, norb)
    assert error_optimized < 0.5 * error
