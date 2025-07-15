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
from opt_einsum import contract

import ffsim
from ffsim.linalg.double_factorized_decomposition_compressed import (
    double_factorized_t2_compress,
)


def test_double_factorized_compressed():
    """Test compressed double factorization"""
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
        # basis="6-31g",
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
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    n_reps = 2
    diag_coulomb_mats, orbital_rotations = double_factorized_t2_compress(
        ccsd.t2,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab),
        method="L-BFGS-B",
        options={"maxiter": 100},
        multi_stage_optimization=True,
        begin_reps=10,
        step=2,
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
            # optimize="greedy"
        )[:nocc, :nocc, nocc:, nocc:]
    )
    error = np.sum((reconstructed - ccsd.t2) ** 2)
    assert diag_coulomb_mats.shape == (n_reps, norb, norb)
    assert orbital_rotations.shape == (n_reps, norb, norb)
    assert error < 0.001
