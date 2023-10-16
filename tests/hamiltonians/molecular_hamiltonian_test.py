# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for molecular Hamiltonian."""


from __future__ import annotations

import numpy as np
import pyscf
import pyscf.mcscf
import scipy.sparse.linalg

import ffsim


def test_linear_operator():
    """Test linear operator."""
    # Construct water molecule
    radius_1 = 0.958  # position for the first H atom
    radius_2 = 0.958  # position for the second H atom
    thetas_in_deg = 104.478  # bond angles.

    H1_x = radius_1
    H2_x = radius_2 * np.cos(np.pi / 180 * thetas_in_deg)
    H2_y = radius_2 * np.sin(np.pi / 180 * thetas_in_deg)

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[
            ["O", (0, 0, 0)],
            ["H", (H1_x, 0, 0)],
            ["H", (H2_x, H2_y, 0)],
        ],
        basis="sto-6g",
        spin=0,
        charge=0,
        symmetry="c2v",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock.kernel()

    # Define active space
    active_space = [1, 2, 4, 5, 6]

    # Compute FCI energy using pySCF
    norb = len(active_space)
    n_electrons = int(np.sum(hartree_fock.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    cas = pyscf.mcscf.CASCI(hartree_fock, ncas=norb, nelecas=nelec)
    mo = cas.sort_mo(active_space, base=0)
    energy_fci = cas.kernel(mo)[0]

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_hartree_fock(
        hartree_fock, active_space=active_space
    )
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # compute FCI energy from molecular Hamiltonian
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    eigs, _ = scipy.sparse.linalg.eigsh(hamiltonian, k=1, which="SA")
    eig = eigs[0]

    # Check that they match
    np.testing.assert_allclose(eig, energy_fci)
