# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test orbital optimization."""

import numpy as np
import pyscf
import pyscf.ci
from opt_einsum import contract

import ffsim


def test_optimize_orbitals():
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
    mol_hamiltonian = mol_data.hamiltonian

    # Run CISD
    cisd = pyscf.ci.CISD(scf, frozen=n_frozen).run()
    cisd_energy = cisd.e_tot
    np.testing.assert_allclose(cisd_energy, -109.0277962741491)

    # Get RDMs
    rdm1 = cisd.make_rdm1()[n_frozen:, n_frozen:]
    rdm2 = cisd.make_rdm2()[n_frozen:, n_frozen:, n_frozen:, n_frozen:]

    # Optimize orbitals
    orbital_rotation = ffsim.optimize_orbitals(
        rdm1, rdm2, mol_hamiltonian.one_body_tensor, mol_hamiltonian.two_body_tensor
    )

    # Compute energy
    one_body_tensor_rotated = contract(
        "ab,Aa,Bb->AB",
        mol_hamiltonian.one_body_tensor,
        orbital_rotation,
        orbital_rotation.conj(),
        optimize="greedy",
    )
    two_body_tensor_rotated = contract(
        "abcd,Aa,Bb,Cc,Dd->ABCD",
        mol_hamiltonian.two_body_tensor,
        orbital_rotation,
        orbital_rotation.conj(),
        orbital_rotation,
        orbital_rotation.conj(),
        optimize="greedy",
    )
    energy = np.einsum("ab,ab->", one_body_tensor_rotated, rdm1) + 0.5 * np.einsum(
        "abcd,abcd->", two_body_tensor_rotated, rdm2
    )

    # Check results
    np.testing.assert_allclose(energy + mol_data.core_energy, -109.02783860818124)
