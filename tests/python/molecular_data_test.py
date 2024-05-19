# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import pyscf
import pyscf.data.elements

import ffsim


def test_molecular_data_sym():
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )

    # Define active space
    n_frozen = pyscf.data.elements.chemcore(mol)
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular data and Hamiltonian
    mol_data = ffsim.MolecularData.from_mole(mol, active_space=active_space)

    assert mol_data.orbital_symmetries == [
        "A1g",
        "A1u",
        "E1uy",
        "E1ux",
        "A1g",
        "E1gx",
        "E1gy",
        "A1u",
    ]


def test_molecular_data_no_sym():
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
    )

    # Define active space
    n_frozen = pyscf.data.elements.chemcore(mol)
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular data and Hamiltonian
    mol_data = ffsim.MolecularData.from_mole(mol, active_space=active_space)

    assert mol_data.orbital_symmetries is None


def test_molecular_data_run_methods():
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )

    # Define active space
    n_frozen = pyscf.data.elements.chemcore(mol)
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular data and Hamiltonian
    mol_data = ffsim.MolecularData.from_mole(mol, active_space=active_space)

    # Run calculations
    mol_data.run_mp2()
    mol_data.run_fci()
    mol_data.run_ccsd()

    np.testing.assert_allclose(mol_data.mp2_energy, -108.58852784026)
    np.testing.assert_allclose(mol_data.fci_energy, -108.595987350986)
    np.testing.assert_allclose(mol_data.ccsd_energy, -108.5933309085008)
