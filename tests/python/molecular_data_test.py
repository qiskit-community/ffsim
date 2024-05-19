# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
