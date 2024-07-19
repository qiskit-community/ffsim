# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import dataclasses
import pathlib

import numpy as np
import pyscf
import pyscf.data.elements

import ffsim


def _assert_mol_data_equal(
    actual_mol_data: ffsim.MolecularData, expected_mol_data: ffsim.MolecularData
):
    for field in dataclasses.fields(actual_mol_data):
        actual = getattr(actual_mol_data, field.name)
        expected = getattr(expected_mol_data, field.name)
        if field.type == "np.ndarray":
            assert isinstance(actual, np.ndarray)
            np.testing.assert_array_equal(actual, expected)
        elif field.type in [
            "np.ndarray | None",
            "np.ndarray | tuple[np.ndarray, np.ndarray] | None",
            "np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | None",
            "tuple[np.ndarray, np.ndarray, np.ndarray] | None",
        ]:
            if actual is not None:
                if isinstance(actual, tuple):
                    for actual_val, expected_val in zip(actual, expected):
                        assert isinstance(actual_val, np.ndarray)
                        assert isinstance(expected_val, np.ndarray)
                        np.testing.assert_array_equal(actual_val, expected_val)
                else:
                    assert isinstance(actual, np.ndarray)
                    np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


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
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)

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
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)

    # Run calculations
    mol_data.run_mp2()
    mol_data.run_ccsd()
    mol_data.run_cisd()
    mol_data.run_sci()
    mol_data.run_fci()

    np.testing.assert_allclose(mol_data.mp2_energy, -108.58852784026)
    np.testing.assert_allclose(mol_data.ccsd_energy, -108.5933309085008)
    np.testing.assert_allclose(mol_data.cisd_energy, -108.5878344909782)
    np.testing.assert_allclose(mol_data.sci_energy, -108.59598682615388)
    np.testing.assert_allclose(mol_data.fci_energy, -108.595987350986)


def test_json_closed_shell(tmp_path: pathlib.Path):
    """Test saving to and loading from JSON for a closed-shell molecule."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("N", (0, 0, 0)), ("N", (1.0, 0, 0))],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = pyscf.data.elements.chemcore(mol)
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    mol_data.run_mp2(store_t2=True)
    mol_data.run_ccsd(store_t1=True, store_t2=True)
    mol_data.run_cisd(store_cisd_vec=True)
    mol_data.run_sci(store_sci_vec=True)
    mol_data.run_fci(store_fci_vec=True)

    for compression in [None, "gzip", "bz2", "lzma"]:
        mol_data.to_json(tmp_path / "test.json", compression=compression)
        loaded_mol_data = ffsim.MolecularData.from_json(
            tmp_path / "test.json", compression=compression
        )
        _assert_mol_data_equal(loaded_mol_data, mol_data)


def test_json_open_shell(tmp_path: pathlib.Path):
    """Test saving to and loading from JSON for an open-shell molecule."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("H", (0, 0, 0)), ("O", (0, 0, 1.1))],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    scf = pyscf.scf.ROHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf)
    mol_data.run_mp2(store_t2=True)
    mol_data.run_ccsd(store_t1=True, store_t2=True)
    mol_data.run_cisd(store_cisd_vec=True)
    mol_data.run_sci(store_sci_vec=True)
    mol_data.run_fci(store_fci_vec=True)

    for compression in [None, "gzip", "bz2", "lzma"]:
        mol_data.to_json(tmp_path / "test.json", compression=compression)
        loaded_mol_data = ffsim.MolecularData.from_json(
            tmp_path / "test.json", compression=compression
        )
        _assert_mol_data_equal(loaded_mol_data, mol_data)


def test_fcidump(tmp_path: pathlib.Path):
    """Test saving to and loading from FCIDUMP."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
        symmetry="Dooh",
        spin=0,
    )
    n_frozen = pyscf.data.elements.chemcore(mol)
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    mol_data.to_fcidump(tmp_path / "test.fcidump")
    loaded_mol_data = ffsim.MolecularData.from_fcidump(tmp_path / "test.fcidump")
    assert loaded_mol_data.norb == mol_data.norb
    assert loaded_mol_data.nelec == mol_data.nelec
    assert loaded_mol_data.spin == mol_data.spin
    assert ffsim.approx_eq(loaded_mol_data.hamiltonian, mol_data.hamiltonian)

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["O", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    scf = pyscf.scf.ROHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    mol_data.to_fcidump(tmp_path / "test.fcidump")
    loaded_mol_data = ffsim.MolecularData.from_fcidump(tmp_path / "test.fcidump")
    assert loaded_mol_data.norb == mol_data.norb
    assert loaded_mol_data.nelec == mol_data.nelec
    assert loaded_mol_data.spin == mol_data.spin
    assert ffsim.approx_eq(loaded_mol_data.hamiltonian, mol_data.hamiltonian)
