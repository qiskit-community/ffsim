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

import pathlib

import numpy as np
import pyscf
import pyscf.mcscf
import pyscf.tools
import pytest
import scipy.sparse.linalg

import ffsim


def test_linear_operator():
    """Test linear operator."""
    # Construct water molecule
    radius_1 = 0.958  # position for the first H atom
    radius_2 = 0.958  # position for the second H atom
    thetas_in_deg = 104.478  # bond angles.

    h1_x = radius_1
    h2_x = radius_2 * np.cos(np.pi / 180 * thetas_in_deg)
    h2_y = radius_2 * np.sin(np.pi / 180 * thetas_in_deg)

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[
            ["O", (0, 0, 0)],
            ["H", (h1_x, 0, 0)],
            ["H", (h2_x, h2_y, 0)],
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

    # Compute FCI energy using PySCF
    norb = len(active_space)
    n_electrons = int(np.sum(hartree_fock.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    cas = pyscf.mcscf.CASCI(hartree_fock, ncas=norb, nelecas=nelec)
    mo = cas.sort_mo(active_space, base=0)
    energy_fci = cas.kernel(mo)[0]

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_scf(hartree_fock, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # compute FCI energy from molecular Hamiltonian
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    eigs, _ = scipy.sparse.linalg.eigsh(hamiltonian, k=1, which="SA")
    eig = eigs[0]

    # Check that they match
    np.testing.assert_allclose(eig, energy_fci)


def test_diag():
    """Test computing diagonal."""
    rng = np.random.default_rng(2222)
    norb = 5
    nelec = (3, 2)
    # TODO remove dtype=float once complex is supported
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng, dtype=float)
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    hamiltonian_dense = linop @ np.eye(ffsim.dim(norb, nelec))
    np.testing.assert_allclose(
        ffsim.diag(hamiltonian, norb=norb, nelec=nelec), np.diag(hamiltonian_dense)
    )


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (4, (1, 2)),
        (4, (0, 2)),
        (4, (0, 0)),
    ],
)
def test_fermion_operator(norb: int, nelec: tuple[int, int]):
    """Test FermionOperator."""
    rng = np.random.default_rng()

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    # TODO remove dtype=float after adding support for complex
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    constant = rng.standard_normal()
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant=constant
    )
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)

    op = ffsim.fermion_operator(mol_hamiltonian)
    linop = ffsim.linear_operator(op, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)


def test_rotated():
    """Test rotating orbitals."""
    norb = 5
    nelec = (3, 2)

    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    # TODO remove dtype=float after adding support for complex
    two_body_tensor = ffsim.random.random_two_body_tensor(norb, seed=rng, dtype=float)
    constant = rng.standard_normal()
    mol_hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, two_body_tensor, constant=constant
    )

    # generate a random orbital rotation
    orbital_rotation = ffsim.random.random_orthogonal(norb, seed=rng)

    # rotate the Hamiltonian
    mol_hamiltonian_rotated = mol_hamiltonian.rotated(orbital_rotation)

    # convert the original and rotated Hamiltonians to linear operators
    linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)
    linop_rotated = ffsim.linear_operator(mol_hamiltonian_rotated, norb, nelec)

    # generate a random statevector
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)

    # rotate the statevector
    rotated_vec = ffsim.apply_orbital_rotation(vec, orbital_rotation, norb, nelec)

    # test definition
    actual = linop_rotated @ vec
    expected = ffsim.apply_orbital_rotation(vec, orbital_rotation.T.conj(), norb, nelec)
    expected = linop @ expected
    expected = ffsim.apply_orbital_rotation(expected, orbital_rotation, norb, nelec)
    np.testing.assert_allclose(actual, expected)

    # test expectation is preserved
    original_expectation = np.vdot(vec, linop @ vec)
    rotated_expectation = np.vdot(rotated_vec, linop_rotated @ rotated_vec)
    np.testing.assert_allclose(original_expectation, rotated_expectation)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_from_fcidump(tmp_path: pathlib.Path):
    """Test loading from FCIDUMP."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = pyscf.data.elements.chemcore(mol)
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    pyscf.tools.fcidump.from_integrals(
        tmp_path / "test.fcidump",
        h1e=mol_data.one_body_integrals,
        h2e=mol_data.two_body_integrals,
        nuc=mol_data.core_energy,
        nmo=mol_data.norb,
        nelec=mol_data.nelec,
    )
    mol_ham = ffsim.MolecularHamiltonian.from_fcidump(tmp_path / "test.fcidump")
    assert ffsim.approx_eq(mol_ham, mol_data.hamiltonian)
