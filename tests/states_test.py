# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test states."""

import numpy as np
import pyscf

import ffsim


def test_slater_determinant():
    """Test Slater determinant."""
    norb = 5
    nelec = ffsim.testing.random_nelec(norb)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)
    occ_a, occ_b = occupied_orbitals

    one_body_tensor = ffsim.random.random_hermitian(norb)
    eigs, orbital_rotation = np.linalg.eigh(one_body_tensor)
    eig = sum(eigs[occ_a]) + sum(eigs[occ_b])
    state = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    hamiltonian = ffsim.contract.one_body_linop(one_body_tensor, norb=norb, nelec=nelec)
    np.testing.assert_allclose(hamiltonian @ state, eig * state)


def test_hartree_fock_state():
    """Test Hartree-Fock state."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["H", (0, 0, 1.8)]],
        basis="sto-6g",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock_energy = hartree_fock.kernel()

    mol_data = ffsim.MolecularData.from_hartree_fock(hartree_fock)

    vec = ffsim.hartree_fock_state(mol_data.norb, mol_data.nelec)
    hamiltonian = ffsim.linear_operator(
        mol_data.hamiltonian, norb=mol_data.norb, nelec=mol_data.nelec
    )
    energy = np.vdot(vec, hamiltonian @ vec)

    np.testing.assert_allclose(energy, hartree_fock_energy)


def test_indices_to_strings():
    """Test converting statevector indices to strings."""
    norb = 3
    nelec = (2, 1)

    dim = ffsim.dim(norb, nelec)
    strings = ffsim.indices_to_strings(range(dim), norb, nelec)
    assert strings == [
        "011001",
        "011010",
        "011100",
        "101001",
        "101010",
        "101100",
        "110001",
        "110010",
        "110100",
    ]
