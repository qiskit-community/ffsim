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
import scipy.linalg

import ffsim

RNG = np.random.default_rng(241127903049053917326682280086371763733)


def test_optimize_orbitals():
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (2.4, 0, 0)]],
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
    norb = mol_data.norb

    # Run CISD
    cisd = pyscf.ci.CISD(scf, frozen=n_frozen).run()
    cisd_energy = cisd.e_tot
    np.testing.assert_allclose(cisd_energy, -108.58532210214092)

    # Get RDMs
    rdm1 = cisd.make_rdm1()[n_frozen:, n_frozen:]
    rdm2 = cisd.make_rdm2()[n_frozen:, n_frozen:, n_frozen:, n_frozen:]
    rdm = ffsim.ReducedDensityMatrix(rdm1, rdm2)

    # Optimize orbitals with real rotations
    orbital_rotation, result = ffsim.optimize_orbitals(
        rdm,
        mol_hamiltonian,
        return_optimize_result=True,
    )
    # Compute energy
    energy = rdm.rotated(orbital_rotation).expectation(mol_hamiltonian)
    energy_alt = rdm.expectation(mol_hamiltonian.rotated(orbital_rotation.T.conj()))
    # Check results
    np.testing.assert_allclose(energy, result.fun)
    np.testing.assert_allclose(energy, energy_alt)
    np.testing.assert_allclose(energy, -108.58613393502857)
    assert np.isrealobj(orbital_rotation)
    assert len(result.x) == norb * (norb - 1) // 2
    assert result.nit <= 7
    assert result.nfev <= 9
    assert result.njev <= 9

    # Optimize orbitals with complex rotations
    orbital_rotation, result = ffsim.optimize_orbitals(
        rdm,
        mol_hamiltonian,
        initial_orbital_rotation=scipy.linalg.expm(
            1e-4 * ffsim.random.random_antihermitian(norb, seed=RNG)
        ),
        return_optimize_result=True,
    )
    # Compute energy
    energy = rdm.rotated(orbital_rotation).expectation(mol_hamiltonian)
    energy_alt = rdm.expectation(mol_hamiltonian.rotated(orbital_rotation.T.conj()))
    # Check results
    np.testing.assert_allclose(energy, result.fun)
    np.testing.assert_allclose(energy, energy_alt)
    np.testing.assert_allclose(energy, -108.58613393502857)
    assert np.iscomplexobj(orbital_rotation)
    assert len(result.x) == norb**2
    assert result.nit <= 8
    assert result.nfev <= 11
    assert result.njev <= 11
