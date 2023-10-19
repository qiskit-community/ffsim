# (C) Copyright IBM 2023.
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
import scipy.linalg
from pyscf import cc

import ffsim


def _exponentiate_t1(t1: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    generator = np.zeros((norb, norb), dtype=complex)
    generator[:nocc, nocc:] = t1
    generator[nocc:, :nocc] = -t1.T
    return scipy.linalg.expm(generator)


def test_parameters_roundtrip():
    norb = 2
    n_reps = 2
    diag_coulomb_mats_alpha_alpha = np.array(
        [ffsim.random.random_real_symmetric_matrix(norb) for _ in range(n_reps)]
    )
    diag_coulomb_mats_alpha_beta = np.array(
        [ffsim.random.random_real_symmetric_matrix(norb) for _ in range(n_reps)]
    )
    orbital_rotations = np.array(
        [ffsim.random.random_unitary(norb) for _ in range(n_reps)]
    )
    final_orbital_rotation = ffsim.random.random_unitary(norb)
    operator = ffsim.UCJOperator(
        diag_coulomb_mats_alpha_alpha=diag_coulomb_mats_alpha_alpha,
        diag_coulomb_mats_alpha_beta=diag_coulomb_mats_alpha_beta,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )
    roundtripped = ffsim.UCJOperator.from_parameters(
        operator.to_parameters(),
        norb=norb,
        n_reps=n_reps,
        with_final_orbital_rotation=True,
    )
    np.testing.assert_allclose(
        roundtripped.diag_coulomb_mats_alpha_alpha,
        operator.diag_coulomb_mats_alpha_alpha,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        roundtripped.diag_coulomb_mats_alpha_beta,
        operator.diag_coulomb_mats_alpha_beta,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        roundtripped.orbital_rotations,
        operator.orbital_rotations,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        roundtripped.final_orbital_rotation,
        operator.final_orbital_rotation,
        atol=1e-12,
    )


def test_t_amplitudes_roundtrip():
    norb = 5
    nocc = 3

    rng = np.random.default_rng()

    t2 = ffsim.random.random_t2_amplitudes(norb, nocc)
    t1 = rng.standard_normal((nocc, norb - nocc))

    operator = ffsim.UCJOperator.from_t_amplitudes(t2, t1_amplitudes=t1)
    t2_roundtripped, t1_roundtripped = operator.to_t_amplitudes(nocc=nocc)

    np.testing.assert_allclose(
        t2_roundtripped,
        t2,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _exponentiate_t1(t1_roundtripped, norb=norb, nocc=nocc),
        _exponentiate_t1(t1, norb=norb, nocc=nocc),
        atol=1e-12,
    )


def test_t_amplitudes():
    # Build an H2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["H", (0, 0, 1.8)]],
        basis="sto-6g",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock.kernel()

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_hartree_fock(hartree_fock)
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # Get CCSD t2 amplitudes for initializing the ansatz
    ccsd = cc.CCSD(hartree_fock)
    _, _, t2 = ccsd.kernel()

    # Construct UCJ operator
    n_reps = 2
    operator = ffsim.UCJOperator.from_t_amplitudes(t2, n_reps=n_reps)

    # Construct the Hartree-Fock state to use as the reference state
    n_alpha, n_beta = nelec
    reference_state = ffsim.slater_determinant(
        norb=norb, occupied_orbitals=(range(n_alpha), range(n_beta))
    )

    # Apply the operator to the reference state
    ansatz_state = ffsim.apply_unitary(
        reference_state, operator, norb=norb, nelec=nelec
    )

    # Compute the energy ⟨ψ|H|ψ⟩ of the ansatz state
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))
    np.testing.assert_allclose(energy, -0.96962461, atol=1e-8)
