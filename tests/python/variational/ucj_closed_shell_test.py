# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for closed-shell unitary cluster Jastrow ansatz."""

import itertools

import numpy as np
import pyscf
import pyscf.cc
import pytest

import ffsim


def test_n_params():
    for norb, n_reps, with_final_orbital_rotation in itertools.product(
        [1, 2, 3], [1, 2, 3], [False, True]
    ):
        operator = ffsim.random.random_ucj_operator_closed_shell(
            norb, n_reps=n_reps, with_final_orbital_rotation=with_final_orbital_rotation
        )
        actual = ffsim.UCJOperatorClosedShell.n_params(
            norb, n_reps, with_final_orbital_rotation=with_final_orbital_rotation
        )
        expected = len(operator.to_parameters())
        assert actual == expected

        alpha_alpha_indices = list(
            itertools.combinations_with_replacement(range(norb), 2)
        )[:norb]
        alpha_beta_indices = list(
            itertools.combinations_with_replacement(range(norb), 2)
        )[:norb]

        actual = ffsim.UCJOperatorClosedShell.n_params(
            norb,
            n_reps,
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        expected = len(
            operator.to_parameters(
                alpha_alpha_indices=alpha_alpha_indices,
                alpha_beta_indices=alpha_beta_indices,
            )
        )
        assert actual == expected

        with pytest.raises(ValueError, match="triangular"):
            actual = ffsim.UCJOperatorClosedShell.n_params(
                norb,
                n_reps,
                alpha_alpha_indices=[(1, 0)],
            )
        with pytest.raises(ValueError, match="triangular"):
            actual = ffsim.UCJOperatorClosedShell.n_params(
                norb,
                n_reps,
                alpha_alpha_indices=alpha_alpha_indices,
                alpha_beta_indices=[(1, 0)],
            )


def test_parameters_roundtrip():
    rng = np.random.default_rng()
    norb = 5
    n_reps = 2

    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_ucj_operator_closed_shell(
            norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=rng,
        )
        roundtripped = ffsim.UCJOperatorClosedShell.from_parameters(
            operator.to_parameters(),
            norb=norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        np.testing.assert_allclose(
            roundtripped.diag_coulomb_mats, operator.diag_coulomb_mats
        )
        np.testing.assert_allclose(
            roundtripped.orbital_rotations, operator.orbital_rotations
        )
        if with_final_orbital_rotation:
            np.testing.assert_allclose(
                roundtripped.final_orbital_rotation, operator.final_orbital_rotation
            )


def test_t_amplitudes_energy():
    # Build an H2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["H", (0, 0, 1.8)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # Construct UCJ operator
    n_reps = 2
    operator = ffsim.UCJOperatorClosedShell.from_t_amplitudes(ccsd.t2, n_reps=n_reps)

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
    np.testing.assert_allclose(energy, -0.96962461)


def test_t_amplitudes_restrict_indices():
    # Build an H2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["H", (0, 0, 1.8)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb

    # Construct UCJ operator
    n_reps = 2
    alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
    alpha_beta_indices = [(p, p) for p in range(norb)]

    operator = ffsim.UCJOperatorClosedShell.from_t_amplitudes(
        ccsd.t2,
        n_reps=n_reps,
        alpha_alpha_indices=alpha_alpha_indices,
        alpha_beta_indices=alpha_beta_indices,
    )
    other_operator = ffsim.UCJOperatorClosedShell.from_parameters(
        operator.to_parameters(
            alpha_alpha_indices=alpha_alpha_indices,
            alpha_beta_indices=alpha_beta_indices,
        ),
        norb=norb,
        n_reps=n_reps,
        alpha_alpha_indices=alpha_alpha_indices,
        alpha_beta_indices=alpha_beta_indices,
    )

    assert ffsim.approx_eq(operator, other_operator, rtol=1e-12)
