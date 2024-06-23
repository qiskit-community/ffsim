# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for spin-unbalanced unitary cluster Jastrow ansatz."""

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
        operator = ffsim.random.random_ucj_op_spin_unbalanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=with_final_orbital_rotation
        )
        actual = ffsim.UCJOpSpinUnbalanced.n_params(
            norb, n_reps, with_final_orbital_rotation=with_final_orbital_rotation
        )
        expected = len(operator.to_parameters())
        assert actual == expected

        pairs_aa = list(itertools.combinations_with_replacement(range(norb), 2))[:norb]
        pairs_ab = list(itertools.combinations_with_replacement(range(norb), 2))[:norb]
        pairs_bb = list(itertools.combinations_with_replacement(range(norb), 2))[:norb]

        actual = ffsim.UCJOpSpinUnbalanced.n_params(
            norb,
            n_reps,
            interaction_pairs=(pairs_aa, pairs_ab, pairs_bb),
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        expected = len(
            operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab, pairs_bb))
        )
        assert actual == expected

        with pytest.raises(ValueError, match="triangular"):
            actual = ffsim.UCJOpSpinUnbalanced.n_params(
                norb,
                n_reps,
                interaction_pairs=([(1, 0)], pairs_ab, pairs_bb),
            )
        with pytest.raises(ValueError, match="triangular"):
            actual = ffsim.UCJOpSpinUnbalanced.n_params(
                norb,
                n_reps,
                interaction_pairs=(pairs_aa, pairs_ab, [(1, 0)]),
            )
        with pytest.raises(ValueError, match="Duplicate"):
            actual = ffsim.UCJOpSpinUnbalanced.n_params(
                norb,
                n_reps,
                interaction_pairs=(pairs_aa, [(1, 0), (1, 0)], pairs_bb),
            )


def test_parameters_roundtrip():
    rng = np.random.default_rng()
    norb = 5
    n_reps = 2

    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_ucj_op_spin_unbalanced(
            norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=rng,
        )
        roundtripped = ffsim.UCJOpSpinUnbalanced.from_parameters(
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
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["Be", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    scf = pyscf.scf.ROHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()

    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian
    linop = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)

    n_alpha, n_beta = nelec
    reference_state = ffsim.slater_determinant(
        norb=norb, occupied_orbitals=(range(n_alpha), range(n_beta))
    )

    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, t1=ccsd.t1, n_reps=4
    )
    ansatz_state = ffsim.apply_unitary(
        reference_state, operator, norb=norb, nelec=nelec
    )
    energy = np.real(np.vdot(ansatz_state, linop @ ansatz_state))
    np.testing.assert_allclose(energy, -15.124376)

    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(ccsd.t2, n_reps=(4, 2))
    ansatz_state = ffsim.apply_unitary(
        reference_state, operator, norb=norb, nelec=nelec
    )
    energy = np.real(np.vdot(ansatz_state, linop @ ansatz_state))
    np.testing.assert_allclose(energy, -15.126083)


def test_t_amplitudes_restrict_indices():
    # Build a BeH molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["Be", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    scf = pyscf.scf.ROHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb

    # Construct UCJ operator
    n_reps = 2
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    pairs_bb = [(p, p + 1) for p in range(norb - 1)]

    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)
    )
    other_operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
        operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)),
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab, pairs_bb),
    )

    assert ffsim.approx_eq(operator, other_operator, rtol=1e-12)


def test_validate():
    rng = np.random.default_rng(335)
    n_reps = 3
    norb = 4
    eye = np.eye(norb)
    diag_coulomb_mats = np.stack([np.stack([eye, eye, eye]) for _ in range(n_reps)])
    orbital_rotations = np.stack([np.stack([eye, eye]) for _ in range(n_reps)])

    _ = ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=rng.standard_normal(10),
        orbital_rotations=orbital_rotations,
        validate=False,
    )

    _ = ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=rng.standard_normal((n_reps, 3, norb, norb)),
        orbital_rotations=orbital_rotations,
        atol=10,
    )

    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=rng.standard_normal(10),
            orbital_rotations=orbital_rotations,
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=rng.standard_normal(10),
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=rng.standard_normal(10),
        )
    with pytest.raises(ValueError, match="dimension"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=np.concatenate([orbital_rotations, orbital_rotations]),
        )
    with pytest.raises(ValueError, match="symmetric"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=rng.standard_normal((n_reps, 3, norb, norb)),
            orbital_rotations=orbital_rotations,
        )
    with pytest.raises(ValueError, match="unitary"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=rng.standard_normal((n_reps, 2, norb, norb)),
        )
    with pytest.raises(ValueError, match="unitary"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=rng.standard_normal((2, norb, norb)),
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=np.stack([np.stack([eye, eye]) for _ in range(n_reps)]),
            orbital_rotations=orbital_rotations,
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=np.stack(
                [np.stack([eye, eye, eye]) for _ in range(n_reps)]
            ),
        )
