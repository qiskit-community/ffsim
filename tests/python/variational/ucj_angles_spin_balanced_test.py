# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for spin-balanced UCJ ansatz parameterized by gate rotation angles."""

import itertools
from typing import Any

import numpy as np
import pyscf
import pyscf.cc
import pyscf.ci
import pytest

import ffsim

RNG = np.random.default_rng(301312747948034963541438592504182479355)


def _brickwork(norb: int, n_layers: int):
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)


def test_n_params():
    for norb, n_reps, with_final_orbital_rotation in itertools.product(
        [1, 2, 3], [1, 2, 3], [False, True]
    ):
        triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
        brickwork_indices = list(_brickwork(norb, norb))

        ucj_op = ffsim.random.random_ucj_op_spin_balanced(
            norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            interaction_pairs=(triu_indices, triu_indices),
        )
        operator = ffsim.UCJAnglesOpSpinBalanced.from_ucj_op(ucj_op)
        actual = ffsim.UCJAnglesOpSpinBalanced.n_params(
            norb,
            n_reps,
            num_num_interaction_pairs=(triu_indices, triu_indices),
            givens_interaction_pairs=brickwork_indices,
            with_final_givens_ansatz_op=with_final_orbital_rotation,
        )
        expected = len(operator.to_parameters())
        assert actual == expected

        pairs_aa = triu_indices[:norb]
        pairs_ab = triu_indices[:norb]
        ucj_op = ffsim.random.random_ucj_op_spin_balanced(
            norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            interaction_pairs=(pairs_aa, pairs_ab),
        )
        operator = ffsim.UCJAnglesOpSpinBalanced.from_ucj_op(
            ucj_op, interaction_pairs=(pairs_aa, pairs_ab)
        )
        actual = ffsim.UCJAnglesOpSpinBalanced.n_params(
            norb,
            n_reps,
            num_num_interaction_pairs=(pairs_aa, pairs_ab),
            givens_interaction_pairs=brickwork_indices,
            with_final_givens_ansatz_op=with_final_orbital_rotation,
        )
        expected = len(operator.to_parameters())
        assert actual == expected


def test_parameters_roundtrip():
    norb = 5
    n_reps = 2
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    pairs_aa = triu_indices[:norb]
    pairs_ab = triu_indices[:norb]
    brickwork_indices = list(_brickwork(norb, norb))

    for with_final_orbital_rotation in [False, True]:
        ucj_op = ffsim.random.random_ucj_op_spin_balanced(
            norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            interaction_pairs=(pairs_aa, pairs_ab),
            seed=RNG,
        )
        operator = ffsim.UCJAnglesOpSpinBalanced.from_ucj_op(
            ucj_op, interaction_pairs=(pairs_aa, pairs_ab)
        )
        roundtripped = ffsim.UCJAnglesOpSpinBalanced.from_parameters(
            operator.to_parameters(),
            norb=norb,
            n_reps=n_reps,
            num_num_interaction_pairs=(pairs_aa, pairs_ab),
            givens_interaction_pairs=brickwork_indices,
            with_final_givens_ansatz_op=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(operator, roundtripped)


def test_apply_unitary_consistent_with_ucj_op():
    norb = 8
    nelec = (5, 5)
    n_reps = 3

    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=n_reps, with_final_orbital_rotation=True
    )
    ucj_angles_op = ffsim.UCJAnglesOpSpinBalanced.from_ucj_op(ucj_op)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    actual = ffsim.apply_unitary(vec, ucj_angles_op, norb=norb, nelec=nelec)
    expected = ffsim.apply_unitary(vec, ucj_op, norb=norb, nelec=nelec)

    np.testing.assert_allclose(actual, expected)


def test_from_t_amplitudes_consistent_with_ucj_op():
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, -0.5)], ["N", (0, 0, 0.5)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()

    # Get molecular data and molecular Hamiltonian
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec
    assert norb == 8
    assert nelec == (5, 5)

    # Construct UCJ operator
    n_reps = 2
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        ccsd.t2,
        t1=ccsd.t1,
        n_reps=n_reps,
    )
    ucj_angles_op = ffsim.UCJAnglesOpSpinBalanced.from_t_amplitudes(
        ccsd.t2,
        t1=ccsd.t1,
        n_reps=n_reps,
    )
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    actual = ffsim.apply_unitary(vec, ucj_angles_op, norb=norb, nelec=nelec)
    expected = ffsim.apply_unitary(vec, ucj_op, norb=norb, nelec=nelec)

    np.testing.assert_allclose(actual, expected)


def test_t_amplitudes_energy():
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, -0.5)], ["N", (0, 0, 0.5)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()

    # Get molecular data and molecular Hamiltonian
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec
    assert norb == 8
    assert nelec == (5, 5)
    mol_hamiltonian = mol_data.hamiltonian

    # Construct UCJ operator
    n_reps = 2
    operator = ffsim.UCJAnglesOpSpinBalanced.from_t_amplitudes(
        ccsd.t2, t1=ccsd.t1, n_reps=n_reps
    )

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
    np.testing.assert_allclose(energy, -108.56438)


def test_t_amplitudes_restrict_indices():
    # Build an H2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, -0.9)], ["H", (0, 0, 0.9)]],
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
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]

    operator = ffsim.UCJAnglesOpSpinBalanced.from_t_amplitudes(
        ccsd.t2, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab)
    )
    other_operator = ffsim.UCJAnglesOpSpinBalanced.from_parameters(
        operator.to_parameters(),
        norb=norb,
        n_reps=n_reps,
        num_num_interaction_pairs=(pairs_aa, pairs_ab),
        givens_interaction_pairs=list(_brickwork(norb, norb)),
    )

    assert ffsim.approx_eq(operator, other_operator, rtol=1e-12)


def test_from_ucj_op_parameter_layout_independent_of_zeros():
    norb = 4
    n_reps = 2
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    brickwork_indices = list(_brickwork(norb, norb))
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=n_reps, seed=RNG)
    diag_coulomb_mats = ucj_op.diag_coulomb_mats.copy()
    diag_coulomb_mats[0, 0, 0, 1] = diag_coulomb_mats[0, 0, 1, 0] = 0
    diag_coulomb_mats[1] = 0
    ucj_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=ucj_op.orbital_rotations,
    )
    operator = ffsim.UCJAnglesOpSpinBalanced.from_ucj_op(ucj_op)
    roundtripped = ffsim.UCJAnglesOpSpinBalanced.from_parameters(
        operator.to_parameters(),
        norb=norb,
        n_reps=n_reps,
        num_num_interaction_pairs=(triu_indices, triu_indices),
        givens_interaction_pairs=brickwork_indices,
    )
    assert ffsim.approx_eq(operator, roundtripped)


def test_from_t_amplitudes_padded_n_reps():
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=float)
    n_terms = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2).n_reps
    n_reps = n_terms + 2
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    brickwork_indices = list(_brickwork(norb, norb))

    operator = ffsim.UCJAnglesOpSpinBalanced.from_t_amplitudes(t2, n_reps=n_reps)
    assert operator.n_reps == n_reps
    params = operator.to_parameters()
    assert len(params) == ffsim.UCJAnglesOpSpinBalanced.n_params(
        norb,
        n_reps,
        num_num_interaction_pairs=(triu_indices, triu_indices),
        givens_interaction_pairs=brickwork_indices,
    )
    roundtripped = ffsim.UCJAnglesOpSpinBalanced.from_parameters(
        params,
        norb=norb,
        n_reps=n_reps,
        num_num_interaction_pairs=(triu_indices, triu_indices),
        givens_interaction_pairs=brickwork_indices,
    )
    assert ffsim.approx_eq(operator, roundtripped)

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, n_reps=n_reps)
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_unitary(vec, ucj_op, norb=norb, nelec=nelec)
    np.testing.assert_allclose(actual, expected)


def test_from_t_amplitudes_interaction_pairs_order():
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    n_reps = 2
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=float)
    pairs_aa = [(2, 3), (0, 1), (1, 2)]
    pairs_ab = [(3, 3), (0, 0), (2, 2), (1, 1)]

    operator = ffsim.UCJAnglesOpSpinBalanced.from_t_amplitudes(
        t2, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab)
    )
    roundtripped = ffsim.UCJAnglesOpSpinBalanced.from_parameters(
        operator.to_parameters(),
        norb=norb,
        n_reps=n_reps,
        num_num_interaction_pairs=(pairs_aa, pairs_ab),
        givens_interaction_pairs=list(_brickwork(norb, norb)),
    )
    assert ffsim.approx_eq(operator, roundtripped)

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab)
    )
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_unitary(vec, ucj_op, norb=norb, nelec=nelec)
    np.testing.assert_allclose(actual, expected)


def test_from_t_amplitudes_optimize_consistent_with_ucj_op():
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)
    n_reps = 2
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=float)
    t1 = RNG.standard_normal((nocc, norb - nocc))
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    kwargs: dict[str, Any] = dict(
        t1=t1,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab),
        optimize=True,
        regularization=1e-4,
        multi_stage_start=4,
        multi_stage_step=1,
        options=dict(maxiter=10),
    )

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, **kwargs)
    operator = ffsim.UCJAnglesOpSpinBalanced.from_t_amplitudes(t2, **kwargs)
    assert ffsim.approx_eq(
        operator,
        ffsim.UCJAnglesOpSpinBalanced.from_ucj_op(
            ucj_op, interaction_pairs=(pairs_aa, pairs_ab)
        ),
    )

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_unitary(vec, ucj_op, norb=norb, nelec=nelec)
    np.testing.assert_allclose(actual, expected)


def test_approx_eq_final_givens_ansatz_op_tolerance():
    norb = 4
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=1, with_final_orbital_rotation=True, seed=RNG
    )
    operator = ffsim.UCJAnglesOpSpinBalanced.from_ucj_op(ucj_op)
    params = operator.to_parameters()
    params[-1] += 1e-3
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    other = ffsim.UCJAnglesOpSpinBalanced.from_parameters(
        params,
        norb=norb,
        n_reps=1,
        num_num_interaction_pairs=(triu_indices, triu_indices),
        givens_interaction_pairs=list(_brickwork(norb, norb)),
        with_final_givens_ansatz_op=True,
    )
    assert not ffsim.approx_eq(operator, other)
    assert ffsim.approx_eq(operator, other, atol=1e-2)


@pytest.mark.parametrize("optimize", [False, True])
def test_from_cisd_vec_matches_t_amplitudes(optimize: bool):
    norb = 4
    nocc = 2
    n_reps = 2
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=float)
    t1 = RNG.standard_normal((nocc, norb - nocc))
    c2 = t2 + 0.5 * np.einsum("ia,jb->ijab", t1, t1)
    cisd_vec = pyscf.ci.cisd.amplitudes_to_cisdvec(1.0, t1, c2)
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    kwargs: dict[str, Any] = dict(
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab),
        optimize=optimize,
        options=dict(maxiter=10),
    )

    operator = ffsim.UCJAnglesOpSpinBalanced.from_cisd_vec(
        cisd_vec, norb=norb, nocc=nocc, **kwargs
    )
    expected = ffsim.UCJAnglesOpSpinBalanced.from_t_amplitudes(t2, t1=t1, **kwargs)

    assert ffsim.approx_eq(operator, expected, rtol=1e-12)
