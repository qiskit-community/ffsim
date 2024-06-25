# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Givens rotation ansatz."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pyscf
import pytest
import scipy.linalg

import ffsim
import ffsim.linalg.givens


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_parameters_roundtrip():
    """Test converting to and back from parameters gives consistent results."""
    norb = 5
    n_reps = 2
    rng = np.random.default_rng()

    def ncycles(iterable, n):
        "Returns the sequence elements n times"
        return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))

    pairs = itertools.combinations(range(norb), 2)
    interaction_pairs = list(ncycles(pairs, n_reps))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))

    operator = ffsim.GivensAnsatzOperator(
        norb=norb, interaction_pairs=interaction_pairs, thetas=thetas
    )
    assert len(operator.to_parameters()) == n_reps * math.comb(norb, 2)
    roundtripped = ffsim.GivensAnsatzOperator.from_parameters(
        operator.to_parameters(), norb=norb, interaction_pairs=interaction_pairs
    )

    np.testing.assert_allclose(roundtripped.thetas, operator.thetas)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_orbital_rotation(norb: int, nelec: tuple[int, int]):
    n_reps = 2
    rng = np.random.default_rng()

    def ncycles(iterable, n):
        "Returns the sequence elements n times"
        return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))

    pairs = itertools.combinations(range(norb), 2)
    interaction_pairs = list(ncycles(pairs, n_reps))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))

    operator = ffsim.GivensAnsatzOperator(
        norb=norb, interaction_pairs=interaction_pairs, thetas=thetas
    )

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)
    actual = ffsim.apply_orbital_rotation(
        vec, operator.to_orbital_rotation(), norb=norb, nelec=nelec
    )
    expected = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)

    np.testing.assert_allclose(actual, expected)


def test_givens_parameters_roundtrip():
    """Test converting to and back from parameters gives consistent results."""
    norb = 5
    rng = np.random.default_rng()

    interaction_pairs = list(itertools.combinations(range(norb), 2))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phis = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phase_angles = rng.uniform(-np.pi, np.pi, size=norb)

    operator = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=thetas,
        phis=None,
        phase_angles=None,
    )
    assert (
        operator.n_params(
            norb, interaction_pairs, with_phis=False, with_phase_angles=False
        )
        == len(operator.to_parameters())
        == norb * (norb - 1) // 2
    )
    roundtripped = ffsim.GivensAnsatzOp.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=interaction_pairs,
        with_phis=False,
        with_phase_angles=False,
    )
    assert ffsim.approx_eq(roundtripped, operator)

    operator = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=thetas,
        phis=phis,
        phase_angles=None,
    )
    assert (
        operator.n_params(norb, interaction_pairs, with_phase_angles=False)
        == len(operator.to_parameters())
        == norb * (norb - 1)
    )
    roundtripped = ffsim.GivensAnsatzOp.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=interaction_pairs,
        with_phase_angles=False,
    )
    assert ffsim.approx_eq(roundtripped, operator)

    operator = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=thetas,
        phis=None,
        phase_angles=phase_angles,
    )
    assert (
        operator.n_params(norb, interaction_pairs, with_phis=False)
        == len(operator.to_parameters())
        == norb * (norb - 1) // 2 + norb
    )
    roundtripped = ffsim.GivensAnsatzOp.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=interaction_pairs,
        with_phis=False,
    )
    assert ffsim.approx_eq(roundtripped, operator)

    operator = ffsim.GivensAnsatzOp(
        norb=norb,
        interaction_pairs=interaction_pairs,
        thetas=thetas,
        phis=phis,
        phase_angles=phase_angles,
    )
    assert (
        operator.n_params(norb, interaction_pairs)
        == len(operator.to_parameters())
        == norb**2
    )
    roundtripped = ffsim.GivensAnsatzOp.from_parameters(
        operator.to_parameters(),
        norb=norb,
        interaction_pairs=interaction_pairs,
    )
    assert ffsim.approx_eq(roundtripped, operator)


@pytest.mark.parametrize("norb", range(5))
def test_givens_orbital_rotation_roundtrip(norb: int):
    """Test round-tripping orbital rotation."""
    rng = np.random.default_rng()
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    roundtripped = operator.to_orbital_rotation()
    np.testing.assert_allclose(roundtripped, orbital_rotation)


def test_givens_orbital_rotation_t1_roundtrip():
    """Test round-tripping orbital rotation from t1 amplitudes."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()

    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec
    assert norb == 8
    assert nelec == (5, 5)

    nocc, _, _, _ = ccsd.t2.shape
    orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
    orbital_rotation_generator[:nocc, nocc:] = ccsd.t1
    orbital_rotation_generator[nocc:, :nocc] = -ccsd.t1.T
    orbital_rotation = scipy.linalg.expm(orbital_rotation_generator)
    assert ffsim.linalg.is_unitary(orbital_rotation)

    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    roundtripped = operator.to_orbital_rotation()
    np.testing.assert_allclose(roundtripped, orbital_rotation, atol=1e-12)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_givens_orbital_rotation_unitary(norb: int, nelec: tuple[int, int]):
    """Test initialization from orbital rotation."""
    rng = np.random.default_rng()
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_orbital_rotation(
        vec, orbital_rotation, norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(actual, expected)


def test_givens_orbital_rotation_t_amplitudes():
    """Test initialization from orbital rotation gives fully parametrized ansatz."""
    rng = np.random.default_rng(50783)

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.0)]],
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

    nocc, _, _, _ = ccsd.t2.shape
    orbital_rotation_generator = np.zeros((norb, norb), dtype=complex)
    orbital_rotation_generator[:nocc, nocc:] = ccsd.t1
    orbital_rotation_generator[nocc:, :nocc] = -ccsd.t1.T
    orbital_rotation = scipy.linalg.expm(orbital_rotation_generator)
    assert ffsim.linalg.is_unitary(orbital_rotation)

    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    assert len(operator.interaction_pairs) == norb * (norb - 1) // 2
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_orbital_rotation(
        vec, orbital_rotation, norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(actual, expected)


def test_givens_orbital_rotation_fully_parameterized():
    """Test initialization from orbital rotation gives fully parametrized ansatz."""
    norb = 8
    nelec = 4
    rng = np.random.default_rng()
    orbital_rotation = scipy.linalg.block_diag(
        ffsim.random.random_unitary(norb // 2, seed=rng),
        ffsim.random.random_unitary(norb // 2, seed=rng),
    )
    operator = ffsim.GivensAnsatzOp.from_orbital_rotation(orbital_rotation)
    assert len(operator.interaction_pairs) == norb * (norb - 1) // 2
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    expected = ffsim.apply_orbital_rotation(
        vec, orbital_rotation, norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(actual, expected)


def test_givens_incorrect_num_params():
    """Test that passing incorrect number of parameters throws an error."""
    norb = 5
    interaction_pairs = [(0, 1), (2, 3)]
    with pytest.raises(ValueError, match="number"):
        _ = ffsim.GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=np.zeros(len(interaction_pairs) + 1),
            phis=np.zeros(len(interaction_pairs)),
            phase_angles=np.zeros(norb),
        )
    with pytest.raises(ValueError, match="number"):
        _ = ffsim.GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=np.zeros(len(interaction_pairs)),
            phis=np.zeros(len(interaction_pairs) + 1),
            phase_angles=np.zeros(norb),
        )
    with pytest.raises(ValueError, match="number"):
        _ = ffsim.GivensAnsatzOp(
            norb=norb,
            interaction_pairs=interaction_pairs,
            thetas=np.zeros(len(interaction_pairs)),
            phis=np.zeros(len(interaction_pairs)),
            phase_angles=np.zeros(norb + 1),
        )
