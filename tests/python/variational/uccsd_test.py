# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for UCCSD ansatz."""

import dataclasses

import numpy as np
import pyscf
import pyscf.cc

import ffsim

RNG = np.random.default_rng(241296678209900701723124712348955121818)


def test_uccsd_real_norb():
    """Test norb property."""
    norb = 5
    nocc = 3
    operator = ffsim.random.random_uccsd_op_restricted_real(norb, nocc, seed=RNG)
    assert operator.norb == norb


def test_uccsd_real_n_params():
    """Test computing number of parameters."""
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_restricted_real(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        actual = ffsim.UCCSDOpRestrictedReal.n_params(
            norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
        )
        expected = len(operator.to_parameters())
        assert actual == expected


def test_uccsd_real_parameters_roundtrip():
    """Test parameters roundtrip."""
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_restricted_real(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = ffsim.UCCSDOpRestrictedReal.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nocc=nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(roundtripped, operator)


def test_uccsd_real_approx_eq():
    """Test approximate equality."""
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_restricted_real(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = ffsim.UCCSDOpRestrictedReal.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nocc=nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(operator, roundtripped)
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, t1=2 * operator.t1)
        )
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, t2=2 * operator.t2)
        )


def test_uccsd_real_apply_unitary():
    """Test unitary."""
    norb = 5
    nocc = 3
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, (nocc, nocc)), seed=RNG)
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_restricted_real(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        result = ffsim.apply_unitary(vec, operator, norb=norb, nelec=(nocc, nocc))
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)


def test_uccsd_real_energy():
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
    mol_hamiltonian = mol_data.hamiltonian

    # Construct UCCSD operator
    operator = ffsim.UCCSDOpRestrictedReal(t1=ccsd.t1, t2=ccsd.t2)

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
    np.testing.assert_allclose(energy, -108.594284)


def test_uccsd_complex_norb():
    """Test norb property."""
    norb = 5
    nocc = 3
    operator = ffsim.random.random_uccsd_op_restricted(norb, nocc, seed=RNG)
    assert operator.norb == norb


def test_uccsd_complex_n_params():
    """Test computing number of parameters."""
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_restricted(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        actual = ffsim.UCCSDOpRestricted.n_params(
            norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
        )
        expected = len(operator.to_parameters())
        assert actual == expected


def test_uccsd_complex_parameters_roundtrip():
    """Test parameters roundtrip."""
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_restricted(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = ffsim.UCCSDOpRestricted.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nocc=nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(roundtripped, operator)


def test_uccsd_complex_approx_eq():
    """Test approximate equality."""
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_restricted(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = ffsim.UCCSDOpRestricted.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nocc=nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(operator, roundtripped)
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, t1=2 * operator.t1)
        )
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, t2=2 * operator.t2)
        )


def test_uccsd_complex_apply_unitary():
    """Test unitary."""
    norb = 5
    nocc = 3
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, (nocc, nocc)), seed=RNG)
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_restricted(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        result = ffsim.apply_unitary(vec, operator, norb=norb, nelec=(nocc, nocc))
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)


def test_uccsd_complex_energy():
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
    mol_hamiltonian = mol_data.hamiltonian

    # Construct UCCSD operator
    operator = ffsim.UCCSDOpRestricted(t1=ccsd.t1, t2=ccsd.t2)

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
    np.testing.assert_allclose(energy, -108.594284)


def test_uccsd_unrestricted_real_norb():
    """Test norb property."""
    norb = 5
    nelec = (3, 2)
    operator = ffsim.random.random_uccsd_op_unrestricted_real(norb, nelec, seed=RNG)
    assert operator.norb == norb


def test_uccsd_unrestricted_real_n_params():
    """Test computing number of parameters."""
    norb = 5
    nelec = (3, 2)
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_unrestricted_real(
            norb,
            nelec,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        actual = ffsim.UCCSDOpUnrestrictedReal.n_params(
            norb, nelec, with_final_orbital_rotation=with_final_orbital_rotation
        )
        expected = len(operator.to_parameters())
        assert actual == expected


def test_uccsd_unrestricted_real_parameters_roundtrip():
    """Test parameters roundtrip."""
    norb = 5
    nelec = (3, 2)
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_unrestricted_real(
            norb,
            nelec,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = ffsim.UCCSDOpUnrestrictedReal.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nelec=nelec,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(roundtripped, operator)


def test_uccsd_unrestricted_real_approx_eq():
    """Test approximate equality."""
    norb = 5
    nelec = (3, 2)
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_unrestricted_real(
            norb,
            nelec,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = ffsim.UCCSDOpUnrestrictedReal.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nelec=nelec,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(operator, roundtripped)
        assert not ffsim.approx_eq(
            operator,
            dataclasses.replace(operator, t1=tuple(2 * t1 for t1 in operator.t1)),
        )
        assert not ffsim.approx_eq(
            operator,
            dataclasses.replace(operator, t2=tuple(2 * t2 for t2 in operator.t2)),
        )


def test_uccsd_unrestricted_real_apply_unitary():
    """Test unitary."""
    norb = 5
    nelec = (3, 2)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_op_unrestricted_real(
            norb,
            nelec,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        result = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)


def test_uccsd_unrestricted_real_energy():
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("H", (0, 0, i)) for i in range(5)],
        basis="sto-6g",
        spin=1,
        symmetry="Dooh",
        verbose=0,
    )
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()

    # Get molecular data and molecular Hamiltonian
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb
    nelec = mol_data.nelec
    assert norb == 5
    assert nelec == (3, 2)
    mol_hamiltonian = mol_data.hamiltonian

    # Construct UCCSD operator
    operator = ffsim.UCCSDOpUnrestrictedReal(t1=ccsd.t1, t2=ccsd.t2)

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
    np.testing.assert_allclose(energy, -2.672171)
