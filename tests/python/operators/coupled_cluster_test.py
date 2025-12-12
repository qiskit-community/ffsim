# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for coupled cluster operators."""

from __future__ import annotations

import numpy as np
import pyscf
import pyscf.cc
import scipy.sparse.linalg

import ffsim

RNG = np.random.default_rng(322844653896162309177764412791243616300)


def ccsd_energy(
    linop: scipy.sparse.linalg.LinearOperator,
    ham_linop: scipy.sparse.linalg.LinearOperator,
    norb: int,
    nelec: tuple[int, int],
) -> float:
    vec = ffsim.hartree_fock_state(norb, nelec)
    bra = scipy.sparse.linalg.expm_multiply(-linop.adjoint(), vec)
    ket = scipy.sparse.linalg.expm_multiply(linop, vec)
    energy = np.vdot(bra, ham_linop @ ket).real
    np.testing.assert_allclose(energy.imag, 0)
    return energy.real


def test_coupled_cluster_singles_and_doubles_restricted():
    bond_distance = 2.4
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("H", (0, 0, i * bond_distance)) for i in range(4)],
        basis="sto-6g",
        symmetry="c1",
        verbose=0,
    )
    scf = pyscf.scf.RHF(mol)
    # Prevent SCF convergence so that the Fock operator is not diagonal and we can
    # test the sign of the singles operator
    scf.max_cycle = 2
    scf.kernel()
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb
    nelec = mol_data.nelec
    ham_linop = ffsim.linear_operator(mol_data.hamiltonian, norb=norb, nelec=nelec)
    ccsd = pyscf.cc.RCCSD(scf)
    ccsd.kernel()

    t1_energy = ccsd.energy(ccsd.t1, np.zeros_like(ccsd.t2), ccsd.ao2mo(ccsd.mo_coeff))
    t2_energy = ccsd.energy(np.zeros_like(ccsd.t1), ccsd.t2, ccsd.ao2mo(ccsd.mo_coeff))
    np.testing.assert_allclose(t1_energy + t2_energy, ccsd.e_corr)

    # Test singles energy
    cc_singles = ffsim.singles_excitations_restricted(ccsd.t1)
    cc_singles_linop = ffsim.linear_operator(cc_singles, norb=norb, nelec=nelec)
    energy_ferm = ccsd_energy(cc_singles_linop, ham_linop, norb=norb, nelec=nelec)
    nocc, _ = ccsd.t1.shape
    one_body_tensor = np.zeros((norb, norb))
    one_body_tensor[:nocc, nocc:] = -ccsd.t1
    one_body_linop = ffsim.contract.one_body_linop(
        one_body_tensor, norb=norb, nelec=nelec
    )
    energy_contract = ccsd_energy(one_body_linop, ham_linop, norb=norb, nelec=nelec)
    np.testing.assert_allclose(energy_ferm, energy_contract)
    np.testing.assert_allclose(energy_ferm, t1_energy + scf.e_tot)

    # Test doubles energy
    # Add a constant to test rmatvec
    cc_doubles = ffsim.doubles_excitations_restricted(ccsd.t2) + ffsim.FermionOperator(
        {(): 1.0}
    )
    cc_doubles_linop = ffsim.linear_operator(cc_doubles, norb=norb, nelec=nelec)
    energy_ferm = ccsd_energy(cc_doubles_linop, ham_linop, norb=norb, nelec=nelec)
    nocc, _ = ccsd.t1.shape
    two_body_tensor = np.zeros((norb, norb, norb, norb))
    two_body_tensor[nocc:, :nocc, nocc:, :nocc] = ccsd.t2.transpose(2, 0, 3, 1)
    two_body_linop = ffsim.contract.two_body_linop(
        two_body_tensor, norb=norb, nelec=nelec, constant=1.0
    )
    energy_contract = ccsd_energy(two_body_linop, ham_linop, norb=norb, nelec=nelec)
    np.testing.assert_allclose(energy_ferm, energy_contract)
    np.testing.assert_allclose(energy_ferm, t2_energy + scf.e_tot)

    # Test action on a random state vector
    # The one-body linear operator from the contract method doesn't pass this test
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    result_ferm = cc_doubles_linop @ vec
    result_contract = two_body_linop @ vec
    np.testing.assert_allclose(result_ferm, result_contract)


def test_ccsd_generator_restricted():
    bond_distance = 2.4
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("H", (0, 0, i * bond_distance)) for i in range(4)],
        basis="sto-6g",
        symmetry="c1",
        verbose=0,
    )
    scf = pyscf.scf.RHF(mol)
    # Prevent SCF convergence so that the Fock operator is not diagonal and we can
    # test the sign of the singles operator
    scf.max_cycle = 2
    scf.kernel()
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb
    nelec = mol_data.nelec
    ham_linop = ffsim.linear_operator(mol_data.hamiltonian, norb=norb, nelec=nelec)
    ccsd = pyscf.cc.RCCSD(scf)
    ccsd.kernel()

    ccsd_gen = ffsim.ccsd_generator_restricted(t1=ccsd.t1, t2=ccsd.t2)
    ccsd_gen_linop = ffsim.linear_operator(ccsd_gen, norb=norb, nelec=nelec)
    energy = ccsd_energy(ccsd_gen_linop, ham_linop, norb=norb, nelec=nelec)
    np.testing.assert_allclose(energy, ccsd.e_tot)


def test_uccsd_generator_restricted_real():
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)

    uccsd_op = ffsim.random.random_uccsd_op_restricted_real(norb, nocc, seed=RNG)
    uccsd_gen = ffsim.uccsd_generator_restricted(t1=uccsd_op.t1, t2=uccsd_op.t2)
    uccsd_gen_linop = ffsim.linear_operator(uccsd_gen, norb=norb, nelec=nelec)

    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    result_ferm = scipy.sparse.linalg.expm_multiply(uccsd_gen_linop, vec, traceA=0.0)
    result_contract = ffsim.apply_unitary(vec, uccsd_op, norb=norb, nelec=nelec)
    np.testing.assert_allclose(np.linalg.norm(result_ferm), 1.0)
    np.testing.assert_allclose(np.linalg.norm(result_contract), 1.0)
    np.testing.assert_allclose(result_ferm, result_contract)


def test_uccsd_generator_restricted_complex():
    norb = 4
    nocc = 2
    nelec = (nocc, nocc)

    uccsd_op = ffsim.random.random_uccsd_op_restricted(norb, nocc, seed=RNG)
    uccsd_gen = ffsim.uccsd_generator_restricted(t1=uccsd_op.t1, t2=uccsd_op.t2)
    uccsd_gen_linop = ffsim.linear_operator(uccsd_gen, norb=norb, nelec=nelec)

    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    result_ferm = scipy.sparse.linalg.expm_multiply(uccsd_gen_linop, vec, traceA=0.0)
    result_contract = ffsim.apply_unitary(vec, uccsd_op, norb=norb, nelec=nelec)
    np.testing.assert_allclose(np.linalg.norm(result_ferm), 1.0)
    np.testing.assert_allclose(np.linalg.norm(result_contract), 1.0)
    np.testing.assert_allclose(result_ferm, result_contract)
