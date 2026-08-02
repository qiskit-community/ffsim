"""Tests for UCJ energy evaluation by fermionic backpropagation."""

import numpy as np
import pyscf

import ffsim


def _statevector_energy(ucj_op, hamiltonian, norb, nelec):
    reference_state = ffsim.slater_determinant(
        norb=norb, occupied_orbitals=(range(nelec[0]), range(nelec[1]))
    )
    ansatz_state = ffsim.apply_unitary(
        reference_state, ucj_op, norb=norb, nelec=nelec
    )
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    return np.real(np.vdot(ansatz_state, linop @ ansatz_state))


def test_ucj_energy_spin_balanced_n2():
    """Compare fermionic backpropagation against statevector simulation."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.0)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    scf = pyscf.scf.RHF(mol).run()

    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()

    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    mol_hamiltonian = mol_data.hamiltonian
    norb = mol_data.norb
    nelec = mol_data.nelec
    assert norb == 8
    assert nelec == (5, 5)

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        ccsd.t2, t1=ccsd.t1, n_reps=1
    )

    backprop_energy = ffsim.ucj_energy_spin_balanced(
        ucj_op, mol_hamiltonian, nelec
    )
    statevector_energy = _statevector_energy(ucj_op, mol_hamiltonian, norb, nelec)

    np.testing.assert_allclose(backprop_energy, statevector_energy)

    optimized_ucj_op, result = ffsim.optimize_ucj_energy_spin_balanced(
        ucj_op,
        mol_hamiltonian,
        nelec,
        options={"maxiter": 1},
        return_optimize_result=True,
    )
    optimized_backprop_energy = ffsim.ucj_energy_spin_balanced(
        optimized_ucj_op, mol_hamiltonian, nelec
    )
    optimized_statevector_energy = _statevector_energy(
        optimized_ucj_op, mol_hamiltonian, norb, nelec
    )

    np.testing.assert_allclose(optimized_backprop_energy, result.fun)
    np.testing.assert_allclose(optimized_backprop_energy, optimized_statevector_energy)
