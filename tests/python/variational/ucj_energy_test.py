"""Tests for UCJ energy evaluation by fermionic backpropagation."""

import numpy as np
import pyscf

import ffsim
import pytest

RNG = np.random.default_rng(4978)

def statevector_energy(ucj_op, hamiltonian, norb, nelec):
    occupied_orbitals = (
        range(nelec) if isinstance(nelec, int) else (range(nelec[0]), range(nelec[1]))
    )
    reference_state = ffsim.slater_determinant(
        norb=norb, occupied_orbitals=occupied_orbitals
    )
    ansatz_state = ffsim.apply_unitary(reference_state, ucj_op, norb=norb, nelec=nelec)
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    return np.real(np.vdot(ansatz_state, linop @ ansatz_state))

@pytest.mark.parametrize("pairs", [None, (None, None), ([(0, 0), (1, 1)], [(0, 0), (1, 1)]), ([(0, 0), (1, 1)], None)])
def test_ucj_energy_spin_balanced(pairs): 
    norb = 3
    nelec = (1, 1)
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )

    ucj_energy = ffsim.ucj_energy_spin_balanced(ucj_op, mol_hamiltonian, nelec)
    sv_energy = statevector_energy(ucj_op, mol_hamiltonian, norb, nelec)

    np.testing.assert_allclose(ucj_energy, sv_energy)

    optimized_ucj_op = ffsim.optimize_ucj_energy_spin_balanced(
        ucj_op, mol_hamiltonian, nelec, interaction_pairs=pairs, options={"maxiter": 5}
    )

    optimized_energy = ffsim.ucj_energy_spin_balanced(optimized_ucj_op, mol_hamiltonian, nelec)
    assert optimized_energy < ucj_energy

@pytest.mark.parametrize("pairs", [None, (None, None, None), ([(0, 0), (1, 1)], [(0, 0), (1, 1)], [(0, 0), (1, 1)]), ([(0, 0), (1, 1)], None, None)])
def test_ucj_energy_spin_unbalanced(pairs): 
    norb = 3
    nelec = (1, 1)
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
        norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )

    ucj_energy = ffsim.ucj_energy_spin_unbalanced(ucj_op, mol_hamiltonian, nelec)
    sv_energy = statevector_energy(ucj_op, mol_hamiltonian, norb, nelec)

    np.testing.assert_allclose(ucj_energy, sv_energy)

    optimized_ucj_op = ffsim.optimize_ucj_energy_spin_unbalanced(
        ucj_op, mol_hamiltonian, nelec, interaction_pairs=pairs, options={"maxiter": 5}
    )

    optimized_energy = ffsim.ucj_energy_spin_unbalanced(optimized_ucj_op, mol_hamiltonian, nelec)
    assert optimized_energy < ucj_energy

# def test_ucj_energy_spin_balanced_n2():
#     """Compare fermionic backpropagation against statevector simulation."""
#     mol = pyscf.gto.Mole()
#     mol.build(
#         atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.0)]],
#         basis="sto-6g",
#         symmetry="Dooh",
#     )
#     scf = pyscf.scf.RHF(mol).run()

#     n_frozen = 2
#     active_space = range(n_frozen, mol.nao_nr())
#     ccsd = pyscf.cc.CCSD(
#         scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
#     ).run()

#     mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
#     mol_hamiltonian = mol_data.hamiltonian
#     norb = mol_data.norb
#     nelec = mol_data.nelec
#     assert norb == 8
#     assert nelec == (5, 5)

#     pairs_aa = [(p, p) for p in range(norb - 1)]
#     pairs_ab = [(p, p) for p in range(norb - 1)]

#     ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
#         ccsd.t2, t1=ccsd.t1, interaction_pairs=(pairs_aa, pairs_ab), n_reps=1
#     )

#     backprop_energy = ffsim.ucj_energy_spin_balanced(ucj_op, mol_hamiltonian, nelec)
#     statevector_energy = _statevector_energy(ucj_op, mol_hamiltonian, norb, nelec)

#     np.testing.assert_allclose(backprop_energy, statevector_energy)

#     optimized_ucj_op, result = ffsim.optimize_ucj_energy_spin_balanced(
#         ucj_op,
#         mol_hamiltonian,
#         nelec,
#         options={"maxiter": 1},
#         return_optimize_result=True,
#     )
#     optimized_backprop_energy = ffsim.ucj_energy_spin_balanced(
#         optimized_ucj_op, mol_hamiltonian, nelec
#     )
#     optimized_statevector_energy = _statevector_energy(
#         optimized_ucj_op, mol_hamiltonian, norb, nelec
#     )

#     np.testing.assert_allclose(optimized_backprop_energy, result.fun)
#     np.testing.assert_allclose(optimized_backprop_energy, optimized_statevector_energy)


# def test_ucj_energy_spin_unbalanced_n2():
#     """Compare spin-unbalanced backpropagation against statevector simulation."""
#     mol = pyscf.gto.Mole()
#     mol.build(
#         atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.0)]],
#         basis="sto-6g",
#         symmetry="Dooh",
#     )
#     scf = pyscf.scf.RHF(mol).run()

#     n_frozen = 2
#     active_space = range(n_frozen, mol.nao_nr())
#     mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
#     mol_hamiltonian = mol_data.hamiltonian
#     norb = mol_data.norb
#     nelec = mol_data.nelec

#     ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
#         norb,
#         n_reps=1,
#         with_final_orbital_rotation=True,
#         diag_coulomb_scale=0.5,
#         seed=RNG,
#     )

#     backprop_energy = ffsim.ucj_energy_spin_unbalanced(ucj_op, mol_hamiltonian, nelec)
#     statevector_energy = _statevector_energy(ucj_op, mol_hamiltonian, norb, nelec)

#     np.testing.assert_allclose(backprop_energy, statevector_energy)


# def test_ucj_energy_spinless_n2():
#     """Compare spinless backpropagation against statevector simulation."""
#     mol = pyscf.gto.Mole()
#     mol.build(
#         atom=[["N", (0, 0, 0)], ["N", (0, 0, 1.0)]],
#         basis="sto-6g",
#         symmetry="Dooh",
#     )
#     scf = pyscf.scf.RHF(mol).run()

#     n_frozen = 2
#     active_space = range(n_frozen, mol.nao_nr())
#     mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
#     mol_hamiltonian = ffsim.MolecularHamiltonianSpinless(
#         one_body_tensor=mol_data.hamiltonian.one_body_tensor,
#         two_body_tensor=mol_data.hamiltonian.two_body_tensor,
#         constant=mol_data.hamiltonian.constant,
#     )
#     norb = mol_data.norb
#     nelec = mol_data.nelec[0]

#     ucj_op = ffsim.random.random_ucj_op_spinless(
#         norb,
#         n_reps=1,
#         with_final_orbital_rotation=True,
#         diag_coulomb_scale=0.5,
#         seed=RNG,
#     )

#     backprop_energy = ffsim.ucj_energy_spinless(ucj_op, mol_hamiltonian, nelec)
#     statevector_energy = _statevector_energy(ucj_op, mol_hamiltonian, norb, nelec)

#     np.testing.assert_allclose(backprop_energy, statevector_energy)


# def test_optimize_ucj_energy_spin_unbalanced():
#     """Compare spin-unbalanced optimization objective against statevector simulation."""
#     norb = 3
#     nelec = (1, 1)
#     mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
#     ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
#         norb,
#         n_reps=1,
#         with_final_orbital_rotation=True,
#         diag_coulomb_scale=0.5,
#         seed=RNG,
#     )

#     optimized_ucj_op, result = ffsim.optimize_ucj_energy_spin_unbalanced(
#         ucj_op,
#         mol_hamiltonian,
#         nelec,
#         options={"maxiter": 1},
#         return_optimize_result=True,
#     )
#     optimized_backprop_energy = ffsim.ucj_energy_spin_unbalanced(
#         optimized_ucj_op, mol_hamiltonian, nelec
#     )
#     optimized_statevector_energy = _statevector_energy(
#         optimized_ucj_op, mol_hamiltonian, norb, nelec
#     )

#     np.testing.assert_allclose(optimized_backprop_energy, result.fun)
#     np.testing.assert_allclose(optimized_backprop_energy, optimized_statevector_energy)


# def test_optimize_ucj_energy_spinless():
#     """Compare spinless optimization objective against statevector simulation."""
#     norb = 4
#     nelec = 2
#     mol_hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(norb, seed=RNG)
#     ucj_op = ffsim.random.random_ucj_op_spinless(
#         norb,
#         n_reps=1,
#         with_final_orbital_rotation=True,
#         diag_coulomb_scale=0.5,
#         seed=RNG,
#     )

#     optimized_ucj_op, result = ffsim.optimize_ucj_energy_spinless(
#         ucj_op,
#         mol_hamiltonian,
#         nelec,
#         options={"maxiter": 1},
#         return_optimize_result=True,
#     )
#     optimized_backprop_energy = ffsim.ucj_energy_spinless(
#         optimized_ucj_op, mol_hamiltonian, nelec
#     )
#     optimized_statevector_energy = _statevector_energy(
#         optimized_ucj_op, mol_hamiltonian, norb, nelec
#     )

#     np.testing.assert_allclose(optimized_backprop_energy, result.fun)
#     np.testing.assert_allclose(optimized_backprop_energy, optimized_statevector_energy)


# def test_ucj_energy_chunk_size_nondivisor():
#     """Test two-body chunking with chunk sizes that do not divide term counts."""
#     norb = 4

#     nelec = (2, 1)
#     mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
#     ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
#         norb,
#         n_reps=1,
#         with_final_orbital_rotation=True,
#         diag_coulomb_scale=0.5,
#         seed=RNG,
#     )
#     unchunked = ffsim.ucj_energy_spin_unbalanced(ucj_op, mol_hamiltonian, nelec)
#     chunked = ffsim.ucj_energy_spin_unbalanced(
#         ucj_op, mol_hamiltonian, nelec, chunk_size=7
#     )
#     np.testing.assert_allclose(chunked, unchunked)

#     nelec_spinless = 2
#     mol_hamiltonian_spinless = ffsim.random.random_molecular_hamiltonian_spinless(
#         norb, seed=RNG
#     )
#     ucj_op_spinless = ffsim.random.random_ucj_op_spinless(
#         norb,
#         n_reps=1,
#         with_final_orbital_rotation=True,
#         diag_coulomb_scale=0.5,
#         seed=RNG,
#     )
#     unchunked = ffsim.ucj_energy_spinless(
#         ucj_op_spinless, mol_hamiltonian_spinless, nelec_spinless
#     )
#     chunked = ffsim.ucj_energy_spinless(
#         ucj_op_spinless,
#         mol_hamiltonian_spinless,
#         nelec_spinless,
#         chunk_size=7,
#     )
#     np.testing.assert_allclose(chunked, unchunked)


# def test_ucj_energy_and_grad():
#     """Test UCJ energy-and-gradient wrappers."""
#     norb = 3

#     nelec = (1, 1)
#     mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)

#     ucj_op_balanced = ffsim.random.random_ucj_op_spin_balanced(
#         norb,
#         n_reps=1,
#         with_final_orbital_rotation=True,
#         diag_coulomb_scale=0.5,
#         seed=RNG,
#     )
#     value, grad = ffsim.ucj_energy_and_grad_spin_balanced(
#         ucj_op_balanced, mol_hamiltonian, nelec, chunk_size=5
#     )
#     np.testing.assert_allclose(
#         value, ffsim.ucj_energy_spin_balanced(ucj_op_balanced, mol_hamiltonian, nelec)
#     )
#     assert grad.shape == ucj_op_balanced.to_parameters().shape

#     ucj_op_unbalanced = ffsim.random.random_ucj_op_spin_unbalanced(
#         norb,
#         n_reps=1,
#         with_final_orbital_rotation=True,
#         diag_coulomb_scale=0.5,
#         seed=RNG,
#     )
#     value, grad = ffsim.ucj_energy_and_grad_spin_unbalanced(
#         ucj_op_unbalanced, mol_hamiltonian, nelec
#     )
#     np.testing.assert_allclose(
#         value,
#         ffsim.ucj_energy_spin_unbalanced(ucj_op_unbalanced, mol_hamiltonian, nelec),
#     )
#     assert grad.shape == ucj_op_unbalanced.to_parameters().shape

#     nelec_spinless = 2
#     mol_hamiltonian_spinless = ffsim.random.random_molecular_hamiltonian_spinless(
#         norb, seed=RNG
#     )
#     ucj_op_spinless = ffsim.random.random_ucj_op_spinless(
#         norb,
#         n_reps=1,
#         with_final_orbital_rotation=True,
#         diag_coulomb_scale=0.5,
#         seed=RNG,
#     )
#     interaction_pairs = [(0, 0), (0, 1), (1, 2)]
#     value, grad = ffsim.ucj_energy_and_grad_spinless(
#         ucj_op_spinless,
#         mol_hamiltonian_spinless,
#         nelec_spinless,
#         interaction_pairs=interaction_pairs,
#     )
#     params = ucj_op_spinless.to_parameters(interaction_pairs=interaction_pairs)
#     restricted_ucj_op_spinless = ffsim.UCJOpSpinless.from_parameters(
#         params,
#         norb=norb,
#         n_reps=1,
#         interaction_pairs=interaction_pairs,
#         with_final_orbital_rotation=True,
#     )
#     np.testing.assert_allclose(
#         value,
#         ffsim.ucj_energy_spinless(
#             restricted_ucj_op_spinless, mol_hamiltonian_spinless, nelec_spinless
#         ),
#     )
#     assert grad.shape == params.shape

#     eps = 1e-6
#     index = 0
#     step = np.zeros_like(params)
#     step[index] = eps
#     plus_op = ffsim.UCJOpSpinless.from_parameters(
#         params + step,
#         norb=norb,
#         n_reps=1,
#         interaction_pairs=interaction_pairs,
#         with_final_orbital_rotation=True,
#     )
#     minus_op = ffsim.UCJOpSpinless.from_parameters(
#         params - step,
#         norb=norb,
#         n_reps=1,
#         interaction_pairs=interaction_pairs,
#         with_final_orbital_rotation=True,
#     )
#     finite_diff = (
#         ffsim.ucj_energy_spinless(plus_op, mol_hamiltonian_spinless, nelec_spinless)
#         - ffsim.ucj_energy_spinless(minus_op, mol_hamiltonian_spinless, nelec_spinless)
#     ) / (2 * eps)
#     np.testing.assert_allclose(grad[index], finite_diff, rtol=1e-4, atol=1e-5)
