"""Tests for UCJ energy evaluation by fermionic backpropagation."""

import numpy as np
import pytest
import scipy.optimize

import ffsim

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

@pytest.mark.parametrize(
    "pairs",
    [
        None,
        (None, None),
        ([(0, 0), (1, 1)], [(0, 0), (1, 1)]),
        ([(0, 0), (1, 1)], None),
    ],
)
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

    value_and_grad = ffsim.ucj_energy_and_grad_func_spin_balanced(
        ucj_op, mol_hamiltonian, nelec, interaction_pairs=pairs
    )
    result = scipy.optimize.minimize(
        value_and_grad,
        ucj_op.to_parameters(interaction_pairs=pairs),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5},
    )
    optimized_ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
        result.x,
        norb=norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )

    optimized_energy = ffsim.ucj_energy_spin_balanced(
        optimized_ucj_op, mol_hamiltonian, nelec
    )
    assert optimized_energy < ucj_energy


@pytest.mark.parametrize(
    "pairs",
    [
        None,
        (None, None, None),
        ([(0, 0), (1, 1)], [(0, 0), (1, 1)], [(0, 0), (1, 1)]),
        ([(0, 0), (1, 1)], None, None),
    ],
)
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

    value_and_grad = ffsim.ucj_energy_and_grad_func_spin_unbalanced(
        ucj_op, mol_hamiltonian, nelec, interaction_pairs=pairs
    )
    result = scipy.optimize.minimize(
        value_and_grad,
        ucj_op.to_parameters(interaction_pairs=pairs),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5},
    )
    optimized_ucj_op = ffsim.UCJOpSpinUnbalanced.from_parameters(
        result.x,
        norb=norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )

    optimized_energy = ffsim.ucj_energy_spin_unbalanced(
        optimized_ucj_op, mol_hamiltonian, nelec
    )
    assert optimized_energy < ucj_energy


@pytest.mark.parametrize("pairs", [None, [(0, 0), (1, 1)]])
def test_ucj_energy_spinless(pairs):
    norb = 3
    nelec = 2
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spinless(
        norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )

    ucj_energy = ffsim.ucj_energy_spinless(ucj_op, mol_hamiltonian, nelec)
    sv_energy = statevector_energy(ucj_op, mol_hamiltonian, norb, nelec)

    np.testing.assert_allclose(ucj_energy, sv_energy)

    value_and_grad = ffsim.ucj_energy_and_grad_func_spinless(
        ucj_op, mol_hamiltonian, nelec, interaction_pairs=pairs
    )
    result = scipy.optimize.minimize(
        value_and_grad,
        ucj_op.to_parameters(interaction_pairs=pairs),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5},
    )
    optimized_ucj_op = ffsim.UCJOpSpinless.from_parameters(
        result.x,
        norb=norb,
        n_reps=1,
        interaction_pairs=pairs,
        with_final_orbital_rotation=True,
    )

    optimized_energy = ffsim.ucj_energy_spinless(
        optimized_ucj_op, mol_hamiltonian, nelec
    )
    assert optimized_energy < ucj_energy


def test_ucj_energy_chunk_size_nondivisor_spin_balanced():
    """Test two-body chunking with chunk sizes that do not divide term counts."""
    norb = 4

    nelec = (2, 1)
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=1,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )
    unchunked = ffsim.ucj_energy_spin_balanced(ucj_op, mol_hamiltonian, nelec)
    chunked = ffsim.ucj_energy_spin_balanced(
        ucj_op, mol_hamiltonian, nelec, chunk_size=7
    )
    np.testing.assert_allclose(chunked, unchunked)


def test_ucj_energy_chunk_size_nondivisor_spin_unbalanced():
    """Test two-body chunking with chunk sizes that do not divide term counts."""
    norb = 4

    nelec = (2, 1)
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
        norb,
        n_reps=1,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )
    unchunked = ffsim.ucj_energy_spin_unbalanced(ucj_op, mol_hamiltonian, nelec)
    chunked = ffsim.ucj_energy_spin_unbalanced(
        ucj_op, mol_hamiltonian, nelec, chunk_size=7
    )
    np.testing.assert_allclose(chunked, unchunked)


def test_ucj_energy_chunk_size_nondivisor_spinless():
    """Test two-body chunking with chunk sizes that do not divide term counts."""
    norb = 4

    nelec_spinless = 2
    mol_hamiltonian_spinless = ffsim.random.random_molecular_hamiltonian_spinless(
        norb, seed=RNG
    )
    ucj_op_spinless = ffsim.random.random_ucj_op_spinless(
        norb,
        n_reps=1,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )
    unchunked = ffsim.ucj_energy_spinless(
        ucj_op_spinless, mol_hamiltonian_spinless, nelec_spinless
    )
    chunked = ffsim.ucj_energy_spinless(
        ucj_op_spinless,
        mol_hamiltonian_spinless,
        nelec_spinless,
        chunk_size=7,
    )
    np.testing.assert_allclose(chunked, unchunked)


def test_ucj_energy_and_grad_func():
    """Test UCJ energy-and-gradient callables."""
    norb = 3

    nelec = (1, 1)
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)

    ucj_op_balanced = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=1,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )
    value_and_grad = ffsim.ucj_energy_and_grad_func_spin_balanced(
        ucj_op_balanced, mol_hamiltonian, nelec, chunk_size=5
    )
    value, grad = value_and_grad(ucj_op_balanced.to_parameters())
    np.testing.assert_allclose(
        value, ffsim.ucj_energy_spin_balanced(ucj_op_balanced, mol_hamiltonian, nelec)
    )
    assert grad.shape == ucj_op_balanced.to_parameters().shape

    ucj_op_unbalanced = ffsim.random.random_ucj_op_spin_unbalanced(
        norb,
        n_reps=1,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )
    value_and_grad = ffsim.ucj_energy_and_grad_func_spin_unbalanced(
        ucj_op_unbalanced, mol_hamiltonian, nelec
    )
    value, grad = value_and_grad(ucj_op_unbalanced.to_parameters())
    np.testing.assert_allclose(
        value,
        ffsim.ucj_energy_spin_unbalanced(ucj_op_unbalanced, mol_hamiltonian, nelec),
    )
    assert grad.shape == ucj_op_unbalanced.to_parameters().shape

    nelec_spinless = 2
    mol_hamiltonian_spinless = ffsim.random.random_molecular_hamiltonian_spinless(
        norb, seed=RNG
    )
    ucj_op_spinless = ffsim.random.random_ucj_op_spinless(
        norb,
        n_reps=1,
        with_final_orbital_rotation=True,
        diag_coulomb_scale=0.5,
        seed=RNG,
    )
    interaction_pairs = [(0, 0), (0, 1), (1, 2)]
    value_and_grad = ffsim.ucj_energy_and_grad_func_spinless(
        ucj_op_spinless,
        mol_hamiltonian_spinless,
        nelec_spinless,
        interaction_pairs=interaction_pairs,
    )
    params = ucj_op_spinless.to_parameters(interaction_pairs=interaction_pairs)
    value, grad = value_and_grad(params)
    restricted_ucj_op_spinless = ffsim.UCJOpSpinless.from_parameters(
        params,
        norb=norb,
        n_reps=1,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=True,
    )
    np.testing.assert_allclose(
        value,
        ffsim.ucj_energy_spinless(
            restricted_ucj_op_spinless, mol_hamiltonian_spinless, nelec_spinless
        ),
    )
    assert grad.shape == params.shape

    eps = 1e-6
    index = 0
    step = np.zeros_like(params)
    step[index] = eps
    plus_op = ffsim.UCJOpSpinless.from_parameters(
        params + step,
        norb=norb,
        n_reps=1,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=True,
    )
    minus_op = ffsim.UCJOpSpinless.from_parameters(
        params - step,
        norb=norb,
        n_reps=1,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=True,
    )
    finite_diff = (
        ffsim.ucj_energy_spinless(plus_op, mol_hamiltonian_spinless, nelec_spinless)
        - ffsim.ucj_energy_spinless(minus_op, mol_hamiltonian_spinless, nelec_spinless)
    ) / (2 * eps)
    np.testing.assert_allclose(grad[index], finite_diff, rtol=1e-4, atol=1e-5)
