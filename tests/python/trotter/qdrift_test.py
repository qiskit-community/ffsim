# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qDRIFT simulation."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg
from pyscf import ao2mo, gto, mcscf, scf

import ffsim
from ffsim.trotter.qdrift import (
    one_body_square_decomposition,
    spectral_norm_diag_coulomb,
    spectral_norm_one_body_tensor,
    variance_diag_coulomb,
    variance_one_body_tensor,
)


def expectation(
    operator: scipy.sparse.linalg.LinearOperator, state: np.ndarray
) -> complex:
    """Expectation value of operator with state."""
    return np.vdot(state, operator @ state)


def variance(
    operator: scipy.sparse.linalg.LinearOperator, state: np.ndarray
) -> complex:
    """Variance of operator with state."""
    return expectation(operator @ operator, state) - expectation(operator, state) ** 2


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (5, (2, 2)),
        (5, (2, 3)),
        (5, (1, 3)),
        (4, (3, 2)),
    ],
)
def test_spectral_norm_one_body_tensor(norb: int, nelec: tuple[int, int]):
    """Test spectral norm of one-body operator."""
    one_body_tensor = ffsim.random.random_hermitian(norb, seed=8034)
    one_body_linop = ffsim.contract.one_body_linop(
        one_body_tensor, norb=norb, nelec=nelec
    )
    actual = spectral_norm_one_body_tensor(one_body_tensor, nelec=nelec)
    singular_vals = scipy.sparse.linalg.svds(
        one_body_linop, k=1, which="LM", return_singular_vectors=False
    )
    np.testing.assert_allclose(actual, singular_vals[0])


@pytest.mark.parametrize(
    "norb, nelec, rank, z_representation",
    [
        (4, (2, 2), 1, False),
        (5, (1, 3), 1, False),
        (4, (2, 2), 1, True),
        (4, (2, 2), 2, False),
        (5, (1, 3), 3, False),
        (4, (2, 2), 4, True),
    ],
)
def test_spectral_norm_diag_coulomb(
    norb: int, nelec: tuple[int, int], rank: int, z_representation: bool
):
    """Test spectral norm of diagonal Coulomb operator."""
    rng = np.random.default_rng(5745)
    # TODO increasing the number of repetitions to 20 breaks the z-rep test
    for _ in range(5):
        diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
            norb, rank=rank, seed=rng
        )
        two_body_linop = ffsim.contract.diag_coulomb_linop(
            diag_coulomb_mat, norb=norb, nelec=nelec, z_representation=z_representation
        )
        actual = spectral_norm_diag_coulomb(
            diag_coulomb_mat, nelec=nelec, z_representation=z_representation
        )
        singular_vals = scipy.sparse.linalg.svds(
            two_body_linop, k=1, which="LM", return_singular_vectors=False
        )
        if rank == 1:
            np.testing.assert_allclose(actual, singular_vals[0])
        else:
            assert actual >= singular_vals[0] - 1e-12


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (5, (1, 2)),
        (5, (1, 3)),
        (5, (2, 2)),
        (4, (2, 2)),
        (4, (3, 2)),
    ],
)
def test_one_body_squared_decomposition(norb: int, nelec: tuple[int, int]):
    """Test one-body squared decomposition."""
    rng = np.random.default_rng()
    dim = ffsim.dim(norb, nelec)

    for _ in range(10):
        diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
        orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

        one_body_tensors = one_body_square_decomposition(
            diag_coulomb_mat, orbital_rotation
        )
        zero = scipy.sparse.linalg.LinearOperator(
            shape=(dim, dim), matvec=lambda x: np.zeros_like(x)
        )
        actual = sum(
            [
                ffsim.contract.one_body_linop(tensor, norb=norb, nelec=nelec) ** 2
                for tensor in one_body_tensors
            ],
            start=zero,
        )
        expected = ffsim.contract.diag_coulomb_linop(
            diag_coulomb_mat, norb=norb, nelec=nelec, orbital_rotation=orbital_rotation
        )

        vec = ffsim.random.random_state_vector(dim, seed=rng)
        np.testing.assert_allclose(actual @ vec, expected @ vec)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (5, (1, 2)),
        (5, (1, 3)),
        (5, (2, 2)),
        (4, (3, 2)),
    ],
)
def test_variance_one_body_tensor(norb: int, nelec: tuple[int, int]):
    """Test variance of one-body tensor."""
    n_alpha, n_beta = nelec
    rng = np.random.default_rng()

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    one_body_linop = ffsim.contract.one_body_linop(
        one_body_tensor, norb=norb, nelec=nelec
    )

    # generate a random Slater determinant
    vecs = ffsim.random.random_unitary(norb, seed=rng)
    occupied_orbitals_a = vecs[:, :n_alpha]
    occupied_orbitals_b = vecs[:, :n_beta]
    one_rdm_a = occupied_orbitals_a.conj() @ occupied_orbitals_a.T
    one_rdm_b = occupied_orbitals_b.conj() @ occupied_orbitals_b.T
    one_rdm = scipy.linalg.block_diag(one_rdm_a, one_rdm_b)

    # get the full statevector
    state = ffsim.slater_determinant(norb, (range(n_alpha), range(n_beta)))
    state = ffsim.apply_orbital_rotation(state, vecs, norb=norb, nelec=nelec)

    expected = variance(one_body_linop, state)
    actual = variance_one_body_tensor(
        one_rdm, scipy.linalg.block_diag(one_body_tensor, one_body_tensor)
    )
    np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize(
    "norb, nelec, z_representation",
    [
        (4, (2, 2), False),
        (5, (1, 3), False),
    ],
)
def test_variance_diag_coulomb(
    norb: int, nelec: tuple[int, int], z_representation: bool
):
    """Test variance of two-body tensor."""
    n_alpha, n_beta = nelec
    rng = np.random.default_rng()

    diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

    # generate a random Slater determinant
    vecs = ffsim.random.random_unitary(norb, seed=rng)
    occupied_orbitals_a = vecs[:, :n_alpha]
    occupied_orbitals_b = vecs[:, :n_beta]
    one_rdm_a = occupied_orbitals_a.conj() @ occupied_orbitals_a.T
    one_rdm_b = occupied_orbitals_b.conj() @ occupied_orbitals_b.T
    one_rdm = scipy.linalg.block_diag(one_rdm_a, one_rdm_b)

    # get the full statevector
    state = ffsim.slater_determinant(norb, (range(n_alpha), range(n_beta)))
    state = ffsim.apply_orbital_rotation(state, vecs, norb=norb, nelec=nelec)

    linop = ffsim.contract.diag_coulomb_linop(
        diag_coulomb_mat,
        norb=norb,
        nelec=nelec,
        orbital_rotation=orbital_rotation,
    )
    expected = variance(linop, state)
    actual = variance_diag_coulomb(
        one_rdm,
        diag_coulomb_mat,
        orbital_rotation=orbital_rotation,
        z_representation=z_representation,
    )
    np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize(
    "length, bond_distance, basis, time, n_steps, symmetric, probabilities, "
    "z_representation, target_fidelity",
    [
        (4, 1.0, "sto-3g", 1.0, 200, False, "optimal", False, 0.99),
        (4, 1.0, "sto-3g", 1.0, 100, True, "optimal", False, 0.99),
        (4, 1.0, "sto-3g", 1.0, 100, False, "optimal", True, 0.99),
        (4, 1.0, "sto-3g", 1.0, 50, True, "optimal", True, 0.99),
        (4, 1.0, "sto-3g", 1.0, 50, True, "norm", True, 0.99),
    ],
)
def test_simulate_qdrift_double_factorized_h_chain(
    length: int,
    bond_distance: float,
    basis: str,
    time: float,
    n_steps: int,
    symmetric: bool,
    probabilities: str | np.ndarray,
    z_representation: bool,
    target_fidelity: float,
):
    rng = np.random.default_rng(1733)
    mol = gto.Mole()
    mol.build(
        verbose=0,
        atom=[["H", (i * bond_distance, 0, 0)] for i in range(length)],
        basis=basis,
    )
    hartree_fock = scf.RHF(mol)
    hartree_fock.kernel()
    norb = hartree_fock.mol.nao_nr()
    mc = mcscf.CASCI(hartree_fock, norb, mol.nelec)
    one_body_tensor, _ = mc.get_h1cas()
    two_body_tensor = ao2mo.restore(1, mc.get_h2cas(), mc.ncas)
    norb, _ = one_body_tensor.shape
    nelec = mol.nelec
    mol_hamiltonian = ffsim.MolecularHamiltonian(one_body_tensor, two_body_tensor)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)

    # perform double factorization
    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian,
        z_representation=z_representation,
    )

    # generate initial state
    n_alpha, n_beta = nelec
    occupied_orbitals = (range(n_alpha), range(n_beta))
    initial_state = ffsim.slater_determinant(norb, occupied_orbitals)
    original_state = initial_state.copy()
    one_rdm = scipy.linalg.block_diag(
        *ffsim.slater_determinant_rdms(norb, occupied_orbitals)
    )

    # compute exact state
    exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * time * hamiltonian,
        initial_state,
        traceA=-1j * time * ffsim.trace(mol_hamiltonian, norb=norb, nelec=nelec),
    )

    # make sure time is not too small
    assert abs(np.vdot(exact_state, initial_state)) < 0.98

    # simulate
    final_state = ffsim.simulate_qdrift_double_factorized(
        initial_state,
        df_hamiltonian,
        time,
        norb=norb,
        nelec=nelec,
        n_steps=n_steps,
        symmetric=symmetric,
        probabilities=probabilities,
        one_rdm=one_rdm,
        seed=rng,
    )

    # check that initial state was not modified
    np.testing.assert_allclose(initial_state, original_state)

    # check agreement
    np.testing.assert_allclose(np.linalg.norm(final_state), 1.0)
    fidelity = np.abs(np.vdot(final_state, exact_state))
    assert fidelity >= target_fidelity

    # simulate
    final_states = ffsim.simulate_qdrift_double_factorized(
        initial_state,
        df_hamiltonian,
        time,
        norb=norb,
        nelec=nelec,
        n_steps=n_steps,
        symmetric=symmetric,
        probabilities=probabilities,
        one_rdm=one_rdm,
        n_samples=2,
        seed=rng,
    )

    # check agreement
    for final_state in final_states:
        np.testing.assert_allclose(np.linalg.norm(final_state), 1.0)
        fidelity = np.abs(np.vdot(final_state, exact_state))
        assert fidelity >= target_fidelity


@pytest.mark.parametrize(
    "norb, nelec, time, n_steps, probabilities, optimize, z_representation, "
    "target_fidelity",
    [
        (3, (1, 1), 0.05, 100, "optimal", False, False, 0.99),
        (3, (1, 1), 0.05, 100, "norm", True, False, 0.99),
        (4, (2, 2), 0.05, 200, "optimal", False, True, 0.99),
    ],
)
def test_simulate_qdrift_double_factorized_random(
    norb: int,
    nelec: tuple[int, int],
    time: float,
    n_steps: int,
    probabilities: str | np.ndarray,
    optimize: bool,
    z_representation: bool,
    target_fidelity: float,
):
    rng = np.random.default_rng(2030)
    # generate random Hamiltonian
    # TODO test with complex one-body tensor fails due to the following issue
    # https://github.com/qiskit-community/ffsim/issues/14
    one_body_tensor = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    two_body_tensor = ffsim.random.random_two_body_tensor(
        norb, rank=norb, seed=rng, dtype=float
    )
    mol_hamiltonian = ffsim.MolecularHamiltonian(one_body_tensor, two_body_tensor)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)

    # perform double factorization
    df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
        mol_hamiltonian,
        optimize=optimize,
        z_representation=z_representation,
    )

    # generate initial state
    n_alpha, n_beta = nelec
    occupied_orbitals = (range(n_alpha), range(n_beta))
    initial_state = ffsim.slater_determinant(norb, occupied_orbitals)
    original_state = initial_state.copy()
    one_rdm = scipy.linalg.block_diag(
        *ffsim.slater_determinant_rdms(norb, occupied_orbitals)
    )

    # compute exact state
    exact_state = scipy.sparse.linalg.expm_multiply(
        -1j * time * hamiltonian,
        initial_state,
        traceA=-1j * time * ffsim.trace(mol_hamiltonian, norb=norb, nelec=nelec),
    )

    # make sure time is not too small
    assert abs(np.vdot(exact_state, initial_state)) < 0.98

    # simulate
    final_state = ffsim.simulate_qdrift_double_factorized(
        initial_state,
        df_hamiltonian,
        time,
        norb=norb,
        nelec=nelec,
        n_steps=n_steps,
        symmetric=True,
        probabilities=probabilities,
        one_rdm=one_rdm,
        seed=rng,
    )

    # check that initial state was not modified
    np.testing.assert_allclose(initial_state, original_state)

    # check agreement
    np.testing.assert_allclose(np.linalg.norm(final_state), 1.0)
    fidelity = np.abs(np.vdot(final_state, exact_state))
    assert fidelity >= target_fidelity
