# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for linear algebra utilities."""

from __future__ import annotations

import itertools

import numpy as np
import pyscf
import pyscf.cc
import pytest

import ffsim

RNG = np.random.default_rng(316042750382801529624398161754791060982)


def assert_t2_has_correct_symmetry(t2: np.ndarray):
    nocc, _, nvrt, _ = t2.shape
    for i, j, a, b in itertools.product(
        range(nocc), range(nocc), range(nvrt), range(nvrt)
    ):
        np.testing.assert_allclose(t2[i, j, a, b], t2[j, i, b, a])


def sample_matrices(sampler, n_samples: int = 8000, dim: int = 4) -> np.ndarray:
    """Draw many matrices from a sampler, for testing its distribution."""
    return np.array([sampler(dim, seed=RNG) for _ in range(n_samples)])


def pair_matrix(two_body_tensor: np.ndarray) -> np.ndarray:
    """Return a two-body tensor as a matrix indexed by orbital pairs.

    The returned matrix is ``mat[(p, q), (r, s)] = two_body_tensor[p, q, s, r]``, which
    is the form in which the tensor is positive semidefinite.
    """
    dim = two_body_tensor.shape[0]
    return two_body_tensor.transpose(0, 1, 3, 2).reshape(dim**2, dim**2)


def test_assert_t2_has_correct_symmetry():
    """Test that t2 amplitudes from a real molecule passes our symmetry test."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["Be", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    scf = pyscf.scf.ROHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()
    t2aa, _, t2bb = ccsd.t2
    assert_t2_has_correct_symmetry(t2aa)
    assert_t2_has_correct_symmetry(t2bb)


@pytest.mark.parametrize("dtype", [float, complex])
def test_random_t2_amplitudes(dtype):
    """Test random t2 amplitudes."""
    norb = 5
    nocc = 3
    nvrt = norb - nocc
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=dtype)
    assert t2.shape == (nocc, nocc, nvrt, nvrt)
    assert t2.dtype == dtype
    assert_t2_has_correct_symmetry(t2)


def test_random_two_body_tensor_symmetry_real():
    """Test random real two-body tensor symmetry."""
    n_orbitals = 5
    two_body_tensor = ffsim.random.random_two_body_tensor(
        n_orbitals, seed=RNG, dtype=float
    )
    assert np.issubdtype(two_body_tensor.dtype, np.floating)
    for i, j, k, ell in itertools.product(range(n_orbitals), repeat=4):
        val = two_body_tensor[i, j, k, ell]
        np.testing.assert_allclose(two_body_tensor[k, ell, i, j], val)
        np.testing.assert_allclose(two_body_tensor[j, i, ell, k], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[ell, k, j, i], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[j, i, k, ell], val)
        np.testing.assert_allclose(two_body_tensor[ell, k, i, j], val)
        np.testing.assert_allclose(two_body_tensor[i, j, ell, k], val)
        np.testing.assert_allclose(two_body_tensor[k, ell, j, i], val)


def test_random_two_body_tensor_symmetry():
    """Test random two-body tensor symmetry."""
    n_orbitals = 5
    two_body_tensor = ffsim.random.random_two_body_tensor(n_orbitals, seed=RNG)
    for i, j, k, ell in itertools.product(range(n_orbitals), repeat=4):
        val = two_body_tensor[i, j, k, ell]
        np.testing.assert_allclose(two_body_tensor[k, ell, i, j], val)
        np.testing.assert_allclose(two_body_tensor[j, i, ell, k], val.conjugate())
        np.testing.assert_allclose(two_body_tensor[ell, k, j, i], val.conjugate())


@pytest.mark.parametrize("dtype", [float, complex])
def test_random_two_body_tensor_positive_semidefinite(dtype):
    """Test that the two-body tensor is positive semidefinite in the pair basis."""
    mat = pair_matrix(ffsim.random.random_two_body_tensor(5, seed=RNG, dtype=dtype))
    assert ffsim.linalg.is_hermitian(mat)
    assert np.min(np.linalg.eigvalsh(mat)) > -1e-8


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("rank", [10, 50])
def test_random_two_body_tensor_scale_independent_of_rank(dtype, rank: int):
    """Test that the scale of the two-body tensor does not depend on the rank."""
    dim = 5
    # In the pair basis, the diagonal of the tensor averages the squared magnitudes of
    # the entries of the Hermitian matrices being sampled. Those have variance four in
    # the complex case, and two off the diagonal and four on the diagonal in the real
    # case, and averaging leaves the expected value independent of the rank.
    expected = 4 if np.issubdtype(dtype, np.complexfloating) else 2 + 2 / dim
    means = [
        np.mean(
            np.diag(
                pair_matrix(
                    ffsim.random.random_two_body_tensor(
                        dim, rank=rank, seed=RNG, dtype=dtype
                    )
                )
            )
        ).real
        for _ in range(20)
    ]
    np.testing.assert_allclose(np.mean(means), expected, rtol=0.1)


@pytest.mark.parametrize("dim", range(10))
def test_random_unitary(dim: int):
    """Test random unitary."""
    mat = ffsim.random.random_unitary(dim, seed=RNG)
    assert mat.dtype == complex
    assert ffsim.linalg.is_unitary(mat)


@pytest.mark.parametrize("dim", range(10))
def test_random_orthogonal(dim: int):
    """Test random orthogonal."""
    mat = ffsim.random.random_orthogonal(dim, seed=RNG)
    assert mat.dtype == float
    assert ffsim.linalg.is_orthogonal(mat)

    mat = ffsim.random.random_orthogonal(dim, seed=RNG, dtype=complex)
    assert mat.dtype == complex
    assert ffsim.linalg.is_orthogonal(mat)


@pytest.mark.parametrize("dim", range(10))
def test_random_special_orthogonal(dim: int):
    """Test random special orthogonal."""
    mat = ffsim.random.random_special_orthogonal(dim, seed=RNG)
    assert mat.dtype == float
    assert ffsim.linalg.is_special_orthogonal(mat)

    mat = ffsim.random.random_special_orthogonal(dim, seed=RNG, dtype=np.float32)
    assert mat.dtype == np.float32
    assert ffsim.linalg.is_special_orthogonal(mat, atol=1e-5)


def test_random_real_symmetric_matrix():
    """Test random real symmetric matrix."""
    dim = 5
    mat = ffsim.random.random_real_symmetric_matrix(dim, seed=RNG)
    assert ffsim.linalg.is_real_symmetric(mat)
    np.testing.assert_allclose(np.linalg.matrix_rank(mat), dim)

    rank = 3
    mats = [
        ffsim.random.random_real_symmetric_matrix(dim, rank=rank, seed=RNG)
        for _ in range(10)
    ]
    for mat in mats:
        assert ffsim.linalg.is_real_symmetric(mat)
        np.testing.assert_allclose(np.linalg.matrix_rank(mat), rank)
    # The sampled matrices are indefinite, not positive semidefinite.
    eigs = np.linalg.eigvalsh(mats)
    assert np.any(eigs < -1e-8)
    assert np.any(eigs > 1e-8)


def test_random_real_symmetric_matrix_goe():
    """Test that random real symmetric matrices are distributed with the GOE."""
    mats = sample_matrices(ffsim.random.random_real_symmetric_matrix)
    # The diagonal entries have variance two and the off-diagonal entries have unit
    # variance.
    np.testing.assert_allclose(np.mean(mats), 0, atol=0.05)
    np.testing.assert_allclose(np.var(mats[:, 0, 0]), 2, rtol=0.1)
    np.testing.assert_allclose(np.var(mats[:, 0, 1]), 1, rtol=0.1)
    # Half of the eigenvalues are negative.
    eigs = np.linalg.eigvalsh(mats)
    np.testing.assert_allclose(np.mean(eigs < 0), 0.5, rtol=0.1)

    # The distribution is invariant under conjugation by an orthogonal matrix.
    orthogonal = ffsim.random.random_orthogonal(mats.shape[-1], seed=RNG)
    rotated = orthogonal @ mats @ orthogonal.T
    np.testing.assert_allclose(np.var(rotated[:, 0, 0]), 2, rtol=0.15)
    np.testing.assert_allclose(np.var(rotated[:, 0, 1]), 1, rtol=0.15)


@pytest.mark.parametrize("dim", range(10))
def test_random_hermitian(dim: int):
    """Test random Hermitian matrix."""
    mat = ffsim.random.random_hermitian(dim, seed=RNG)
    assert mat.dtype == complex
    assert ffsim.linalg.is_hermitian(mat)


def test_random_hermitian_gue():
    """Test that random Hermitian matrices are distributed with the GUE."""
    mats = sample_matrices(ffsim.random.random_hermitian)
    # The diagonal entries have unit variance, and the real and imaginary parts of the
    # off-diagonal entries have variance one half.
    np.testing.assert_allclose(np.mean(mats), 0, atol=0.05)
    np.testing.assert_allclose(np.var(mats[:, 0, 0].real), 1, rtol=0.1)
    np.testing.assert_allclose(np.var(mats[:, 0, 1].real), 0.5, rtol=0.1)
    np.testing.assert_allclose(np.var(mats[:, 0, 1].imag), 0.5, rtol=0.1)

    # The distribution is invariant under conjugation by a unitary.
    unitary = ffsim.random.random_unitary(mats.shape[-1], seed=RNG)
    rotated = unitary @ mats @ unitary.T.conj()
    np.testing.assert_allclose(np.var(rotated[:, 0, 0].real), 1, rtol=0.15)
    np.testing.assert_allclose(np.var(rotated[:, 0, 1].real), 0.5, rtol=0.15)
    np.testing.assert_allclose(np.var(rotated[:, 0, 1].imag), 0.5, rtol=0.15)


@pytest.mark.parametrize("dim", range(10))
def test_random_antihermitian_matrix(dim: int):
    """Test random anti-Hermitian matrix."""
    mat = ffsim.random.random_antihermitian(dim, seed=RNG)
    assert ffsim.linalg.is_antihermitian(mat)


def test_random_antihermitian_gue():
    """Test that random anti-Hermitian matrices are 1j times a GUE sample."""
    mats = -1j * sample_matrices(ffsim.random.random_antihermitian)
    np.testing.assert_allclose(np.mean(mats), 0, atol=0.05)
    np.testing.assert_allclose(np.var(mats[:, 0, 0].real), 1, rtol=0.1)
    np.testing.assert_allclose(np.var(mats[:, 0, 1].real), 0.5, rtol=0.1)
    np.testing.assert_allclose(np.var(mats[:, 0, 1].imag), 0.5, rtol=0.1)


@pytest.mark.parametrize("dim", range(1, 10))
def test_random_state_vector(dim: int):
    """Test random state vector."""
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    assert vec.dtype == complex
    np.testing.assert_allclose(np.linalg.norm(vec), 1)

    vec = ffsim.random.random_state_vector(dim, seed=RNG, dtype=float)
    assert vec.dtype == float
    np.testing.assert_allclose(np.linalg.norm(vec), 1)


@pytest.mark.parametrize("dim", range(1, 10))
def test_random_density_matrix(dim: int):
    """Test random density matrix."""
    mat = ffsim.random.random_density_matrix(dim, seed=RNG)
    assert mat.dtype == complex
    assert ffsim.linalg.is_hermitian(mat)
    eigs, _ = np.linalg.eigh(mat)
    assert all(eigs >= 0)
    np.testing.assert_allclose(np.trace(mat), 1)

    mat = ffsim.random.random_density_matrix(dim, seed=RNG, dtype=float)
    assert mat.dtype == float
    assert ffsim.linalg.is_hermitian(mat)
    eigs, _ = np.linalg.eigh(mat)
    assert all(eigs >= 0)
    np.testing.assert_allclose(np.trace(mat), 1)


def test_random_diagonal_coulomb_hamiltonian():
    """Test random diagonal Coulomb Hamiltonian."""
    norb = 5

    dc_ham = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=RNG)
    assert dc_ham.one_body_tensor.dtype == complex

    dc_ham = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=RNG, real=True)
    assert dc_ham.one_body_tensor.dtype == float


def test_random_fermion_operator():
    """Test random fermion operator."""
    norb = 5

    # Generic operator
    op = ffsim.random.random_fermion_operator(
        norb, n_terms=10, max_term_length=3, seed=RNG
    )
    assert len(op) <= 10
    assert all(len(term) <= 3 for term in op)


def test_random_fermion_operator_num_and_spin_conserving():
    """Test random number- and spin-z-conserving fermion operator."""
    norb = 5

    op = ffsim.random.random_fermion_operator(
        norb, n_terms=20, num_and_spin_conserving=True, seed=RNG
    )
    assert op.conserves_particle_number()
    assert op.conserves_spin_z()
    # The default max_term_length is 2 * norb.
    assert all(len(term) <= 2 * norb for term in op)

    op = ffsim.random.random_fermion_operator(
        norb, n_terms=20, max_term_length=6, num_and_spin_conserving=True, seed=RNG
    )
    assert op.conserves_particle_number()
    assert op.conserves_spin_z()
    assert all(len(term) <= 6 for term in op)


def test_random_fermion_hamiltonian():
    """Test random fermion Hamiltonian."""
    norb = 5

    op = ffsim.random.random_fermion_hamiltonian(norb, n_terms=10, seed=RNG)
    assert op.conserves_particle_number()
    assert op.conserves_spin_z()
    # A Hamiltonian is Hermitian.
    assert op == op.adjoint()


@pytest.mark.parametrize("dtype", [float, complex])
def test_random_molecular_hamiltonian(dtype):
    """Test random molecular Hamiltonian."""
    norb = 4
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=dtype)
    assert hamiltonian.one_body_tensor.dtype == dtype
    assert hamiltonian.two_body_tensor.dtype == dtype
    assert ffsim.linalg.is_hermitian(hamiltonian.one_body_tensor)

    # A Hamiltonian is Hermitian.
    nelec = (2, 1)
    dim = ffsim.dim(norb, nelec)
    mat = ffsim.linear_operator(hamiltonian, norb, nelec) @ np.eye(dim)
    np.testing.assert_allclose(mat, mat.T.conj(), atol=1e-12)


@pytest.mark.parametrize(
    "sampler",
    [
        ffsim.random.random_molecular_hamiltonian,
        ffsim.random.random_molecular_hamiltonian_spinless,
    ],
)
@pytest.mark.parametrize("dtype", [float, complex])
def test_random_molecular_hamiltonian_scales(sampler, dtype):
    """Test scaling the terms of a random molecular Hamiltonian."""
    norb = 4
    seed = RNG.integers(1 << 32)
    hamiltonian = sampler(norb, seed=seed, dtype=dtype)
    scaled = sampler(
        norb,
        one_body_scale=2.0,
        two_body_scale=0.5,
        constant_scale=0.0,
        seed=seed,
        dtype=dtype,
    )
    np.testing.assert_allclose(scaled.one_body_tensor, 2 * hamiltonian.one_body_tensor)
    np.testing.assert_allclose(
        scaled.two_body_tensor, 0.5 * hamiltonian.two_body_tensor
    )
    assert scaled.constant == 0


@pytest.mark.parametrize("rank", [1, 5, 10])
def test_random_molecular_hamiltonian_rank(rank: int):
    """Test the rank of the two-body tensor of a random molecular Hamiltonian."""
    norb = 4
    hamiltonian = ffsim.random.random_molecular_hamiltonian(
        norb, rank=rank, seed=RNG, dtype=float
    )
    np.testing.assert_allclose(
        np.linalg.matrix_rank(pair_matrix(hamiltonian.two_body_tensor)), rank
    )


@pytest.mark.parametrize("dtype", [float, complex])
def test_random_molecular_hamiltonian_unrestricted_scales(dtype):
    """Test scaling the terms of a random unrestricted molecular Hamiltonian."""
    norb = 4
    seed = RNG.integers(1 << 32)
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(
        norb, seed=seed, dtype=dtype
    )
    scaled = ffsim.random.random_molecular_hamiltonian_unrestricted(
        norb,
        one_body_scale=2.0,
        two_body_scale=0.5,
        constant_scale=3.0,
        seed=seed,
        dtype=dtype,
    )
    np.testing.assert_allclose(
        scaled.one_body_tensors, 2 * hamiltonian.one_body_tensors
    )
    np.testing.assert_allclose(
        scaled.two_body_tensors, 0.5 * hamiltonian.two_body_tensors
    )
    np.testing.assert_allclose(scaled.constant, 3 * hamiltonian.constant)


def test_random_double_factorized_hamiltonian_scale_independent_of_rank():
    """Test that the scale of the DF two-body part does not depend on the rank."""
    norb = 4

    def mean_norm(rank: int) -> float:
        return float(
            np.mean(
                [
                    np.linalg.norm(
                        ffsim.random.random_double_factorized_hamiltonian(
                            norb, rank=rank, real=True, seed=RNG
                        )
                        .to_molecular_hamiltonian()
                        .two_body_tensor
                    )
                    for _ in range(50)
                ]
            )
        )

    np.testing.assert_allclose(mean_norm(2), mean_norm(20), rtol=0.15)


@pytest.mark.parametrize("dtype", [float, complex])
def test_random_molecular_hamiltonian_unrestricted_symmetries(dtype):
    """Test symmetries of the random spin-unrestricted molecular Hamiltonian."""
    norb = 4
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(
        norb, seed=RNG, dtype=dtype
    )
    assert hamiltonian.one_body_tensors.dtype == dtype
    assert hamiltonian.two_body_tensors.dtype == dtype
    two_body_aa, two_body_ab, two_body_bb = hamiltonian.two_body_tensors

    # The same-spin tensors are symmetric under exchanging their two index pairs.
    np.testing.assert_allclose(two_body_aa, two_body_aa.transpose(2, 3, 0, 1))
    np.testing.assert_allclose(two_body_bb, two_body_bb.transpose(2, 3, 0, 1))
    # The alpha-beta tensor is not, since the beta-alpha term of the Hamiltonian
    # supplies the transposed contribution.
    assert not np.allclose(two_body_ab, two_body_ab.transpose(2, 3, 0, 1))
    # All tensors retain the symmetry within each index pair that makes the
    # Hamiltonian Hermitian.
    for tensor in hamiltonian.two_body_tensors:
        np.testing.assert_allclose(tensor, tensor.transpose(1, 0, 3, 2).conj())
    for tensor in hamiltonian.one_body_tensors:
        np.testing.assert_allclose(tensor, tensor.T.conj())

    # The spin sectors are sampled independently.
    assert not np.allclose(*hamiltonian.one_body_tensors)
    assert not np.allclose(two_body_aa, two_body_bb)

    # A Hamiltonian is Hermitian.
    nelec = (2, 1)
    dim = ffsim.dim(norb, nelec)
    mat = ffsim.linear_operator(hamiltonian, norb, nelec) @ np.eye(dim)
    np.testing.assert_allclose(mat, mat.T.conj(), atol=1e-12)


@pytest.mark.parametrize("norb", range(1, 5))
@pytest.mark.parametrize("n_reps", range(1, 4))
def test_random_ucj_op_spinless_empty_interaction_pairs(norb: int, n_reps: int):
    """Test sampling spinless UCJ operator with empty interaction pairs."""
    op = ffsim.random.random_ucj_op_spinless(
        norb, n_reps=n_reps, interaction_pairs=[], seed=RNG
    )
    np.testing.assert_allclose(op.diag_coulomb_mats, np.zeros((n_reps, norb, norb)))


@pytest.mark.parametrize("norb", range(1, 5))
@pytest.mark.parametrize("n_reps", range(1, 4))
def test_random_ucj_op_spin_balanced_empty_interaction_pairs(norb: int, n_reps: int):
    """Test sampling spin-balanced UCJ operator with empty interaction pairs."""
    op = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=n_reps, interaction_pairs=([], []), seed=RNG
    )
    np.testing.assert_allclose(op.diag_coulomb_mats, np.zeros((n_reps, 2, norb, norb)))

    # An empty list zeros out only its own diagonal Coulomb matrices
    for index in range(2):
        interaction_pairs: list[list[tuple[int, int]] | None] = [None, None]
        interaction_pairs[index] = []
        op = ffsim.random.random_ucj_op_spin_balanced(
            norb,
            n_reps=n_reps,
            interaction_pairs=(interaction_pairs[0], interaction_pairs[1]),
            seed=RNG,
        )
        np.testing.assert_allclose(
            op.diag_coulomb_mats[:, index], np.zeros((n_reps, norb, norb))
        )
        for other in set(range(2)) - {index}:
            assert np.any(op.diag_coulomb_mats[:, other])


@pytest.mark.parametrize("norb", range(1, 5))
@pytest.mark.parametrize("n_reps", range(1, 4))
def test_random_ucj_op_spin_unbalanced_empty_interaction_pairs(norb: int, n_reps: int):
    """Test sampling spin-unbalanced UCJ operator with empty interaction pairs."""
    op = ffsim.random.random_ucj_op_spin_unbalanced(
        norb, n_reps=n_reps, interaction_pairs=([], [], []), seed=RNG
    )
    np.testing.assert_allclose(op.diag_coulomb_mats, np.zeros((n_reps, 3, norb, norb)))

    # An empty list zeros out only its own diagonal Coulomb matrices
    for index in range(3):
        interaction_pairs: list[list[tuple[int, int]] | None] = [None, None, None]
        interaction_pairs[index] = []
        op = ffsim.random.random_ucj_op_spin_unbalanced(
            norb,
            n_reps=n_reps,
            interaction_pairs=(
                interaction_pairs[0],
                interaction_pairs[1],
                interaction_pairs[2],
            ),
            seed=RNG,
        )
        np.testing.assert_allclose(
            op.diag_coulomb_mats[:, index], np.zeros((n_reps, norb, norb))
        )
        for other in set(range(3)) - {index}:
            assert np.any(op.diag_coulomb_mats[:, other])


@pytest.mark.parametrize("diag_coulomb_normal", [False, True])
def test_random_ucj_op_diag_coulomb_distribution(diag_coulomb_normal: bool):
    """Test the mean and scale of the sampled diagonal Coulomb matrices."""
    norb = 4
    mean = 10.0
    scale = 1e-3
    op_balanced = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=2,
        diag_coulomb_mean=mean,
        diag_coulomb_scale=scale,
        diag_coulomb_normal=diag_coulomb_normal,
        seed=RNG,
    )
    op_unbalanced = ffsim.random.random_ucj_op_spin_unbalanced(
        norb,
        n_reps=2,
        diag_coulomb_mean=mean,
        diag_coulomb_scale=scale,
        diag_coulomb_normal=diag_coulomb_normal,
        seed=RNG,
    )
    op_spinless = ffsim.random.random_ucj_op_spinless(
        norb,
        n_reps=2,
        diag_coulomb_mean=mean,
        diag_coulomb_scale=scale,
        diag_coulomb_normal=diag_coulomb_normal,
        seed=RNG,
    )
    for diag_coulomb_mats in [
        op_balanced.diag_coulomb_mats,
        op_unbalanced.diag_coulomb_mats,
        op_spinless.diag_coulomb_mats,
    ]:
        np.testing.assert_allclose(diag_coulomb_mats, mean, atol=1e-2)


@pytest.mark.parametrize("norb", range(1, 5))
def test_random_givens_ansatz_op(norb: int):
    """Test sampling a Givens rotation ansatz operator."""
    op = ffsim.random.random_givens_ansatz_op(norb, seed=RNG)
    assert op.norb == norb
    assert len(op.thetas) == len(op.interaction_pairs)
    assert op.phis is not None
    assert len(op.phis) == len(op.interaction_pairs)
    assert op.phase_angles is not None
    assert len(op.phase_angles) == norb
    assert ffsim.linalg.is_unitary(op.to_orbital_rotation())

    # The Givens rotation phases and the layer of phase gates are optional
    op = ffsim.random.random_givens_ansatz_op(
        norb, with_phis=False, with_phase_angles=False, seed=RNG
    )
    assert op.phis is None
    assert op.phase_angles is None

    # The interaction pairs can be specified
    interaction_pairs = list(itertools.combinations(range(norb), 2))
    op = ffsim.random.random_givens_ansatz_op(
        norb, interaction_pairs=interaction_pairs, seed=RNG
    )
    assert op.interaction_pairs == interaction_pairs
    assert len(op.thetas) == len(interaction_pairs)


@pytest.mark.parametrize("norb", range(1, 5))
def test_random_num_num_ansatz_op_spin_balanced(norb: int):
    """Test sampling a spin-balanced number-number interaction ansatz operator."""
    op = ffsim.random.random_num_num_ansatz_op_spin_balanced(norb, seed=RNG)
    assert op.norb == norb
    # By default, all pairs of orbitals interact, including an orbital with itself
    pairs = list(itertools.combinations_with_replacement(range(norb), 2))
    for these_pairs, thetas in zip(op.interaction_pairs, op.thetas):
        assert these_pairs == pairs
        assert len(thetas) == len(pairs)

    # The interaction pairs can be specified
    pairs_aa: list[tuple[int, int]] = [(0, norb - 1)]
    pairs_ab: list[tuple[int, int]] = []
    op = ffsim.random.random_num_num_ansatz_op_spin_balanced(
        norb, interaction_pairs=(pairs_aa, pairs_ab), seed=RNG
    )
    assert op.interaction_pairs == (pairs_aa, pairs_ab)
    assert len(op.thetas[0]) == len(pairs_aa)
    assert len(op.thetas[1]) == len(pairs_ab)


def test_raise_errors():
    """Test errors are raised as expected."""
    with pytest.raises(ValueError, match="Dimension"):
        _ = ffsim.random.random_state_vector(0, seed=RNG)

    with pytest.raises(ValueError, match="Dimension"):
        _ = ffsim.random.random_density_matrix(0, seed=RNG)

    # Each list of interaction pairs is validated, including the alpha-beta list,
    # which is allowed to contain lower triangular pairs but not duplicates.
    with pytest.raises(ValueError, match="Duplicate"):
        _ = ffsim.random.random_ucj_op_spin_unbalanced(
            4, interaction_pairs=(None, [(0, 1), (0, 1)], None), seed=RNG
        )
    with pytest.raises(ValueError, match="triangular"):
        _ = ffsim.random.random_ucj_op_spin_unbalanced(
            4, interaction_pairs=([(1, 0)], None, None), seed=RNG
        )
    with pytest.raises(ValueError, match="triangular"):
        _ = ffsim.random.random_ucj_op_spin_unbalanced(
            4, interaction_pairs=(None, None, [(1, 0)]), seed=RNG
        )

    with pytest.raises(ValueError, match="Duplicate"):
        _ = ffsim.random.random_num_num_ansatz_op_spin_balanced(
            4, interaction_pairs=([(0, 1), (0, 1)], []), seed=RNG
        )
    with pytest.raises(ValueError, match="triangular"):
        _ = ffsim.random.random_num_num_ansatz_op_spin_balanced(
            4, interaction_pairs=([], [(1, 0)]), seed=RNG
        )
