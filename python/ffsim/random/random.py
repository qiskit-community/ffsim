# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
from typing_extensions import deprecated

from ffsim import hamiltonians, operators, variational


@deprecated(
    "ffsim.random.random_statevector is deprecated. "
    "Instead, use ffsim.random.random_state_vector."
)
def random_statevector(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random state vector sampled from the uniform distribution.

    .. warning::
        This function is deprecated. Use :func:`ffsim.random.random_state_vector`
        instead.

    Args:
        dim: The dimension of the state vector.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled state vector.
    """
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(dtype, copy=False)
    if np.issubdtype(dtype, np.complexfloating):
        vec += 1j * rng.standard_normal(dim).astype(dtype, copy=False)
    vec /= np.linalg.norm(vec)
    return vec


def random_state_vector(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random state vector sampled from the uniform distribution.

    Args:
        dim: The dimension of the state vector.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled state vector.
    """
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(dtype, copy=False)
    if np.issubdtype(dtype, np.complexfloating):
        vec += 1j * rng.standard_normal(dim).astype(dtype, copy=False)
    vec /= np.linalg.norm(vec)
    return vec


def random_unitary(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random unitary matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled unitary matrix.

    References:
        - `arXiv:math-ph/0609050`_

    .. _arXiv:math-ph/0609050: https://arxiv.org/abs/math-ph/0609050
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    z += 1j * rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def random_orthogonal(dim: int, seed=None, dtype=float) -> np.ndarray:
    """Return a random orthogonal matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled orthogonal matrix.

    References:
        - `arXiv:math-ph/0609050`_

    .. _arXiv:math-ph/0609050: https://arxiv.org/abs/math-ph/0609050
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    q, r = np.linalg.qr(m)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def random_special_orthogonal(dim: int, seed=None, dtype=float) -> np.ndarray:
    """Return a random special orthogonal matrix distributed with Haar measure.

    Args:
        dim: The width and height of the matrix.
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled special orthogonal matrix.
    """
    mat = random_orthogonal(dim, seed=seed, dtype=dtype)
    if np.linalg.det(mat) < 0:
        mat[0] *= -1
    return mat


def random_hermitian(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random Hermitian matrix.

    Args:
        dim: The width and height of the matrix.
        rank: The rank of the matrix. If `None`, the maximum rank is used.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled Hermitian matrix.
    """
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    mat += 1j * rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    return mat + mat.T.conj()


def random_real_symmetric_matrix(
    dim: int, *, rank: int | None = None, seed=None, dtype=float
) -> np.ndarray:
    """Return a random real symmetric matrix.

    Args:
        dim: The width and height of the matrix.
        rank: The rank of the matrix. If `None`, the maximum rank is used.
        seed: The pseudorandom number generator or seed. Should be an
            instance of `np.random.Generator` or else a valid input to
            `np.random.default_rng`.

    Returns:
        The sampled real symmetric matrix.
    """
    rng = np.random.default_rng(seed)
    if rank is None:
        rank = dim
    mat = rng.standard_normal((dim, rank)).astype(dtype, copy=False)
    return mat @ mat.T


def random_antihermitian(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random anti-Hermitian matrix.

    Args:
        dim: The width and height of the matrix.
        rank: The rank of the matrix. If `None`, the maximum rank is used.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled anti-Hermitian matrix.
    """
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    mat += 1j * rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    return mat - mat.T.conj()


def random_two_body_tensor(
    dim: int, *, rank: int | None = None, seed=None, dtype=complex
) -> np.ndarray:
    """Sample a random two-body tensor.

    Args:
        dim: The dimension of the tensor. The shape of the returned tensor will be
            (dim, dim, dim, dim).
        rank: Rank of the sampled tensor. The default behavior is to use
            the maximum rank, which is `norb * (norb + 1) // 2`.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled two-body tensor.
    """
    rng = np.random.default_rng(seed)
    if rank is None:
        rank = dim * (dim + 1) // 2
    cholesky_vecs = rng.standard_normal((rank, dim, dim)).astype(dtype, copy=False)
    cholesky_vecs += cholesky_vecs.transpose((0, 2, 1))
    two_body_tensor = np.einsum("ipr,iqs->prqs", cholesky_vecs, cholesky_vecs)
    if np.issubdtype(dtype, np.complexfloating):
        orbital_rotation = random_unitary(dim, seed=rng)
        two_body_tensor = np.einsum(
            "abcd,aA,bB,cC,dD->ABCD",
            two_body_tensor,
            orbital_rotation,
            orbital_rotation.conj(),
            orbital_rotation,
            orbital_rotation.conj(),
            optimize=True,
        )
    return two_body_tensor


def random_t2_amplitudes(
    norb: int, nocc: int, *, seed=None, dtype=complex
) -> np.ndarray:
    """Sample a random t2 amplitudes tensor.

    Args:
        norb: The number of orbitals.
        nocc: The number of orbitals that are occupied by an electron.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the result.

    Returns:
        The sampled t2 amplitudes tensor.
    """
    rng = np.random.default_rng(seed)
    nvrt = norb - nocc
    t2 = np.zeros((nocc, nocc, nvrt, nvrt), dtype=dtype)
    pairs = itertools.product(range(nocc), range(nocc, norb))
    for (i, a), (j, b) in itertools.combinations_with_replacement(pairs, 2):
        val = rng.standard_normal()
        t2[i, j, a - nocc, b - nocc] = val
        t2[j, i, b - nocc, a - nocc] = val
    if np.issubdtype(dtype, np.complexfloating):
        t2_large = np.zeros((norb, norb, norb, norb), dtype=dtype)
        t2_large[:nocc, :nocc, nocc:, nocc:] = t2
        orbital_rotation = random_unitary(norb, seed=rng)
        t2_large = np.einsum(
            "ijab,iI,jJ,aA,bB->IJAB",
            t2_large,
            orbital_rotation.conj(),
            orbital_rotation.conj(),
            orbital_rotation,
            orbital_rotation,
        )
        t2 = t2_large[:nocc, :nocc, nocc:, nocc:]
    return t2


def random_molecular_hamiltonian(
    norb: int, seed=None, dtype=complex
) -> hamiltonians.MolecularHamiltonian:
    """Sample a random molecular Hamiltonian.

    Args:
        norb: The number of spatial orbitals.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dtype: The data type to use for the one- and two-body tensors. The constant
            term will always be of type ``float``.

    Returns:
        The sampled molecular Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.complexfloating):
        one_body_tensor = random_hermitian(norb, seed=rng, dtype=dtype)
    else:
        one_body_tensor = random_real_symmetric_matrix(norb, seed=rng, dtype=dtype)
    two_body_tensor = random_two_body_tensor(norb, seed=rng, dtype=dtype)
    constant = rng.standard_normal()
    return hamiltonians.MolecularHamiltonian(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        constant=constant,
    )


@deprecated(
    "The random_ucj_operator function is deprecated. Use "
    "random_ucj_operator_closed_shell or random_ucj_operator_open_shell instead."
)
def random_ucj_operator(
    norb: int,
    *,
    n_reps: int = 1,
    with_final_orbital_rotation: bool = False,
    seed=None,
) -> variational.UCJOperator:
    """Sample a random unitary cluster Jastrow (UCJ) operator.

    Args:
        norb: The number of orbitals.
        n_reps: The number of ansatz repetitions.
        with_final_orbital_rotation: Whether to include a final orbital rotation
            in the operator.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled UCJ operator.
    """
    rng = np.random.default_rng(seed)
    diag_coulomb_mats_alpha_alpha = np.stack(
        [random_real_symmetric_matrix(norb, seed=rng) for _ in range(n_reps)]
    )
    diag_coulomb_mats_alpha_beta = np.stack(
        [random_real_symmetric_matrix(norb, seed=rng) for _ in range(n_reps)]
    )
    orbital_rotations = np.stack(
        [random_unitary(norb, seed=rng) for _ in range(n_reps)]
    )
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = random_unitary(norb, seed=rng)
    return variational.UCJOperator(
        diag_coulomb_mats_alpha_alpha=diag_coulomb_mats_alpha_alpha,
        diag_coulomb_mats_alpha_beta=diag_coulomb_mats_alpha_beta,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )


def random_ucj_op_spin_balanced(
    norb: int,
    *,
    n_reps: int = 1,
    with_final_orbital_rotation: bool = False,
    seed=None,
) -> variational.UCJOpSpinBalanced:
    """Sample a random spin-balanced unitary cluster Jastrow (UCJ) operator.

    Args:
        norb: The number of spatial orbitals.
        n_reps: The number of ansatz repetitions.
        with_final_orbital_rotation: Whether to include a final orbital rotation
            in the operator.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled UCJ operator.
    """
    rng = np.random.default_rng(seed)
    diag_coulomb_mats = np.stack(
        [
            np.stack(
                [
                    random_real_symmetric_matrix(norb, seed=rng),
                    random_real_symmetric_matrix(norb, seed=rng),
                ]
            )
            for _ in range(n_reps)
        ]
    )
    orbital_rotations = np.stack(
        [random_unitary(norb, seed=rng) for _ in range(n_reps)]
    )
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = random_unitary(norb, seed=rng)
    return variational.UCJOpSpinBalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )


def random_ucj_op_spin_unbalanced(
    norb: int,
    *,
    n_reps: int = 1,
    with_final_orbital_rotation: bool = False,
    seed=None,
) -> variational.UCJOpSpinUnbalanced:
    """Sample a random spin-unbalanced unitary cluster Jastrow (UCJ) operator.

    Args:
        norb: The number of orbitals.
        n_reps: The number of ansatz repetitions.
        with_final_orbital_rotation: Whether to include a final orbital rotation
            in the operator.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled UCJ operator.
    """
    rng = np.random.default_rng(seed)
    diag_coulomb_mats = np.stack(
        [
            np.stack(
                [
                    random_real_symmetric_matrix(norb, seed=rng),
                    rng.standard_normal((norb, norb)),
                    random_real_symmetric_matrix(norb, seed=rng),
                ]
            )
            for _ in range(n_reps)
        ]
    )
    orbital_rotations = np.stack(
        [
            np.stack([random_unitary(norb, seed=rng), random_unitary(norb, seed=rng)])
            for _ in range(n_reps)
        ]
    )
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = np.stack(
            [random_unitary(norb, seed=rng), random_unitary(norb, seed=rng)]
        )
    return variational.UCJOpSpinUnbalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )


def random_ucj_op_spinless(
    norb: int,
    *,
    n_reps: int = 1,
    with_final_orbital_rotation: bool = False,
    seed=None,
) -> variational.UCJOpSpinless:
    """Sample a random spinless unitary cluster Jastrow (UCJ) operator.

    Args:
        norb: The number of orbitals.
        n_reps: The number of ansatz repetitions.
        with_final_orbital_rotation: Whether to include a final orbital rotation
            in the operator.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled UCJ operator.
    """
    rng = np.random.default_rng(seed)
    diag_coulomb_mats = np.stack(
        [random_real_symmetric_matrix(norb, seed=rng) for _ in range(n_reps)]
    )
    orbital_rotations = np.stack(
        [random_unitary(norb, seed=rng) for _ in range(n_reps)]
    )
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = random_unitary(norb, seed=rng)
    return variational.UCJOpSpinless(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )


def random_diagonal_coulomb_hamiltonian(
    norb: int, *, seed=None
) -> hamiltonians.DiagonalCoulombHamiltonian:
    """Sample a random diagonal Coulomb Hamiltonian.

    Args:
        norb: The number of spatial orbitals.
        rank: The desired number of terms in the two-body part of the Hamiltonian.
            If not specified, it will be set to ``norb * (norb + 1) // 2``.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled diagonal Coulomb Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    one_body_tensor = random_hermitian(norb, seed=rng)
    diag_coulomb_mat_a = random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb_mat_b = random_real_symmetric_matrix(norb, seed=rng)
    diag_coulomb_mats = np.stack([diag_coulomb_mat_a, diag_coulomb_mat_b])
    constant = rng.standard_normal()
    return hamiltonians.DiagonalCoulombHamiltonian(
        one_body_tensor, diag_coulomb_mats, constant=constant
    )


def random_double_factorized_hamiltonian(
    norb: int,
    *,
    rank: int | None = None,
    z_representation: bool = False,
    real: bool = False,
    seed=None,
) -> hamiltonians.DoubleFactorizedHamiltonian:
    """Sample a random double-factorized Hamiltonian.

    Args:
        norb: The number of spatial orbitals.
        rank: The desired number of terms in the two-body part of the Hamiltonian.
            If not specified, it will be set to ``norb * (norb + 1) // 2``.
        z_representation: Whether to return a Hamiltonian in the "Z" representation.
        real: Whether to sample a real-valued object rather than a complex-valued one.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled double-factorized Hamiltonian.
    """
    if rank is None:
        rank = norb * (norb + 1) // 2
    rng = np.random.default_rng(seed)
    if real:
        one_body_tensor = random_real_symmetric_matrix(norb, seed=rng)
        orbital_rotations = np.stack(
            [random_orthogonal(norb, seed=rng) for _ in range(rank)]
        )
    else:
        one_body_tensor = random_hermitian(norb, seed=rng)
        orbital_rotations = np.stack(
            [random_unitary(norb, seed=rng) for _ in range(rank)]
        )
    diag_coulomb_mats = np.stack(
        [random_real_symmetric_matrix(norb, seed=rng) for _ in range(rank)]
    )
    constant = rng.standard_normal()
    return hamiltonians.DoubleFactorizedHamiltonian(
        one_body_tensor=one_body_tensor,
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        constant=constant,
        z_representation=z_representation,
    )


def random_fermion_operator(
    norb: int, n_terms: int | None = None, max_term_length: int | None = None, seed=None
) -> operators.FermionOperator:
    """Sample a random fermion operator.

    Args:
        norb: The number of spatial orbitals.
        n_terms: The number of terms to include in the operator. If not specified,
            `norb` is used.
        max_term_length: The maximum length of a term. If not specified, `norb` is used.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled fermion operator.
    """
    rng = np.random.default_rng(seed)
    if n_terms is None:
        n_terms = norb
    if max_term_length is None:
        max_term_length = norb
    coeffs: defaultdict[tuple[tuple[bool, bool, int], ...], complex] = defaultdict(
        complex
    )
    for _ in range(n_terms):
        term_length = int(rng.integers(1, max_term_length + 1))
        actions = [bool(i) for i in rng.integers(2, size=term_length)]
        spins = [bool(i) for i in rng.integers(2, size=term_length)]
        indices = [int(i) for i in rng.integers(norb, size=term_length)]
        coeff = rng.standard_normal() + 1j * rng.standard_normal()
        term = tuple(zip(actions, spins, indices))
        coeffs[term] += coeff
    return operators.FermionOperator(coeffs)


def random_fermion_hamiltonian(
    norb: int, n_terms: int | None = None, seed=None
) -> operators.FermionOperator:
    """Sample a random fermion Hamiltonian.

    A fermion Hamiltonian is hermitian and conserves particle number and spin Z.

    Args:
        norb: The number of spatial orbitals.
        n_terms: The number of terms to include in the operator. If not specified,
            `norb` is used.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled fermion Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    if n_terms is None:
        n_terms = norb
    coeffs: defaultdict[tuple[tuple[bool, bool, int], ...], complex] = defaultdict(
        complex
    )
    for _ in range(n_terms):
        n_excitations = int(rng.integers(1, norb + 1))
        term = _random_num_and_spin_z_conserving_term(norb, n_excitations, seed=rng)
        term_adjoint = _adjoint_term(term)
        coeff = rng.standard_normal() + 1j * rng.standard_normal()
        coeffs[term] += coeff
        coeffs[term_adjoint] += coeff.conjugate()
    return operators.FermionOperator(coeffs)


def _random_num_and_spin_z_conserving_term(
    norb: int, n_excitations: int, seed=None
) -> tuple[tuple[bool, bool, int], ...]:
    rng = np.random.default_rng(seed)
    term = []
    for _ in range(n_excitations):
        spin = bool(rng.integers(2))
        orb_1, orb_2 = [int(x) for x in rng.integers(norb, size=2)]
        action_1, action_2 = [
            bool(x) for x in rng.choice([True, False], size=2, replace=False)
        ]
        term.append(operators.FermionAction(action_1, spin, orb_1))
        term.append(operators.FermionAction(action_2, spin, orb_2))
    return tuple(term)


def _adjoint_term(
    term: tuple[tuple[bool, bool, int], ...],
) -> tuple[tuple[bool, bool, int], ...]:
    return tuple(
        operators.FermionAction(bool(1 - action), spin, orb)
        for action, spin, orb in reversed(term)
    )
