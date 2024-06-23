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

import numpy as np
from typing_extensions import deprecated

from ffsim import hamiltonians, variational


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
        norb: The number of orbitals.
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
