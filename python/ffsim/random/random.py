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


def random_statevector(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    """Return a random state vector sampled from the uniform distribution.

    Args:
        dim: The dimension of the state vector.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.
        dype: The data type to use for the result.

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
        dype: The data type to use for the result.

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
        dype: The data type to use for the result.

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
        dype: The data type to use for the result.

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
        dype: The data type to use for the result.

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
        dype: The data type to use for the result.

    Returns:
        The sampled t2 amplitudes tensor.
    """
    rng = np.random.default_rng(seed)
    nvrt = norb - nocc
    t2 = np.zeros((nocc, nocc, nvrt, nvrt), dtype=dtype)
    pairs = itertools.product(range(nocc), range(nocc, norb))
    for (m, (i, a)), (n, (j, b)) in itertools.product(enumerate(pairs), repeat=2):
        if m <= n:
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
