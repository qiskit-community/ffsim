# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear algebra utilities."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


def antihermitian_to_parameters(mat: np.ndarray, real: bool = False) -> np.ndarray:
    """Convert an antihermitian matrix to parameters.

    Converts an antihermitian matrix to a real-valued parameter vector.

    Args:
        mat: The antihermitian matrix.
        real: Whether to take only the real part of the matrix, and discard the
            imaginary part.

    Returns:
        The list of real numbers parameterizing the antihermitian matrix.
    """
    return antihermitians_to_parameters(mat[None, :], real=real)


def antihermitian_from_parameters(
    params: np.ndarray, dim: int, real: bool = False
) -> np.ndarray:
    """Construct an antihermitian matrix from parameters.

    Converts a real-valued parameter vector to an antihermitian matrix.

    Args:
        params: The real-valued parameters.
        dim: The width and height of the matrix.
        real: Whether the parameter vector describes a real-valued antihermitian matrix.

    Returns:
        The antihermitian matrix.
    """
    return antihermitians_from_parameters(params, dim, 1, real=real)[0]


def antihermitian_from_parameters_jax(
    params: np.ndarray, dim: int, real: bool = False
) -> jax.Array:
    """JAX version of antihermitian_from_parameters."""
    return antihermitians_from_parameters_jax(params, dim, 1, real=real)[0]


def antihermitians_to_parameters(mats: np.ndarray, real: bool = False) -> np.ndarray:
    """Convert a batch of antihermitian matrices to parameters.

    Converts an array of antihermitian matrices to a real-valued parameter vector.

    Args:
        mats: The batch of antihermitian matrices, with shape (n_mats, dim, dim).
        real: Whether to take only the real part of the matrices, and discard the
            imaginary part.

    Returns:
        The list of real numbers parameterizing the antihermitian matrices.
    """
    n_mats, dim, _ = mats.shape
    triu_indices = np.triu_indices(dim, k=1)
    n_triu = dim * (dim - 1) // 2
    n_params_per_mat = n_triu if real else dim**2
    params = np.zeros((n_mats, n_params_per_mat))
    # real part
    params[:, :n_triu] = mats[:, triu_indices[0], triu_indices[1]].real
    # imaginary part
    if not real:
        triu_indices = np.triu_indices(dim)
        params[:, n_triu:] = mats[:, triu_indices[0], triu_indices[1]].imag
    return params.reshape(-1)


def antihermitians_from_parameters(
    params: np.ndarray, dim: int, n_mats: int, real: bool = False
) -> np.ndarray:
    """Construct a batch of antihermitian matrices from parameters.

    Converts a real-valued parameter vector to an array of antihermitian matrices.

    Args:
        params: The 1D real-valued parameters.
        dim: The width and height of each matrix.
        n_mats: The number of matrices in the batch.
        real: Whether the parameter vector describes real-valued antihermitian matrices.

    Returns:
        The array of antihermitian matrices, with shape (n_mats, dim, dim).
    """
    n_params_per_mat = dim * (dim - 1) // 2 if real else dim**2
    params = params.reshape(n_mats, n_params_per_mat)
    mats = np.zeros((n_mats, dim, dim), dtype=float if real else complex)
    n_triu = dim * (dim - 1) // 2
    if not real:
        # imaginary part
        rows, cols = np.triu_indices(dim)
        vals = 1j * params[:, n_triu:]
        mats[:, rows, cols] = vals
        mats[:, cols, rows] = vals
    # real part
    vals = params[:, :n_triu]
    rows, cols = np.triu_indices(dim, k=1)
    mats[:, rows, cols] += vals
    mats[:, cols, rows] -= vals
    return mats


def antihermitians_from_parameters_jax(
    params: np.ndarray, dim: int, n_mats: int, real: bool = False
) -> jax.Array:
    """JAX version of antihermitians_from_parameters."""
    n_params_per_mat = dim * (dim - 1) // 2 if real else dim**2
    params = params.reshape(n_mats, n_params_per_mat)
    mats = jnp.zeros((n_mats, dim, dim), dtype=float if real else complex)
    n_triu = dim * (dim - 1) // 2
    if not real:
        # imaginary part
        rows, cols = jnp.triu_indices(dim)
        vals = 1j * params[:, n_triu:]
        mats = mats.at[:, rows, cols].set(vals)
        mats = mats.at[:, cols, rows].set(vals)
    # real part
    vals = params[:, :n_triu]
    rows, cols = jnp.triu_indices(dim, k=1)
    mats = mats.at[:, rows, cols].add(vals)
    # the subtract method is only available in JAX starting with Python 3.10
    mats = mats.at[:, cols, rows].add(-vals)
    return mats


def unitary_to_parameters(mat: np.ndarray, real: bool = False) -> np.ndarray:
    """Convert a unitary matrix to parameters.

    Converts a unitary to a real-valued parameter vector. The parameter vector
    contains non-redundant real and imaginary parts of the elements of the matrix
    logarithm of the unitary.

    Args:
        mat: The unitary.
        real: Whether to take only the real part of the matrix logarithm of the unitary,
            and discard the imaginary part.

    Returns:
        The list of real numbers parameterizing the unitary.
    """
    return unitaries_to_parameters(mat[None, :], real=real)


def unitary_from_parameters(
    params: np.ndarray, dim: int, real: bool = False
) -> np.ndarray:
    """Construct a unitary matrix from parameters.

    Converts a real-valued parameter vector to a unitary matrix. The parameter vector
    contains non-redundant real and imaginary parts of the elements of the matrix
    logarithm of the unitary matrix.

    Args:
        params: The real-valued parameters.
        dim: The width and height of the unitary matrix.
        real: Whether the parameter vector describes a real-valued unitary matrix.

    Returns:
        The unitary.
    """
    return unitaries_from_parameters(params, dim, 1, real=real)[0]


def unitary_from_parameters_jax(
    params: np.ndarray, dim: int, real: bool = False
) -> jax.Array:
    """JAX version of unitary_from_parameters."""
    return unitaries_from_parameters_jax(params, dim, 1, real=real)[0]


def unitaries_to_parameters(mats: np.ndarray, real: bool = False) -> np.ndarray:
    """Convert a batch of unitary matrices to parameters.

    Converts an array of unitaries to a real-valued parameter vector. The parameter
    vector contains non-redundant real and imaginary parts of the elements of the matrix
    logarithms of the unitaries.

    Args:
        mats: The batch of unitary matrices, with shape (n_mats, dim, dim).
        real: Whether to take only the real part of the matrix logarithm of the unitary,
            and discard the imaginary part.

    Returns:
        The list of real numbers parameterizing the unitaries.
    """
    # TODO in Python 3.11 this becomes
    # return antihermitians_to_parameters(scipy.linalg.logm(mats), real=real)
    return antihermitians_to_parameters(
        np.stack([scipy.linalg.logm(mat) for mat in mats]), real=real
    )


def unitaries_from_parameters(
    params: np.ndarray, dim: int, n_mats: int, real: bool = False
) -> np.ndarray:
    """Construct a batch of unitary matrices from parameters.

    Converts a real-valued parameter vector to an array of unitary matrices.
    The parameter vector contains non-redundant real and imaginary parts of the elements
    of the matrix logarithms of the unitary matrices.

    Args:
        params: The real-valued parameters.
        dim: The width and height of the unitary matrix.
        n_mats: The number of matrices in the batch.
        real: Whether the parameter vector describes a real-valued unitary matrix.

    Returns:
        The array of unitary matrices, with shape (n_mats, dim, dim).
    """
    return scipy.linalg.expm(
        antihermitians_from_parameters(params, dim, n_mats, real=real)
    )


def unitaries_from_parameters_jax(
    params: np.ndarray, dim: int, n_mats: int, real: bool = False
) -> jax.Array:
    """JAX version of unitaries_from_parameters."""
    return jax.scipy.linalg.expm(
        antihermitians_from_parameters_jax(params, dim, n_mats, real=real)
    )


def real_symmetric_to_parameters(
    mat: np.ndarray, triu_indices: list[tuple[int, int]] | None = None
) -> np.ndarray:
    """Convert a real symmetric matrix to parameters.

    Args:
        mat: The real symmetric matrix.
        triu_indices: Upper triangular indices to take values from. If not given,
            the entire upper triangle is taken.

    Returns:
        The list of real numbers parameterizing the real symmetric matrix.
    """
    return real_symmetrics_to_parameters(mat[None, :], triu_indices)


def real_symmetric_from_parameters(
    params: np.ndarray, dim: int, triu_indices: list[tuple[int, int]] | None = None
) -> np.ndarray:
    """Construct a real symmetric matrix from parameters.

    Args:
        params: The real-valued parameters.
        dim: The width and height of the matrix.
        triu_indices: Upper triangular indices to place the parameters. If not given,
            the entire upper triangle is used.

    Returns:
        The real symmetric matrix.
    """
    return real_symmetrics_from_parameters(params, dim, 1, triu_indices)[0]


def real_symmetric_from_parameters_jax(
    params: np.ndarray, dim: int, triu_indices: list[tuple[int, int]] | None = None
) -> jax.Array:
    """JAX version of real_symmetric_from_parameters."""
    return real_symmetrics_from_parameters_jax(params, dim, 1, triu_indices)[0]


def real_symmetrics_to_parameters(
    mats: np.ndarray, triu_indices: list[tuple[int, int]] | None = None
) -> np.ndarray:
    """Convert a batch of real symmetric matrices to parameters.

    Converts an array of real symmetric matrices to a real-valued parameter vector.

    Args:
        mats: The batch of real symmetric matrices, with shape (n_mats, dim, dim).
        triu_indices: Upper triangular indices to take values from. If not given,
            the entire upper triangle is taken.

    Returns:
        The list of real numbers parameterizing the real symmetric matrices.
    """
    n_mats, dim, _ = mats.shape
    if triu_indices is None:
        rows, cols = np.triu_indices(dim)
    else:
        rows, cols = zip(*triu_indices)  # type: ignore
    n_params_per_mat = len(rows)
    params = np.zeros((n_mats, n_params_per_mat))
    params[:, :] = mats[:, rows, cols]
    return params.reshape(-1)


def real_symmetrics_from_parameters(
    params: np.ndarray,
    dim: int,
    n_mats: int,
    triu_indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Construct a batch of real symmetric matrices from parameters.

    Converts a real-valued parameter vector to an array of real symmetric matrices.

    Args:
        params: The 1D real-valued parameters.
        dim: The width and height of each matrix.
        n_mats: The number of matrices in the batch.
        triu_indices: Upper triangular indices to place the parameters. If not given,
            the entire upper triangle is used.

    Returns:
        The array of real symmetric matrices, with shape (n_mats, dim, dim).
    """
    if triu_indices is None:
        rows, cols = np.triu_indices(dim)
        n_params_per_mat = dim * (dim + 1) // 2
    else:
        rows, cols = zip(*triu_indices)  # type: ignore
        n_params_per_mat = len(triu_indices)
    params = params.reshape(n_mats, n_params_per_mat)
    mats = np.zeros((n_mats, dim, dim))
    mats[:, rows, cols] = params
    mats[:, cols, rows] = params
    return mats


def real_symmetrics_from_parameters_jax(
    params: np.ndarray,
    dim: int,
    n_mats: int,
    triu_indices: list[tuple[int, int]] | None = None,
) -> jax.Array:
    """JAX version of real_symmetrics_from_parameters."""
    if triu_indices is None:
        rows, cols = jnp.triu_indices(dim)
        n_params_per_mat = dim * (dim + 1) // 2
    else:
        rows, cols = zip(*triu_indices)  # type: ignore
        n_params_per_mat = len(triu_indices)
    params = params.reshape(n_mats, n_params_per_mat)
    mats = jnp.zeros((n_mats, dim, dim))
    mats = mats.at[:, rows, cols].set(params)
    mats = mats.at[:, cols, rows].set(params)
    return mats


def real_matrices_to_parameters(
    mats: np.ndarray, indices: list[tuple[int, int]] | None = None
) -> np.ndarray:
    """Convert a batch of real matrices to parameters.

    Converts an array of real matrices to a real-valued parameter vector.

    Args:
        mats: The batch of real matrices, with shape (n_mats, dim, dim).
        indices: Indices to take values from. If not given, the entire matrix is taken.

    Returns:
        The list of real numbers parameterizing the real matrices.
    """
    n_mats, dim, _ = mats.shape
    if indices is None:
        rows, cols = zip(*itertools.product(range(dim), repeat=2))
    else:
        rows, cols = zip(*indices)  # type: ignore
    n_params_per_mat = len(rows)
    params = np.zeros((n_mats, n_params_per_mat))
    params[:, :] = mats[:, rows, cols]
    return params.reshape(-1)


def real_matrices_from_parameters(
    params: np.ndarray,
    dim: int,
    n_mats: int,
    indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Construct a batch of real matrices from parameters.

    Converts a real-valued parameter vector to an array of real matrices.

    Args:
        params: The 1D real-valued parameters.
        dim: The width and height of each matrix.
        n_mats: The number of matrices in the batch.
        indices: Indices to place the parameters. If not given, the entire matrix
            is used.

    Returns:
        The array of real matrices, with shape (n_mats, dim, dim).
    """
    if indices is None:
        rows, cols = zip(*itertools.product(range(dim), repeat=2))
        n_params_per_mat = dim**2
    else:
        rows, cols = zip(*indices)  # type: ignore
        n_params_per_mat = len(indices)
    params = params.reshape(n_mats, n_params_per_mat)
    mats = np.zeros((n_mats, dim, dim))
    mats[:, rows, cols] = params
    return mats


def real_matrices_from_parameters_jax(
    params: np.ndarray,
    dim: int,
    n_mats: int,
    indices: list[tuple[int, int]] | None = None,
) -> jax.Array:
    """JAX version of real_matrices_from_parameters."""
    if indices is None:
        rows, cols = zip(*itertools.product(range(dim), repeat=2))
        n_params_per_mat = dim**2
    else:
        rows, cols = zip(*indices)  # type: ignore
        n_params_per_mat = len(indices)
    params = params.reshape(n_mats, n_params_per_mat)
    mats = jnp.zeros((n_mats, dim, dim))
    mats = mats.at[:, rows, cols].set(params)
    return mats


def df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    real: bool = False,
):
    """Convert double factorization tensors to parameters.

    Converts arrays of diagonal Coulomb matrices and orbital rotations to a
    single real-valued parameter vector.

    Args:
        diag_coulomb_mats: The diagonal Coulomb matrices.
        orbital_rotations: The orbital rotations.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices.
        real: Whether to keep only the real parts of the logarithms of the orbital
            rotations, and discard the imaginary parts.

    Returns:
        The list of real numbers parameterizing the double factorization tensors.
    """
    orbital_rotation_params = unitaries_to_parameters(orbital_rotations, real=real)
    diag_coulomb_params = real_symmetrics_to_parameters(
        diag_coulomb_mats, diag_coulomb_indices
    )
    return np.concatenate([orbital_rotation_params, diag_coulomb_params])


def df_tensors_from_params(
    params: np.ndarray,
    n_tensors: int,
    norb: int,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    real: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct double factorization tensors from parameters.

    Converts a real-valued  parameter vector to arrays of diagonal Coulomb matrices and
    orbital rotations

    Args:
        params: The real-valued parameters.
        n_tensors: The number of tensors. This is the number of diagonal Coulomb
            matrices expected, or equivalently, the number of orbital rotations.
        norb: The number of spatial orbitals, which gives the width and height of the
            matrices.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices.
        real: Whether the parameter vector describes real-valued orbital rotations.

    Returns:
        Two arrays. The first contains the diagonal Coulomb matrices, and the
        second contains the orbital rotations.
    """
    n_params_per_orb_rot = norb * (norb - 1) // 2 if real else norb**2
    n_params_orb_rot = n_tensors * n_params_per_orb_rot
    orbital_rotation_params = params[:n_params_orb_rot]
    diag_coulomb_params = params[n_params_orb_rot:]
    orbital_rotations = unitaries_from_parameters(
        orbital_rotation_params, dim=norb, n_mats=n_tensors, real=real
    )
    diag_coulomb_mats = real_symmetrics_from_parameters(
        diag_coulomb_params,
        dim=norb,
        n_mats=n_tensors,
        triu_indices=diag_coulomb_indices,
    )
    return diag_coulomb_mats, orbital_rotations


def df_tensors_from_params_jax(
    params: np.ndarray,
    n_tensors: int,
    norb: int,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    real: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """JAX version of df_tensors_from_params."""
    n_params_per_orb_rot = norb * (norb - 1) // 2 if real else norb**2
    n_params_orb_rot = n_tensors * n_params_per_orb_rot
    orbital_rotation_params = params[:n_params_orb_rot]
    diag_coulomb_params = params[n_params_orb_rot:]
    orbital_rotations = unitaries_from_parameters_jax(
        orbital_rotation_params, dim=norb, n_mats=n_tensors, real=real
    )
    diag_coulomb_mats = real_symmetrics_from_parameters_jax(
        diag_coulomb_params,
        dim=norb,
        n_mats=n_tensors,
        triu_indices=diag_coulomb_indices,
    ).astype(float if real else complex)
    return diag_coulomb_mats, orbital_rotations


def df_tensors_alpha_beta_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    real: bool = False,
):
    """Convert double factorization tensors to parameters.

    Converts arrays of diagonal Coulomb matrices and orbital rotations to a
    single real-valued parameter vector.

    Args:
        diag_coulomb_mats: The diagonal Coulomb matrices.
        orbital_rotations: The orbital rotations.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices.
        real: Whether to keep only the real parts of the logarithms of the orbital
            rotations, and discard the imaginary parts.

    Returns:
        The list of real numbers parameterizing the double factorization tensors.
    """
    _, _, norb, _ = diag_coulomb_mats.shape
    orbital_rotation_params = unitaries_to_parameters(
        orbital_rotations.reshape(-1, norb, norb), real=real
    )
    if diag_coulomb_indices is None:
        pairs_aa, pairs_ab, pairs_bb = None, None, None
    else:
        pairs_aa, pairs_ab, pairs_bb = diag_coulomb_indices
    diag_coulomb_params_aa = real_symmetrics_to_parameters(
        diag_coulomb_mats[:, 0], pairs_aa
    )
    diag_coulomb_params_ab = real_matrices_to_parameters(
        diag_coulomb_mats[:, 1], pairs_ab
    )
    diag_coulomb_params_bb = real_symmetrics_to_parameters(
        diag_coulomb_mats[:, 2], pairs_bb
    )
    return np.concatenate(
        [
            orbital_rotation_params,
            diag_coulomb_params_aa,
            diag_coulomb_params_ab,
            diag_coulomb_params_bb,
        ]
    )


def df_tensors_alpha_beta_from_params(
    params: np.ndarray,
    n_tensors: int,
    norb: int,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    real: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct double factorization tensors from parameters.

    Converts a real-valued  parameter vector to arrays of diagonal Coulomb matrices and
    orbital rotations

    Args:
        params: The real-valued parameters.
        n_tensors: The number of tensors. This is the number of diagonal Coulomb
            matrices expected, or equivalently, the number of orbital rotations.
        norb: The number of spatial orbitals, which gives the width and height of the
            matrices.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices.
        real: Whether the parameter vector describes real-valued orbital rotations.

    Returns:
        Two arrays. The first contains the diagonal Coulomb matrices, and the
        second contains the orbital rotations.
    """
    n_params_per_orb_rot = norb * (norb - 1) // 2 if real else norb**2
    n_params_orb_rot = n_tensors * 2 * n_params_per_orb_rot

    if diag_coulomb_indices is None:
        pairs_aa, pairs_ab, pairs_bb = None, None, None
    else:
        pairs_aa, pairs_ab, pairs_bb = diag_coulomb_indices
    n_params_aa = n_tensors * (
        norb * (norb + 1) // 2 if pairs_aa is None else len(pairs_aa)
    )
    n_params_ab = n_tensors * (norb**2 if pairs_ab is None else len(pairs_ab))

    (
        orbital_rotation_params,
        diag_coulomb_params_aa,
        diag_coulomb_params_ab,
        diag_coulomb_params_bb,
    ) = np.split(
        params,
        [
            n_params_orb_rot,
            n_params_orb_rot + n_params_aa,
            n_params_orb_rot + n_params_aa + n_params_ab,
        ],
    )

    orbital_rotations = unitaries_from_parameters(
        orbital_rotation_params, dim=norb, n_mats=n_tensors * 2, real=real
    ).reshape(n_tensors, 2, norb, norb)
    diag_coulomb_mats_aa = real_symmetrics_from_parameters(
        diag_coulomb_params_aa,
        dim=norb,
        n_mats=n_tensors,
        triu_indices=pairs_aa,
    )
    diag_coulomb_mats_ab = real_matrices_from_parameters(
        diag_coulomb_params_ab,
        dim=norb,
        n_mats=n_tensors,
        indices=pairs_ab,
    )
    diag_coulomb_mats_bb = real_symmetrics_from_parameters(
        diag_coulomb_params_bb,
        dim=norb,
        n_mats=n_tensors,
        triu_indices=pairs_bb,
    )

    diag_coulomb_mats = np.stack(
        [diag_coulomb_mats_aa, diag_coulomb_mats_ab, diag_coulomb_mats_bb], axis=1
    )
    return diag_coulomb_mats, orbital_rotations


def df_tensors_alpha_beta_from_params_jax(
    params: np.ndarray,
    n_tensors: int,
    norb: int,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    real: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """JAX version of df_tensors_alpha_beta_from_params."""
    n_params_per_orb_rot = norb * (norb - 1) // 2 if real else norb**2
    n_params_orb_rot = n_tensors * 2 * n_params_per_orb_rot

    if diag_coulomb_indices is None:
        pairs_aa, pairs_ab, pairs_bb = None, None, None
    else:
        pairs_aa, pairs_ab, pairs_bb = diag_coulomb_indices
    n_params_aa = n_tensors * (
        norb * (norb + 1) // 2 if pairs_aa is None else len(pairs_aa)
    )
    n_params_ab = n_tensors * (norb**2 if pairs_ab is None else len(pairs_ab))

    (
        orbital_rotation_params,
        diag_coulomb_params_aa,
        diag_coulomb_params_ab,
        diag_coulomb_params_bb,
    ) = np.split(
        params,
        [
            n_params_orb_rot,
            n_params_orb_rot + n_params_aa,
            n_params_orb_rot + n_params_aa + n_params_ab,
        ],
    )

    orbital_rotations = unitaries_from_parameters_jax(
        orbital_rotation_params, dim=norb, n_mats=n_tensors * 2, real=real
    ).reshape(n_tensors, 2, norb, norb)
    diag_coulomb_mats_aa = real_symmetrics_from_parameters_jax(
        diag_coulomb_params_aa,
        dim=norb,
        n_mats=n_tensors,
        triu_indices=pairs_aa,
    )
    diag_coulomb_mats_ab = real_matrices_from_parameters_jax(
        diag_coulomb_params_ab,
        dim=norb,
        n_mats=n_tensors,
        indices=pairs_ab,
    )
    diag_coulomb_mats_bb = real_symmetrics_from_parameters_jax(
        diag_coulomb_params_bb,
        dim=norb,
        n_mats=n_tensors,
        triu_indices=pairs_bb,
    )

    diag_coulomb_mats = jnp.stack(
        [diag_coulomb_mats_aa, diag_coulomb_mats_ab, diag_coulomb_mats_bb], axis=1
    )
    return diag_coulomb_mats, orbital_rotations
