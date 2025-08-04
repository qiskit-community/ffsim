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
    norb, _ = mat.shape
    triu_indices = np.triu_indices(norb, k=1)
    n_triu = norb * (norb - 1) // 2
    params = np.zeros(n_triu if real else norb**2)
    # real part
    params[:n_triu] = mat[triu_indices].real
    # imaginary part
    if not real:
        triu_indices = np.triu_indices(norb)
        params[n_triu:] = mat[triu_indices].imag
    return params


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
    mat = np.zeros((dim, dim), dtype=float if real else complex)
    n_triu = dim * (dim - 1) // 2
    if not real:
        # imaginary part
        rows, cols = np.triu_indices(dim)
        vals = 1j * params[n_triu:]
        mat[rows, cols] = vals
        mat[cols, rows] = vals
    # real part
    vals = params[:n_triu]
    rows, cols = np.triu_indices(dim, k=1)
    mat[rows, cols] += vals
    mat[cols, rows] -= vals
    return mat


def antihermitian_from_parameters_jax(
    params: np.ndarray, dim: int, real: bool = False
) -> jax.Array:
    """JAX version of antihermitian_from_parameters."""
    mat = jnp.zeros((dim, dim), dtype=float if real else complex)
    n_triu = dim * (dim - 1) // 2
    if not real:
        # imaginary part
        rows, cols = jnp.triu_indices(dim)
        vals = 1j * params[n_triu:]
        mat = mat.at[rows, cols].set(vals)
        mat = mat.at[cols, rows].set(vals)
    # real part
    vals = params[:n_triu]
    rows, cols = jnp.triu_indices(dim, k=1)
    mat = mat.at[rows, cols].add(vals)
    # the subtract method is only available in JAX starting with Python 3.10
    mat = mat.at[cols, rows].add(-vals)
    return mat


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
    return antihermitian_to_parameters(scipy.linalg.logm(mat), real=real)


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
    return scipy.linalg.expm(antihermitian_from_parameters(params, dim, real=real))


def unitary_from_parameters_jax(
    params: np.ndarray, dim: int, real: bool = False
) -> jax.Array:
    """JAX version of unitary_from_parameters."""
    return jax.scipy.linalg.expm(antihermitian_from_parameters_jax(params, dim, real))


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
    return antihermitians_to_parameters(scipy.linalg.logm(mats), real=real)


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
    if triu_indices is None:
        norb, _ = mat.shape
        rows, cols = np.triu_indices(norb)
    else:
        rows, cols = zip(*triu_indices)  # type: ignore
    return mat[rows, cols]


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
    if triu_indices is None:
        rows, cols = np.triu_indices(dim)
    else:
        rows, cols = zip(*triu_indices)  # type: ignore
    mat = np.zeros((dim, dim))
    mat[rows, cols] = params
    mat[cols, rows] = params
    return mat


def real_symmetric_from_parameters_jax(
    params: np.ndarray, dim: int, triu_indices: list[tuple[int, int]] | None = None
) -> jax.Array:
    """JAX version of real_symmetric_from_parameters."""
    if triu_indices is None:
        rows, cols = jnp.triu_indices(dim)
    else:
        rows, cols = zip(*triu_indices)  # type: ignore
    mat = jnp.zeros((dim, dim))
    mat = mat.at[rows, cols].set(params)
    mat = mat.at[cols, rows].set(params)
    return mat


def df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    real: bool = False,
):
    """Convert double-factorization tensors to parameters.

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
        The list of real numbers parameterizing the double-factorization tensors.
    """
    orbital_rotation_params = np.concatenate(
        [
            unitary_to_parameters(orbital_rotation, real=real)
            for orbital_rotation in orbital_rotations
        ]
    )
    diag_coulomb_params = np.concatenate(
        [
            real_symmetric_to_parameters(mat, diag_coulomb_indices)
            for mat in diag_coulomb_mats
        ]
    )
    return np.concatenate([orbital_rotation_params, diag_coulomb_params])


def df_tensors_from_params(
    params: np.ndarray,
    n_tensors: int,
    norb: int,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    real: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct double-factorization tensors from parameters.

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
    if diag_coulomb_indices is None:
        n_params_per_diag_coulomb = norb * (norb + 1) // 2
    else:
        n_params_per_diag_coulomb = len(diag_coulomb_indices)

    n_params_orb_rot = n_tensors * n_params_per_orb_rot
    orbital_rotation_params = params[:n_params_orb_rot]
    diag_coulomb_params = params[n_params_orb_rot:]

    orbital_rotations = np.zeros(
        (n_tensors, norb, norb), dtype=float if real else complex
    )
    diag_coulomb_mats = np.zeros((n_tensors, norb, norb))
    for i in range(n_tensors):
        orbital_rotations[i] = unitary_from_parameters(
            orbital_rotation_params[
                i * n_params_per_orb_rot : (i + 1) * n_params_per_orb_rot
            ],
            norb,
            real=real,
        )
        diag_coulomb_mats[i] = real_symmetric_from_parameters(
            diag_coulomb_params[
                i * n_params_per_diag_coulomb : (i + 1) * n_params_per_diag_coulomb
            ],
            norb,
            diag_coulomb_indices,
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
    if diag_coulomb_indices is None:
        n_params_per_diag_coulomb = norb * (norb + 1) // 2
    else:
        n_params_per_diag_coulomb = len(diag_coulomb_indices)

    n_params_orb_rot = n_tensors * n_params_per_orb_rot
    orbital_rotation_params = params[:n_params_orb_rot]
    diag_coulomb_params = params[n_params_orb_rot:]

    orbital_rotations = jnp.zeros(
        (n_tensors, norb, norb), dtype=float if real else complex
    )
    diag_coulomb_mats = jnp.zeros(
        (n_tensors, norb, norb), dtype=float if real else complex
    )

    for i in range(n_tensors):
        orbital_rotations = orbital_rotations.at[i].set(
            unitary_from_parameters_jax(
                orbital_rotation_params[
                    i * n_params_per_orb_rot : (i + 1) * n_params_per_orb_rot
                ],
                norb,
                real=real,
            )
        )
        diag_coulomb_mats = diag_coulomb_mats.at[i].set(
            real_symmetric_from_parameters_jax(
                diag_coulomb_params[
                    i * n_params_per_diag_coulomb : (i + 1) * n_params_per_diag_coulomb
                ],
                norb,
                diag_coulomb_indices,
            )
        )

    return diag_coulomb_mats, orbital_rotations
