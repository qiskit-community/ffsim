# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for performing the compressed double-factorized decomposition."""

from __future__ import annotations

from typing import Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from opt_einsum import contract

from ffsim.linalg import double_factorized_t2


def orbital_rotation_to_parameters(
    orbital_rotation: np.ndarray, real: bool = False
) -> np.ndarray:
    """Convert an orbital rotation to parameters.

    Converts an orbital rotation to a real-valued parameter vector. The parameter vector
    contains non-redundant real and imaginary parts of the elements of the matrix
    logarithm of the orbital rotation matrix.

    Args:
        orbital_rotation: The orbital rotation.
        real: Whether to construct a parameter vector for a real-valued
            orbital rotation. If True, the orbital rotation must have a real-valued
            data type.

    Returns:
        The list of real numbers parameterizing the orbital rotation.
    """
    if real and np.iscomplexobj(orbital_rotation):
        raise TypeError(
            "real was set to True, but the orbital rotation has a complex data type. "
            "Try passing an orbital rotation with a real-valued data type, or else "
            "set real=False."
        )
    norb, _ = orbital_rotation.shape
    triu_indices = np.triu_indices(norb, k=1)
    n_triu = norb * (norb - 1) // 2
    mat = scipy.linalg.logm(orbital_rotation)
    params = np.zeros(n_triu if real else norb**2)
    # real part
    params[:n_triu] = mat[triu_indices].real
    # imaginary part
    if not real:
        triu_indices = np.triu_indices(norb)
        params[n_triu:] = mat[triu_indices].imag
    return params


def orbital_rotation_from_parameters(
    params: np.ndarray, norb: int, real: bool = False
) -> np.ndarray:
    """Construct an orbital rotation from parameters.

    Converts a real-valued parameter vector to an orbital rotation. The parameter vector
    contains non-redundant real and imaginary parts of the elements of the matrix
    logarithm of the orbital rotation matrix.

    Args:
        params: The real-valued parameters.
        norb: The number of spatial orbitals, which gives the width and height of the
            orbital rotation matrix.
        real: Whether the parameter vector describes a real-valued orbital rotation.

    Returns:
        The orbital rotation.
    """
    generator = np.zeros((norb, norb), dtype=float if real else complex)
    n_triu = norb * (norb - 1) // 2
    if not real:
        # imaginary part
        rows, cols = np.triu_indices(norb)
        vals = 1j * params[n_triu:]
        generator[rows, cols] = vals
        generator[cols, rows] = vals
    # real part
    vals = params[:n_triu]
    rows, cols = np.triu_indices(norb, k=1)
    generator[rows, cols] += vals
    generator[cols, rows] -= vals
    return scipy.linalg.expm(generator)


def diag_coulomb_mat_to_parameters(
    mat: np.ndarray, diag_coulomb_indices: list[tuple[int, int]] | None = None
) -> np.ndarray:
    """Convert a diagonal Coulomb matrix to parameters.

    Args:
        mat: The diagonal Coulomb matrix.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrix.

    Returns:
        The list of real numbers parameterizing the diagonal Coulomb matrix.
    """
    if diag_coulomb_indices is None:
        norb, _ = mat.shape
        rows, cols = np.triu_indices(norb)
    else:
        rows, cols = zip(*diag_coulomb_indices)
    return mat[rows, cols]


def diag_coulomb_mat_from_parameters(
    params: np.ndarray,
    norb: int,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Construct a diagonal Coulomb matrix from parameters.

    Args:
        params: The real-valued parameters.
        norb: The number of spatial orbitals, which gives the width and height of the
            diagonal Coulomb matrix.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrix.

    Returns:
        The diagonal Coulomb matrix.
    """
    if diag_coulomb_indices is None:
        rows, cols = np.triu_indices(norb)
    else:
        rows, cols = zip(*diag_coulomb_indices)
    mat = np.zeros((norb, norb))
    mat[rows, cols] = params
    mat[cols, rows] = params
    return mat


def _df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_indices: list[tuple[int, int]] | None,
):
    orbital_rotation_params = np.concatenate(
        [
            orbital_rotation_to_parameters(orbital_rotation)
            for orbital_rotation in orbital_rotations
        ]
    )
    diag_coulomb_params = np.concatenate(
        [
            diag_coulomb_mat_to_parameters(mat, diag_coulomb_indices)
            for mat in diag_coulomb_mats
        ]
    )
    return np.concatenate([orbital_rotation_params, diag_coulomb_params])


def _params_to_df_tensors(
    params: np.ndarray,
    n_tensors: int,
    norb: int,
    diag_coulomb_indices: list[tuple[int, int]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_params_per_orb_rot = norb**2
    if diag_coulomb_indices is None:
        n_params_per_diag_coulomb = norb * (norb + 1) // 2
    else:
        n_params_per_diag_coulomb = len(diag_coulomb_indices)

    n_params_orb_rot = n_tensors * n_params_per_orb_rot
    orbital_rotation_params = params[:n_params_orb_rot]
    diag_coulomb_params = params[n_params_orb_rot:]

    orbital_rotations = np.zeros((n_tensors, norb, norb), dtype=complex)
    diag_coulomb_mats = np.zeros((n_tensors, norb, norb))
    for i in range(n_tensors):
        orbital_rotations[i] = orbital_rotation_from_parameters(
            orbital_rotation_params[
                i * n_params_per_orb_rot : (i + 1) * n_params_per_orb_rot
            ],
            norb,
            real=False,
        )
        diag_coulomb_mats[i] = diag_coulomb_mat_from_parameters(
            diag_coulomb_params[
                i * n_params_per_diag_coulomb : (i + 1) * n_params_per_diag_coulomb
            ],
            norb,
            diag_coulomb_indices,
        )

    return diag_coulomb_mats, orbital_rotations


def _orbital_rotation_from_parameters_jax(
    params: np.ndarray, norb: int, real: bool = False
) -> jax.Array:
    """Construct an orbital rotation from parameters.

    Converts a real-valued parameter vector to an orbital rotation. The parameter vector
    contains non-redundant real and imaginary parts of the elements of the matrix
    logarithm of the orbital rotation matrix.

    Args:
        params: The real-valued parameters.
        norb: The number of spatial orbitals, which gives the width and height of the
            orbital rotation matrix.
        real: Whether the parameter vector describes a real-valued orbital rotation.

    Returns:
        The orbital rotation.
    """
    generator = jnp.zeros((norb, norb), dtype=float if real else complex)
    n_triu = norb * (norb - 1) // 2
    if not real:
        # imaginary part
        rows, cols = jnp.triu_indices(norb)
        vals = 1j * params[n_triu:]
        generator = generator.at[rows, cols].set(vals)
        generator = generator.at[cols, rows].set(vals)
    # real part
    vals = params[:n_triu]
    rows, cols = jnp.triu_indices(norb, k=1)
    generator = generator.at[rows, cols].add(vals)
    # the subtract method is only available in JAX starting with Python 3.10
    generator = generator.at[cols, rows].add(-vals)
    return jax.scipy.linalg.expm(generator)


def _diag_coulomb_mat_from_parameters_jax(
    params: np.ndarray,
    norb: int,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
) -> jax.Array:
    if diag_coulomb_indices is None:
        rows, cols = jnp.triu_indices(norb)
    else:
        rows, cols = zip(*diag_coulomb_indices)
    mat = jnp.zeros((norb, norb))
    mat = mat.at[rows, cols].set(params)
    mat = mat.at[cols, rows].set(params)
    return mat


def _params_to_df_tensors_jax(
    params: np.ndarray,
    n_tensors: int,
    norb: int,
    diag_coulomb_indices: list[tuple[int, int]] | None,
) -> tuple[jax.Array, jax.Array]:
    n_params_per_orb_rot = norb**2
    if diag_coulomb_indices is None:
        n_params_per_diag_coulomb = norb * (norb + 1) // 2
    else:
        n_params_per_diag_coulomb = len(diag_coulomb_indices)

    n_params_orb_rot = n_tensors * n_params_per_orb_rot
    orbital_rotation_params = params[:n_params_orb_rot]
    diag_coulomb_params = params[n_params_orb_rot:]

    orbital_rotations = jnp.zeros((n_tensors, norb, norb), dtype=complex)
    diag_coulomb_mats = jnp.zeros((n_tensors, norb, norb), dtype=complex)

    for i in range(n_tensors):
        orbital_rotations = orbital_rotations.at[i].set(
            _orbital_rotation_from_parameters_jax(
                orbital_rotation_params[
                    i * n_params_per_orb_rot : (i + 1) * n_params_per_orb_rot
                ],
                norb,
                real=False,
            )
        )
        diag_coulomb_mats = diag_coulomb_mats.at[i].set(
            _diag_coulomb_mat_from_parameters_jax(
                diag_coulomb_params[
                    i * n_params_per_diag_coulomb : (i + 1) * n_params_per_diag_coulomb
                ],
                norb,
                diag_coulomb_indices,
            )
        )

    return diag_coulomb_mats, orbital_rotations


@overload
def double_factorized_t2_compressed(
    t2: np.ndarray,
    *,
    tol: float = ...,
    n_reps: int | None = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    method: str = ...,
    callback=...,
    options: dict | None = ...,
    multi_stage_optimization: bool = ...,
    begin_reps: int | None = ...,
    step: int = ...,
    return_optimize_result: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def double_factorized_t2_compressed(
    t2: np.ndarray,
    *,
    tol: float = ...,
    n_reps: int | None = ...,
    diag_coulomb_indices: list[tuple[int, int]] | None = ...,
    method: str = ...,
    callback=...,
    options: dict | None = ...,
    multi_stage_optimization: bool = ...,
    begin_reps: int | None = ...,
    step: int = ...,
    return_optimize_result: Literal[True],
) -> tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]: ...
def double_factorized_t2_compressed(
    t2: np.ndarray,
    *,
    tol: float = 1e-8,
    n_reps: int | None = None,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    multi_stage_optimization: bool = False,
    begin_reps: int | None = None,
    step: int = 2,
    return_optimize_result: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, scipy.optimize.OptimizeResult]
):
    r"""Compressed double-factorized decomposition of t2 amplitudes for LUCJ ansatz.

    The double-factorized decomposition of a t2 amplitudes tensor :math:`t_{ijab}` is

    .. math::

        t_{ijab} = i \sum_{m=1}^L \sum_{k=1}^2 \sum_{pq}
            Z^{(mk)}_{pq}
            U^{(mk)}_{ap} U^{(mk)*}_{ip} U^{(mk)}_{bq} U^{(mk)*}_{jq}

    Here each :math:`Z^{(mk)}` is a real-valued matrix, referred to as a
    "diagonal Coulomb matrix," and each :math:`U^{(mk)}` is a unitary matrix,
    referred to as an "orbital rotation."

    The number of terms :math:`L` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    After decomposition, the goal is to compress the operator down to `n_reps` terms
    while minimizing the difference with the original t2 amplitude with a least-squares
    objective function. This is achieved by first truncating the operator and then
    apply optimizer to minimize the coefficients in the remaining operator.

    Note: Currently, only real-valued t2 amplitudes are supported.

    Args:
        t2: The t2 amplitudes tensor.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        n_reps: The number of ansatz repetitions. If not specified, then it is set
            to the number of terms resulting from the double-factorization of the
            t2 amplitudes. If the specified number of repetitions is larger than the
            number of terms resulting from the double-factorization, then the ansatz
            is padded with additional identity operators up to the specified number
            of repetitions.
        diag_coulomb_indices: Allowed indices for nonzero values of the diagonal
            Coulomb matrices. Matrix entries corresponding to indices not in this
            list will be set to zero. This list should contain only upper
            trianglular indices, i.e., pairs :math:`(i, j)` where :math:`i \leq j`.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        multi_stage_optimization: Iteratively reduce the number of ansatz repetitions
            starting from full configuration if  `begin_reps` is not given. In each
            iteration, the number of repetitions is reduced by `step` until reaching
            `n_reps`.
        begin_reps: The starting point of the multi-stage optimization
        step: The step size for the multi-stage optimization
        return_optimize_result: Whether to also return the `OptimizeResult`_ returned
            by `scipy.optimize.minimize`_.

    Returns:
        - The diagonal Coulomb matrices, as a Numpy array of shape
          `(n_reps, norb, norb)`.
          The first axis indexes the eigenvectors of the decomposition, and he last two
          axes index the rows and columns of the matrices.
        - The orbital rotations, as a Numpy array of shape
          `(n_reps, norb, norb)`.
          The first axis indexes the eigenvectors of the decomposition, and he last two
          axes index the rows and columns of the matrices.
        - If `return_optimize_result` is set to ``True``, the `OptimizeResult`_
          returned by `scipy.optimize.minimize`_ is also returned.

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    nocc, _, nvrt, _ = t2.shape
    norb = nocc + nvrt

    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2, tol=tol)
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)
    n_reps_full, _, _ = orbital_rotations.shape

    if n_reps is None or n_reps_full < n_reps:
        return diag_coulomb_mats, orbital_rotations

    if multi_stage_optimization:
        if begin_reps is None:
            begin_reps = n_reps_full
        begin_reps = min(n_reps_full, begin_reps)
        list_reps = list(range(begin_reps, n_reps, -step))
        list_reps.append(n_reps)
    else:
        list_reps = [n_reps]

    for n_tensors in list_reps:
        diag_coulomb_mats = diag_coulomb_mats[:n_tensors]
        orbital_rotations = orbital_rotations[:n_tensors]

        def fun_jax(x):
            diag_coulomb_mats, orbital_rotations = _params_to_df_tensors_jax(
                x, n_tensors, norb, diag_coulomb_indices
            )
            reconstructed = (
                1j
                * contract(
                    "mpq,map,mip,mbq,mjq->ijab",
                    diag_coulomb_mats,
                    orbital_rotations,
                    orbital_rotations.conj(),
                    orbital_rotations,
                    orbital_rotations.conj(),
                    # optimize="greedy"
                )[:nocc, :nocc, nocc:, nocc:]
            )
            diff = reconstructed - t2
            return 0.5 * jnp.sum(jnp.abs(diff) ** 2)

        value_and_grad_func = jax.value_and_grad(fun_jax)

        x0 = _df_tensors_to_params(
            diag_coulomb_mats, orbital_rotations, diag_coulomb_indices
        )

        result = scipy.optimize.minimize(
            value_and_grad_func,
            x0,
            method=method,
            jac=True,
            callback=callback,
            options=options,
        )

        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            result.x, n_tensors, norb, diag_coulomb_indices
        )

    if return_optimize_result:
        return diag_coulomb_mats, orbital_rotations, result

    return diag_coulomb_mats, orbital_rotations
