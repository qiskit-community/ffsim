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


def _df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_mat_mask: np.ndarray,
):
    _, norb, _ = orbital_rotations.shape
    orb_rot_logs = [scipy.linalg.logm(mat) for mat in orbital_rotations]
    # include the diagonal element
    orb_rot_param_real_indices = np.triu_indices(norb, k=1)
    orb_rot_params_real = np.real(
        np.ravel(
            [orb_rot_log[orb_rot_param_real_indices] for orb_rot_log in orb_rot_logs]
        )
    )
    # add imag part
    orb_rot_param_imag_indices = np.triu_indices(norb)
    orb_rot_params_imag = np.imag(
        np.ravel(
            [orb_rot_log[orb_rot_param_imag_indices] for orb_rot_log in orb_rot_logs]
        )
    )
    diag_coulomb_param_indices = np.nonzero(diag_coulomb_mat_mask)
    diag_coulomb_params = np.ravel(
        [
            diag_coulomb_mat[diag_coulomb_param_indices]
            for diag_coulomb_mat in diag_coulomb_mats
        ]
    )
    return np.concatenate(
        [orb_rot_params_real, orb_rot_params_imag, diag_coulomb_params]
    )


def _params_to_orb_rot_logs(params: np.ndarray, n_tensors: int, norb: int):
    orb_rot_imag_logs = np.zeros((n_tensors, norb, norb), dtype="complex")
    orb_rot_logs = np.zeros((n_tensors, norb, norb), dtype="complex")
    # reconstruct the real part
    triu_indices = np.triu_indices(norb, k=1)
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        orb_rot_logs[i][triu_indices] = params[
            i * param_length : (i + 1) * param_length
        ]
        orb_rot_logs[i] -= orb_rot_logs[i].T
    # reconstruct the imag part
    triu_indices = np.triu_indices(norb)
    real_begin_index = param_length * n_tensors
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        orb_rot_imag_logs[i][triu_indices] = (
            1j
            * params[
                i * param_length + real_begin_index : (i + 1) * param_length
                + real_begin_index
            ]
        )
        orb_rot_imag_logs_transpose = orb_rot_imag_logs[i].T
        # keep the diagonal element
        diagonal_element = np.diag(np.diag(orb_rot_imag_logs_transpose))
        orb_rot_imag_logs[i] += orb_rot_imag_logs_transpose
        orb_rot_imag_logs[i] -= diagonal_element
    orb_rot_logs += orb_rot_imag_logs
    return orb_rot_logs


def _expm_antihermitian(mats: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mats)
    return np.einsum("tij,tj,tkj->tik", vecs, np.exp(1j * eigs), vecs.conj())


def _params_to_df_tensors(
    params: np.ndarray, n_tensors: int, norb: int, diag_coulomb_mat_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    orb_rot_logs = _params_to_orb_rot_logs(params, n_tensors, norb)
    orbital_rotations = _expm_antihermitian(orb_rot_logs)
    n_orb_rot_params = n_tensors * norb**2
    diag_coulomb_params = np.real(params[n_orb_rot_params:])
    param_indices = np.nonzero(diag_coulomb_mat_mask)
    param_length = len(param_indices[0])
    diag_coulomb_mats = np.zeros((n_tensors, norb, norb))
    for i in range(n_tensors):
        diag_coulomb_mats[i][param_indices] = diag_coulomb_params[
            i * param_length : (i + 1) * param_length
        ]
        diag_coulomb_mats[i] += diag_coulomb_mats[i].T
        diag_coulomb_mats[i][range(norb), range(norb)] /= 2
    return diag_coulomb_mats, orbital_rotations


def _params_to_df_tensors_jax(
    params: np.ndarray, n_tensors: int, norb: int, diag_coulomb_mat_mask: np.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # convert orb_rot_logs
    triu_indices_real = jnp.triu_indices(norb, k=1)
    param_length_real = len(triu_indices_real[0])
    triu_indices_imag = jnp.triu_indices(norb)
    real_begin_index = param_length_real * n_tensors
    param_length_imag = len(triu_indices_imag[0])
    list_orb_rot_log = []
    for i in range(n_tensors):
        # reconstruct the real part
        orb_rot_log_real = jnp.zeros((norb, norb), dtype="complex")
        orb_rot_log_real = orb_rot_log_real.at[triu_indices_real].set(
            params[i * param_length_real : (i + 1) * param_length_real]
        )
        orb_rot_log_real = orb_rot_log_real - orb_rot_log_real.T
        # reconstruct the imag part
        orb_rot_imag_log = jnp.zeros((norb, norb), dtype="complex")
        orb_rot_imag_log = orb_rot_imag_log.at[triu_indices_imag].set(
            1j
            * params[
                i * param_length_imag + real_begin_index : (i + 1) * param_length_imag
                + real_begin_index
            ]
        )
        orb_rot_imag_log_transpose = orb_rot_imag_log.T
        # keep the diagonal element
        diagonal_element = jnp.diag(jnp.diag(orb_rot_imag_log_transpose))
        orb_rot_imag_log = (
            orb_rot_imag_log + orb_rot_imag_log_transpose - diagonal_element
        )
        orb_rot_log = orb_rot_log_real + orb_rot_imag_log
        list_orb_rot_log.append(orb_rot_log)
    orb_rot_logs = jnp.stack(
        list_orb_rot_log,
        axis=0,
    )
    eigs, vecs = jnp.linalg.eigh(-1j * orb_rot_logs)
    orbital_rotations = jnp.einsum(
        "tij,tj,tkj->tik", vecs, jnp.exp(1j * eigs), vecs.conj()
    )

    # convert diag coulomb mats
    n_orb_rot_params = n_tensors * (norb**2)
    param_indices = np.nonzero(diag_coulomb_mat_mask)
    param_length = len(param_indices[0])
    list_diag_coulomb_mats = []
    for i in range(n_tensors):
        diag_coulomb_mat = jnp.zeros((norb, norb), complex)
        diag_coulomb_mat = diag_coulomb_mat.at[param_indices].set(
            params[
                n_orb_rot_params + i * param_length : n_orb_rot_params
                + (i + 1) * param_length
            ]
        )
        list_diag_coulomb_mats.append(diag_coulomb_mat)
    diagonal_element = jnp.stack(
        [
            jnp.diag(jnp.diag(diag_coulomb_mat))
            for diag_coulomb_mat in list_diag_coulomb_mats
        ],
        axis=0,
    )

    diag_coulomb_mats_tri = jnp.stack(list_diag_coulomb_mats, axis=0)
    diag_coulomb_mats = (
        diag_coulomb_mats_tri
        + jnp.transpose(diag_coulomb_mats_tri, (0, 2, 1))
        - diagonal_element
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

    if not diag_coulomb_indices:
        diag_coulomb_mask = np.ones((norb, norb), dtype=bool)
    else:
        diag_coulomb_mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*diag_coulomb_indices)
        diag_coulomb_mask[rows, cols] = True
        diag_coulomb_mask[cols, rows] = True
    diag_coulomb_mask = np.triu(diag_coulomb_mask)

    for n_tensors in list_reps:
        diag_coulomb_mats = diag_coulomb_mats[:n_tensors]
        orbital_rotations = orbital_rotations[:n_tensors]

        def fun_jax(x):
            diag_coulomb_mats, orbital_rotations = _params_to_df_tensors_jax(
                x, n_tensors, norb, diag_coulomb_mask
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
            diag_coulomb_mats, orbital_rotations, diag_coulomb_mask
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
            result.x, n_tensors, norb, diag_coulomb_mask
        )

    if return_optimize_result:
        return diag_coulomb_mats, orbital_rotations, result

    return diag_coulomb_mats, orbital_rotations
