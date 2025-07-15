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

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from opt_einsum import contract

from ffsim.linalg import double_factorized_t2


def _reshape_grad(
    core_coulomb_params: jnp.ndarray,
    orbital_rotations_log_jax_tri: jnp.ndarray,
):
    _, norb, _ = orbital_rotations_log_jax_tri.shape
    # include the diagonal element
    leaf_param_real_indices = np.triu_indices(norb, k=1)

    leaf_params_real = np.real(
        np.ravel(
            [
                orbital_rotation[leaf_param_real_indices]
                for orbital_rotation in orbital_rotations_log_jax_tri
            ]
        )
    )
    # add imag part
    leaf_param_imag_indices = np.triu_indices(norb)
    leaf_params_imag = -np.imag(
        np.ravel(
            [
                orbital_rotation[leaf_param_imag_indices]
                for orbital_rotation in orbital_rotations_log_jax_tri
            ]
        )
    )
    core_coulomb_params = np.real(core_coulomb_params)
    return np.concatenate([leaf_params_real, leaf_params_imag, core_coulomb_params])


def _df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_mat_mask: np.ndarray,
):
    _, norb, _ = orbital_rotations.shape
    leaf_logs = [scipy.linalg.logm(mat) for mat in orbital_rotations]
    # include the diagonal element
    leaf_param_real_indices = np.triu_indices(norb, k=1)
    leaf_params_real = np.real(
        np.ravel([leaf_log[leaf_param_real_indices] for leaf_log in leaf_logs])
    )
    # add imag part
    leaf_param_imag_indices = np.triu_indices(norb)
    leaf_params_imag = np.imag(
        np.ravel([leaf_log[leaf_param_imag_indices] for leaf_log in leaf_logs])
    )
    core_param_indices = np.nonzero(diag_coulomb_mat_mask)
    core_params = np.ravel(
        [diag_coulomb_mat[core_param_indices] for diag_coulomb_mat in diag_coulomb_mats]
    )
    return np.concatenate([leaf_params_real, leaf_params_imag, core_params])


def _params_to_leaf_logs(params: np.ndarray, n_tensors: int, norb: int):
    leaf_imag_logs = np.zeros((n_tensors, norb, norb), dtype="complex")
    leaf_logs = np.zeros((n_tensors, norb, norb), dtype="complex")
    # reconstruct the real part
    triu_indices = np.triu_indices(norb, k=1)
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_logs[i][triu_indices] = params[i * param_length : (i + 1) * param_length]
        leaf_logs[i] -= leaf_logs[i].T
    # reconstruct the imag part
    triu_indices = np.triu_indices(norb)
    real_begin_index = param_length * n_tensors
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_imag_logs[i][triu_indices] = (
            1j
            * params[
                i * param_length + real_begin_index : (i + 1) * param_length
                + real_begin_index
            ]
        )
        leaf_imag_logs_transpose = leaf_imag_logs[i].T
        # keep the diagonal element
        diagonal_element = np.diag(np.diag(leaf_imag_logs_transpose))
        leaf_imag_logs[i] += leaf_imag_logs_transpose
        leaf_imag_logs[i] -= diagonal_element
    leaf_logs += leaf_imag_logs
    return leaf_logs


def _expm_antihermitian(mats: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mats)
    return np.einsum("tij,tj,tkj->tik", vecs, np.exp(1j * eigs), vecs.conj())


def _params_to_df_tensors(
    params: np.ndarray, n_tensors: int, norb: int, diag_coulomb_mat_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    leaf_logs = _params_to_leaf_logs(params, n_tensors, norb)
    orbital_rotations = _expm_antihermitian(leaf_logs)
    n_leaf_params = n_tensors * (norb * (norb - 1) // 2 + norb * (norb + 1) // 2)
    core_params = np.real(params[n_leaf_params:])
    param_indices = np.nonzero(diag_coulomb_mat_mask)
    param_length = len(param_indices[0])
    diag_coulomb_mats = np.zeros((n_tensors, norb, norb))
    for i in range(n_tensors):
        diag_coulomb_mats[i][param_indices] = core_params[
            i * param_length : (i + 1) * param_length
        ]
        diag_coulomb_mats[i] += diag_coulomb_mats[i].T
        diag_coulomb_mats[i][range(norb), range(norb)] /= 2
    return diag_coulomb_mats, orbital_rotations


def double_factorized_t2_compress(
    t2: np.ndarray,
    *,
    tol: float = 1e-8,
    n_reps: int | None = None,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    multi_stage_optimization: bool = True,
    begin_reps: int | None = None,
    step: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
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
    Furthermore, the `n_reps` parameter specifies an optional upper bound
    on :math:`L`. The `n_reps` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

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
        interaction_pairs: Optional restrictions on allowed orbital interactions
            for the diagonal Coulomb operators.
            If specified, `interaction_pairs` should be a pair of lists,
            for alpha-alpha and alpha-beta interactions, in that order.
            Either list can be substituted with ``None`` to indicate no restrictions
            on interactions.
            Each list should contain pairs of integers representing the orbitals
            that are allowed to interact. These pairs can also be interpreted as
            indices of diagonal Coulomb matrix entries that are allowed to be
            nonzero.
            Each integer pair must be upper triangular, that is, of the form
            :math:`(i, j)` where :math:`i \leq j`.
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

    Returns:
        - The diagonal Coulomb matrices, as a Numpy array of shape
          `(n_reps, norb, norb)`.
          The first axis indexes the eigenvectors of the decomposition, and he last two
          axes index the rows and columns of the matrices.
        - The orbital rotations, as a Numpy array of shape
          `(n_reps, norb, norb)`.
          The first axis indexes the eigenvectors of the decomposition, and he last two
          axes index the rows and columns of the matrices.
    """
    nocc, _, _, _ = t2.shape
    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2, tol=tol)
    _, _, norb, _ = orbital_rotations.shape
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)
    n_reps_full, norb, _ = orbital_rotations.shape
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)
    if n_reps is None:
        return diag_coulomb_mats, orbital_rotations

    if not multi_stage_optimization:
        n_reps_full = n_reps
    if begin_reps is None:
        begin_reps = n_reps_full

    pairs_aa, pairs_ab = interaction_pairs
    # Zero out diagonal coulomb matrix entries
    pairs: list[tuple[int, int]] = []
    if pairs_aa is not None:
        pairs += pairs_aa
    if pairs_ab is not None:
        pairs += pairs_ab
    if not pairs:
        diag_coulomb_mask = np.ones((norb, norb), dtype=bool)
    else:
        diag_coulomb_mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs)
        diag_coulomb_mask[rows, cols] = True
        diag_coulomb_mask[cols, rows] = True

    # diag_coulomb_mask
    diag_coulomb_mask = np.triu(diag_coulomb_mask)
    list_init_loss = []
    list_final_loss = []

    list_reps = [i for i in range(begin_reps, n_reps, -step)] + [n_reps]
    for n_tensors in list_reps:
        diag_coulomb_mats = diag_coulomb_mats[:n_tensors]
        orbital_rotations = orbital_rotations[:n_tensors]

        def fun_jax(core_coulomb_params, orbital_rotations_log_tri):
            orbital_rotations_log_real_tri = jnp.real(orbital_rotations_log_tri)
            orbital_rotations_log_imag_tri = jnp.imag(orbital_rotations_log_tri)
            orbital_rotations_log_real = orbital_rotations_log_real_tri - jnp.transpose(
                orbital_rotations_log_real_tri, (0, 2, 1)
            )
            diagonal_element = jnp.stack(
                [
                    jnp.diag(jnp.diag(orbital_rotation))
                    for orbital_rotation in orbital_rotations_log_imag_tri
                ],
                axis=0,
            )
            orbital_rotations_log_imag = orbital_rotations_log_imag_tri + jnp.transpose(
                orbital_rotations_log_imag_tri, (0, 2, 1)
            )
            orbital_rotations_log = (
                orbital_rotations_log_real
                + 1j * orbital_rotations_log_imag
                - 1j * diagonal_element
            )
            eigs, vecs = jnp.linalg.eigh(-1j * orbital_rotations_log)

            param_indices = np.nonzero(diag_coulomb_mask)
            param_length = len(param_indices[0])
            list_diag_coulomb_mats = []
            for i in range(n_tensors):
                diag_coulomb_mat = jnp.zeros((norb, norb), complex)
                diag_coulomb_mat = diag_coulomb_mat.at[param_indices].set(
                    core_coulomb_params[i * param_length : (i + 1) * param_length]
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
            orbital_rotations = jnp.einsum(
                "tij,tj,tkj->tik", vecs, jnp.exp(1j * eigs), vecs.conj()
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

        value_and_grad_func = jax.value_and_grad(fun_jax, argnums=(0, 1))

        def fun_jac(x):
            orbital_rotations_log = _params_to_leaf_logs(x, n_tensors, norb)
            orbital_rotations_log_jax = jnp.array(orbital_rotations_log)
            mask = jnp.ones((norb, norb), dtype=bool)
            mask = jnp.triu(mask)
            orbital_rotations_log_jax_tri = orbital_rotations_log_jax * mask
            n_leaf_params = n_tensors * (
                norb * (norb - 1) // 2 + norb * (norb + 1) // 2
            )
            core_coulomb_params = jnp.array(x[n_leaf_params:] + 0j)

            val, (grad_diag_coulomb_params, grad_orbital_rotations_log_jax_tri) = (
                value_and_grad_func(core_coulomb_params, orbital_rotations_log_jax_tri)
            )
            reshaped_grad = _reshape_grad(
                grad_diag_coulomb_params, grad_orbital_rotations_log_jax_tri
            )
            return val, reshaped_grad

        x0 = _df_tensors_to_params(
            diag_coulomb_mats, orbital_rotations, diag_coulomb_mask
        )

        init_loss, _ = fun_jac(x0)
        list_init_loss.append(init_loss)
        result = scipy.optimize.minimize(
            fun_jac,
            x0,
            method=method,
            jac=True,
            callback=callback,
            options=options,
        )

        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            result.x, n_tensors, norb, diag_coulomb_mask
        )
        final_loss, _ = fun_jac(result.x)
        list_final_loss.append(final_loss)

    return diag_coulomb_mats, orbital_rotations
