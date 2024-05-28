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


def contract_diag_coulomb_into_buffer_num_rep_slow(
    vec: np.ndarray,
    mat_aa: np.ndarray,
    mat_ab: np.ndarray,
    mat_bb: np.ndarray,
    norb: int,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
    out: np.ndarray,
) -> None:
    dim_a, dim_b = vec.shape
    alpha_coeffs = np.zeros((dim_a,), dtype=complex)
    beta_coeffs = np.zeros((dim_b,), dtype=complex)
    coeff_map = np.zeros((dim_a, norb), dtype=complex)

    for i, occ in enumerate(occupations_b):
        coeff = 0
        for orb_1, orb_2 in itertools.combinations_with_replacement(occ, 2):
            coeff += mat_bb[orb_1, orb_2]
        beta_coeffs[i] = coeff

    for i, (row, orbs) in enumerate(zip(coeff_map, occupations_a)):
        coeff = 0
        for j in range(len(orbs)):
            row += mat_ab[orbs[j]]
            for k in range(j, len(orbs)):
                coeff += mat_aa[orbs[j], orbs[k]]
        alpha_coeffs[i] = coeff

    for source, target, alpha_coeff, coeff_map in zip(
        vec, out, alpha_coeffs, coeff_map
    ):
        for j, occ_b in enumerate(occupations_b):
            coeff = alpha_coeff + beta_coeffs[j]
            for orb_b in occ_b:
                coeff += coeff_map[orb_b]
            target[j] += coeff * source[j]


def contract_diag_coulomb_into_buffer_z_rep_slow(
    vec: np.ndarray,
    mat_aa: np.ndarray,
    mat_ab: np.ndarray,
    mat_bb: np.ndarray,
    norb: int,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
    out: np.ndarray,
) -> None:
    dim_a, dim_b = vec.shape
    alpha_coeffs = np.zeros((dim_a,), dtype=complex)
    beta_coeffs = np.zeros((dim_b,), dtype=complex)
    coeff_map = np.zeros((dim_a, norb), dtype=complex)

    for i, str0 in enumerate(strings_b):
        coeff = 0
        for j in range(norb):
            sign_j = -1 if str0 >> j & 1 else 1
            for k in range(j + 1, norb):
                sign_k = -1 if str0 >> k & 1 else 1
                coeff += sign_j * sign_k * mat_bb[j, k]
        beta_coeffs[i] = coeff

    for i, (row, str0) in enumerate(zip(coeff_map, strings_a)):
        coeff = 0
        for j in range(norb):
            sign_j = -1 if str0 >> j & 1 else 1
            row += sign_j * mat_ab[j]
            for k in range(j + 1, norb):
                sign_k = -1 if str0 >> k & 1 else 1
                coeff += sign_j * sign_k * mat_aa[j, k]
        alpha_coeffs[i] = coeff

    for source, target, alpha_coeff, coeff_map in zip(
        vec, out, alpha_coeffs, coeff_map
    ):
        for i, str0 in enumerate(strings_b):
            coeff = alpha_coeff + beta_coeffs[i]
            for j in range(norb):
                sign_j = -1 if str0 >> j & 1 else 1
                coeff += sign_j * coeff_map[j]
            target[i] += 0.25 * coeff * source[i]
