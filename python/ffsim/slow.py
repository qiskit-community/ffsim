# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Python versions of functions that were rewritten in Rust."""

from __future__ import annotations

import itertools

import numpy as np

from ffsim.gates.gates import _apply_phase_shift


def apply_givens_rotation_in_place_slow(
    vec: np.ndarray,
    c: float,
    s: float,
    phase: complex,
    slice1: np.ndarray,
    slice2: np.ndarray,
) -> None:
    """Apply a Givens rotation to slices of a state vector."""
    phase_conj = phase.conjugate()
    for i, j in zip(slice1, slice2):
        vec[i] *= phase_conj
        tmp = c * vec[i] + s * vec[j]
        vec[j] = c * vec[j] - s * vec[i]
        vec[i] = tmp
        vec[i] *= phase


def gen_orbital_rotation_index_in_place_slow(
    norb: int,
    nocc: int,
    linkstr_index: np.ndarray,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_strings_index: np.ndarray,
    off_diag_index: np.ndarray,
) -> None:
    """Generate orbital rotation index."""
    diag_counter = np.zeros(norb, dtype=np.uint)
    off_diag_counter = np.zeros(norb, dtype=np.uint)
    for str0, tab in enumerate(linkstr_index[:, :, 0]):
        for orb in tab[:nocc]:
            count = diag_counter[orb]
            diag_strings[orb, count] = str0
            diag_counter[orb] += 1
        for orb in tab[nocc:norb]:
            count = off_diag_counter[orb]
            off_diag_strings[orb, count] = str0
            off_diag_strings_index[orb, str0] = count
            off_diag_counter[orb] += 1

    index_counter = np.zeros_like(off_diag_strings)
    for str0, tab in enumerate(linkstr_index):
        for orb_c, orb_d, str1, sign in tab[nocc:]:
            # str0 -> annihilate orb_d -> create orb_c -> str1
            index = off_diag_strings_index[orb_c, str0]
            count = index_counter[orb_c, index]
            off_diag_index[orb_c, index, count] = orb_d, str1, sign
            index_counter[orb_c, index] += 1


def apply_single_column_transformation_in_place_slow(
    vec: np.ndarray,
    column: np.ndarray,
    diag_val: complex,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_index: np.ndarray,
) -> None:
    """Apply a single-column orbital rotation."""
    for str0, tab in zip(off_diag_strings, off_diag_index):
        for orb, str1, sign in tab:
            vec[str0] += sign * column[orb] * vec[str1]
    for str0 in diag_strings:
        vec[str0] *= diag_val


def apply_num_op_sum_evolution_in_place_slow(
    vec: np.ndarray, phases: np.ndarray, occupations: np.ndarray
) -> None:
    """Apply time evolution by a sum of number operators in-place."""
    for row, orbs in zip(vec, occupations):
        phase = 1
        for orb in orbs:
            phase *= phases[orb]
        row *= phase


def apply_diag_coulomb_evolution_in_place_num_rep_slow(
    vec: np.ndarray,
    mat_exp: np.ndarray,
    norb: int,
    mat_alpha_beta_exp: np.ndarray,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
) -> None:
    """Apply time evolution by a diagonal Coulomb operator in-place."""
    dim_a, dim_b = vec.shape
    alpha_phases = np.empty((dim_a,), dtype=complex)
    beta_phases = np.empty((dim_b,), dtype=complex)
    phase_map = np.ones((dim_a, norb), dtype=complex)

    for i, orbs in enumerate(occupations_b):
        phase = 1
        for orb_1, orb_2 in itertools.combinations_with_replacement(orbs, 2):
            phase *= mat_exp[orb_1, orb_2]
        beta_phases[i] = phase

    for i, (row, orbs) in enumerate(zip(phase_map, occupations_a)):
        phase = 1
        for j in range(len(orbs)):
            row *= mat_alpha_beta_exp[orbs[j]]
            for k in range(j, len(orbs)):
                phase *= mat_exp[orbs[j], orbs[k]]
        alpha_phases[i] = phase

    for row, alpha_phase, phase_map in zip(vec, alpha_phases, phase_map):
        for j, occ_b in enumerate(occupations_b):
            phase = alpha_phase * beta_phases[j]
            for orb_b in occ_b:
                phase *= phase_map[orb_b]
            row[j] *= phase


def apply_diag_coulomb_evolution_in_place_z_rep_slow(
    vec: np.ndarray,
    mat_exp: np.ndarray,
    mat_exp_conj: np.ndarray,
    norb: int,
    mat_alpha_beta_exp: np.ndarray,
    mat_alpha_beta_exp_conj: np.ndarray,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
) -> None:
    """Apply time evolution by a diagonal Coulomb operator in-place."""
    dim_a, dim_b = vec.shape
    alpha_phases = np.empty((dim_a,), dtype=complex)
    beta_phases = np.empty((dim_b,), dtype=complex)
    phase_map = np.ones((dim_a, norb), dtype=complex)

    for i, str0 in enumerate(strings_b):
        phase = 1
        for j in range(norb):
            sign_j = str0 >> j & 1
            for k in range(j + 1, norb):
                sign_k = str0 >> k & 1
                mat = mat_exp_conj if sign_j ^ sign_k else mat_exp
                phase *= mat[j, k]
        beta_phases[i] = phase

    for i, (row, str0) in enumerate(zip(phase_map, strings_a)):
        phase = 1
        for j in range(norb):
            sign_j = str0 >> j & 1
            mat = mat_alpha_beta_exp_conj if sign_j else mat_alpha_beta_exp
            row *= mat[j]
            for k in range(j + 1, norb):
                sign_k = str0 >> k & 1
                mat = mat_exp_conj if sign_j ^ sign_k else mat_exp
                phase *= mat[j, k]
        alpha_phases[i] = phase

    for row, alpha_phase, phase_map in zip(vec, alpha_phases, phase_map):
        for i, str0 in enumerate(strings_b):
            phase = alpha_phase * beta_phases[i]
            for j in range(norb):
                phase_shift = phase_map[j]
                if str0 >> j & 1:
                    phase_shift = phase_shift.conjugate()
                phase *= phase_shift
            row[i] *= phase


def apply_diag_coulomb_evolution_in_place_num_rep_numpy(
    vec: np.ndarray,
    mat_exp: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    mat_alpha_beta_exp: np.ndarray,
) -> None:
    """Apply time evolution by a diagonal Coulomb operator in-place."""
    mat_alpha_beta_exp = mat_alpha_beta_exp.copy()
    mat_alpha_beta_exp[np.diag_indices(norb)] **= 0.5
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        for sigma in range(2):
            orbitals: list[set[int]] = [set(), set()]
            orbitals[sigma].add(i)
            orbitals[sigma].add(j)
            _apply_phase_shift(
                vec,
                mat_exp[i, j],
                (tuple(orbitals[0]), tuple(orbitals[1])),
                norb=norb,
                nelec=nelec,
                copy=False,
            )
            orbitals = [set() for _ in range(2)]
            orbitals[sigma].add(i)
            orbitals[1 - sigma].add(j)
            _apply_phase_shift(
                vec,
                mat_alpha_beta_exp[i, j],
                (tuple(orbitals[0]), tuple(orbitals[1])),
                norb=norb,
                nelec=nelec,
                copy=False,
            )


def contract_diag_coulomb_into_buffer_num_rep_slow(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    mat_alpha_beta: np.ndarray,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
    out: np.ndarray,
) -> None:
    dim_a, dim_b = vec.shape
    alpha_coeffs = np.empty((dim_a,), dtype=complex)
    beta_coeffs = np.empty((dim_b,), dtype=complex)
    coeff_map = np.zeros((dim_a, norb), dtype=complex)

    for i, occ in enumerate(occupations_b):
        coeff = 0
        for orb_1, orb_2 in itertools.combinations_with_replacement(occ, 2):
            coeff += mat[orb_1, orb_2]
        beta_coeffs[i] = coeff

    for i, (row, orbs) in enumerate(zip(coeff_map, occupations_a)):
        coeff = 0
        for j in range(len(orbs)):
            row += mat_alpha_beta[orbs[j]]
            for k in range(j, len(orbs)):
                coeff += mat[orbs[j], orbs[k]]
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
    mat: np.ndarray,
    norb: int,
    mat_alpha_beta: np.ndarray,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
    out: np.ndarray,
) -> None:
    dim_a, dim_b = vec.shape
    alpha_coeffs = np.empty((dim_a,), dtype=complex)
    beta_coeffs = np.empty((dim_b,), dtype=complex)
    coeff_map = np.zeros((dim_a, norb), dtype=complex)

    for i, str0 in enumerate(strings_b):
        coeff = 0
        for j in range(norb):
            sign_j = -1 if str0 >> j & 1 else 1
            for k in range(j + 1, norb):
                sign_k = -1 if str0 >> k & 1 else 1
                coeff += sign_j * sign_k * mat[j, k]
        beta_coeffs[i] = coeff

    for i, (row, str0) in enumerate(zip(coeff_map, strings_a)):
        coeff = 0
        for j in range(norb):
            sign_j = -1 if str0 >> j & 1 else 1
            row += sign_j * mat_alpha_beta[j]
            for k in range(j + 1, norb):
                sign_k = -1 if str0 >> k & 1 else 1
                coeff += sign_j * sign_k * mat[j, k]
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


def contract_num_op_sum_spin_into_buffer_slow(
    vec: np.ndarray, coeffs: np.ndarray, occupations: np.ndarray, out: np.ndarray
) -> None:
    for source_row, target_row, orbs in zip(vec, out, occupations):
        coeff = 0
        for orb in orbs:
            coeff += coeffs[orb]
        target_row += coeff * source_row
