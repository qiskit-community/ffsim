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
    column: np.ndarray,
    vec: np.ndarray,
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
    phases: np.ndarray,
    vec: np.ndarray,
    occupations: np.ndarray,
) -> None:
    """Apply time evolution by a sum of number operators in-place."""
    for row, orbs in zip(vec, occupations):
        phase = 1
        for orb in orbs:
            phase *= phases[orb]
        row *= phase


def apply_diag_coulomb_evolution_in_place_slow(
    mat_exp: np.ndarray,
    vec: np.ndarray,
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

    for i, (row, orbs) in enumerate(zip(phase_map, occupations_a)):
        phase = 1
        for j in range(len(orbs)):
            row *= mat_alpha_beta_exp[orbs[j]]
            for k in range(j, len(orbs)):
                phase *= mat_exp[orbs[j], orbs[k]]
        alpha_phases[i] = phase

    for i, orbs in enumerate(occupations_b):
        phase = 1
        for orb_1, orb_2 in itertools.combinations_with_replacement(orbs, 2):
            phase *= mat_exp[orb_1, orb_2]
        beta_phases[i] = phase

    for row, alpha_phase, phase_map in zip(vec, alpha_phases, phase_map):
        for j, occ_b in enumerate(occupations_b):
            phase = alpha_phase * beta_phases[j]
            for orb_b in occ_b:
                phase *= phase_map[orb_b]
            row[j] *= phase
