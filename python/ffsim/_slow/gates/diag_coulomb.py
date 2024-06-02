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

from ffsim.gates.basic_gates import _apply_phase_shift


def apply_diag_coulomb_evolution_in_place_num_rep_slow(
    vec: np.ndarray,
    mat_exp_aa: np.ndarray,
    mat_exp_ab: np.ndarray,
    mat_exp_bb: np.ndarray,
    norb: int,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
) -> None:
    """Apply time evolution by a diagonal Coulomb operator in-place."""
    dim_a, dim_b = vec.shape
    alpha_phases = np.ones((dim_a,), dtype=complex)
    beta_phases = np.ones((dim_b,), dtype=complex)
    phase_map = np.ones((dim_a, norb), dtype=complex)

    for i, orbs in enumerate(occupations_b):
        phase = 1
        for orb_1, orb_2 in itertools.combinations_with_replacement(orbs, 2):
            phase *= mat_exp_bb[orb_1, orb_2]
        beta_phases[i] = phase

    for i, (row, orbs) in enumerate(zip(phase_map, occupations_a)):
        phase = 1
        for j in range(len(orbs)):
            row *= mat_exp_ab[orbs[j]]
            for k in range(j, len(orbs)):
                phase *= mat_exp_aa[orbs[j], orbs[k]]
        alpha_phases[i] = phase

    for row, alpha_phase, phase_map in zip(vec, alpha_phases, phase_map):
        for j, occ_b in enumerate(occupations_b):
            phase = alpha_phase * beta_phases[j]
            for orb_b in occ_b:
                phase *= phase_map[orb_b]
            row[j] *= phase


def apply_diag_coulomb_evolution_in_place_z_rep_slow(
    vec: np.ndarray,
    mat_exp_aa: np.ndarray,
    mat_exp_ab: np.ndarray,
    mat_exp_bb: np.ndarray,
    mat_exp_aa_conj: np.ndarray,
    mat_exp_ab_conj: np.ndarray,
    mat_exp_bb_conj: np.ndarray,
    norb: int,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
) -> None:
    """Apply time evolution by a diagonal Coulomb operator in-place."""
    dim_a, dim_b = vec.shape
    alpha_phases = np.ones((dim_a,), dtype=complex)
    beta_phases = np.ones((dim_b,), dtype=complex)
    phase_map = np.ones((dim_a, norb), dtype=complex)

    for i, str0 in enumerate(strings_b):
        phase = 1
        for j in range(norb):
            sign_j = str0 >> j & 1
            for k in range(j + 1, norb):
                sign_k = str0 >> k & 1
                mat = mat_exp_bb_conj if sign_j ^ sign_k else mat_exp_bb
                phase *= mat[j, k]
        beta_phases[i] = phase

    for i, (row, str0) in enumerate(zip(phase_map, strings_a)):
        phase = 1
        for j in range(norb):
            sign_j = str0 >> j & 1
            mat = mat_exp_ab_conj if sign_j else mat_exp_ab
            row *= mat[j]
            for k in range(j + 1, norb):
                sign_k = str0 >> k & 1
                mat = mat_exp_aa_conj if sign_j ^ sign_k else mat_exp_aa
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
    mat_exp_aa: np.ndarray,
    mat_exp_ab: np.ndarray,
    mat_exp_bb: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> None:
    """Apply time evolution by a diagonal Coulomb operator in-place."""
    mat_exp_ab = mat_exp_ab.copy()
    mat_exp_ab[np.diag_indices(norb)] **= 0.5
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        for sigma, mat_exp in enumerate([mat_exp_aa, mat_exp_bb]):
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
            orbitals = [set(), set()]
            orbitals[sigma].add(i)
            orbitals[1 - sigma].add(j)
            _apply_phase_shift(
                vec,
                mat_exp_ab[i, j],
                (tuple(orbitals[0]), tuple(orbitals[1])),
                norb=norb,
                nelec=nelec,
                copy=False,
            )
