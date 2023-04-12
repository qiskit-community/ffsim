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

from ffsim.gates import apply_phase_shift


def gen_orbital_rotation_index_in_place_slow(
    norb: int,
    nocc: int,
    linkstr_index: np.ndarray,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_strings_index: np.ndarray,
    off_diag_index: np.ndarray,
) -> None:
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
            index = off_diag_strings_index[orb_d, str1]
            count = index_counter[orb_d, index]
            off_diag_index[orb_d, index, count] = orb_c, str0, sign
            index_counter[orb_d, index] += 1


def apply_single_column_transformation_in_place_slow(
    column: np.ndarray,
    vec: np.ndarray,
    diag_val: complex,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_index: np.ndarray,
) -> None:
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
    dim_a, dim_b = vec.shape
    alpha_phases = np.empty((dim_a,), dtype=complex)
    beta_phases = np.empty((dim_b,), dtype=complex)
    phase_map = np.ones((dim_a, norb), dtype=complex)

    for i, occ in enumerate(occupations_a):
        phase = 1
        for orb_1, orb_2 in itertools.combinations_with_replacement(occ, 2):
            phase *= mat_exp[orb_1, orb_2]
        alpha_phases[i] = phase

    for i, occ in enumerate(occupations_b):
        phase = 1
        for orb_1, orb_2 in itertools.combinations_with_replacement(occ, 2):
            phase *= mat_exp[orb_1, orb_2]
        beta_phases[i] = phase

    for row, orbs in zip(phase_map, occupations_a):
        for orb in orbs:
            row *= mat_alpha_beta_exp[orb]

    for row, alpha_phase, phase_map in zip(vec, alpha_phases, phase_map):
        for j, occ_b in enumerate(occupations_b):
            phase = alpha_phase * beta_phases[j]
            for orb_b in occ_b:
                phase *= phase_map[orb_b]
            row[j] *= phase


def apply_diag_coulomb_evolution_in_place_numpy(
    mat_exp: np.ndarray,
    vec: np.ndarray,
    norb: int,
    n_alpha: int,
    n_beta: int,
    *,
    mat_alpha_beta_exp: np.ndarray,
    **kwargs,
) -> None:
    r"""Apply time evolution by a diagonal Coulomb operator.

    Applies

    .. math::
        \exp(-i t \sum_{i, j, \sigma, \tau} Z_{ij} n_{i, \sigma} n_{j, \tau} / 2)

    where :math:`n_{i, \sigma}` denotes the number operator on orbital :math:`i`
    and spin :math:`\sigma`, and :math:`Z` is the matrix input as ``mat``.
    If ``mat_alpha_beta`` is also given, then it is used in place of :math:`Z`
    for the terms in the sum where the spins differ (:math:`\sigma \neq \tau`).
    """
    mat_alpha_beta_exp = mat_alpha_beta_exp.copy()
    mat_alpha_beta_exp[np.diag_indices(norb)] **= 0.5
    nelec = (n_alpha, n_beta)
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        for sigma in range(2):
            orbitals: list[set[int]] = [set(), set()]
            orbitals[sigma].add(i)
            orbitals[sigma].add(j)
            apply_phase_shift(
                mat_exp[i, j],
                vec,
                (tuple(orbitals[0]), tuple(orbitals[1])),
                norb=norb,
                nelec=nelec,
                copy=False,
            )
            orbitals = [set() for _ in range(2)]
            orbitals[sigma].add(i)
            orbitals[1 - sigma].add(j)
            apply_phase_shift(
                mat_alpha_beta_exp[i, j],
                vec,
                (tuple(orbitals[0]), tuple(orbitals[1])),
                norb=norb,
                nelec=nelec,
                copy=False,
            )
