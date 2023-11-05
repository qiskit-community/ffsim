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

from collections.abc import Iterator

import numpy as np

from ffsim.gates import apply_diag_coulomb_evolution, apply_num_op_sum_evolution
from ffsim.hamiltonians import DoubleFactorizedHamiltonian


def _simulate_trotter_step_iterator(
    n_terms: int, time: float, order: int = 0
) -> Iterator[tuple[int, float]]:
    if order == 0:
        for i in range(n_terms):
            yield i, time
    else:
        yield from _simulate_trotter_step_iterator_symmetric(n_terms, time, order)


def _simulate_trotter_step_iterator_symmetric(
    n_terms: int, time: float, order: int
) -> Iterator[tuple[int, float]]:
    if order == 1:
        for i in range(n_terms - 1):
            yield i, time / 2
        yield n_terms - 1, time
        for i in reversed(range(n_terms - 1)):
            yield i, time / 2
    else:
        split_time = time / (4 - 4 ** (1 / (2 * order - 1)))
        for _ in range(2):
            yield from _simulate_trotter_step_iterator_symmetric(
                n_terms, split_time, order - 1
            )
        yield from _simulate_trotter_step_iterator_symmetric(
            n_terms, time - 4 * split_time, order - 1
        )
        for _ in range(2):
            yield from _simulate_trotter_step_iterator_symmetric(
                n_terms, split_time, order - 1
            )


def simulate_trotter_double_factorized(
    vec: np.ndarray,
    hamiltonian: DoubleFactorizedHamiltonian,
    time: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    n_steps: int = 1,
    order: int = 0,
    copy: bool = True,
) -> np.ndarray:
    """Double-factorized Hamiltonian simulation using Trotter-Suzuki formula.

    Args:
        vec: The state vector to evolve.
        hamiltonian: The Hamiltonian.
        time: The evolution time.
        nelec: The number of alpha and beta electrons.
        n_steps: The number of Trotter steps.
        order: The order of the Trotter decomposition.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.

    Returns:
        The final state of the simulation.
    """
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}.")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}.")
    if copy:
        vec = vec.copy()
    if n_steps == 0:
        return vec

    one_body_energies, one_body_basis_change = np.linalg.eigh(
        hamiltonian.one_body_tensor
    )
    step_time = time / n_steps

    for _ in range(n_steps):
        vec = _simulate_trotter_step_double_factorized(
            vec,
            one_body_energies,
            one_body_basis_change,
            hamiltonian.diag_coulomb_mats,
            hamiltonian.orbital_rotations,
            step_time,
            norb=norb,
            nelec=nelec,
            order=order,
            z_representation=hamiltonian.z_representation,
        )

    return vec


def _simulate_trotter_step_double_factorized(
    vec: np.ndarray,
    one_body_energies: np.ndarray,
    one_body_basis_change: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    order: int,
    z_representation: bool,
) -> np.ndarray:
    for term_index, time in _simulate_trotter_step_iterator(
        1 + len(diag_coulomb_mats), time, order
    ):
        if term_index == 0:
            vec = apply_num_op_sum_evolution(
                vec,
                one_body_energies,
                time,
                norb=norb,
                nelec=nelec,
                orbital_rotation=one_body_basis_change,
                copy=False,
            )
        else:
            vec = apply_diag_coulomb_evolution(
                vec,
                diag_coulomb_mats[term_index - 1],
                time,
                norb=norb,
                nelec=nelec,
                orbital_rotation=orbital_rotations[term_index - 1],
                z_representation=z_representation,
                copy=False,
            )
    return vec
