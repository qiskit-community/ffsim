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


def simulate_trotter_suzuki_double_factorized(
    one_body_tensor: np.ndarray,
    core_tensors: np.ndarray,
    leaf_tensors: np.ndarray,
    time: float,
    initial_state: np.ndarray,
    nelec: tuple[int, int],
    n_steps: int = 1,
    order: int = 0,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Double-factorized Hamiltonian simulation using Trotter-Suzuki formula.

    Args:
        one_body_tensor: The one-body tensor of the double-factorized Hamiltonian.
        core_tensors: The core tensors of the double-factorized Hamiltonian.
        leaf_tensors: The leaf tensors of the double-factorized Hamiltonian.
        time: The evolution time.
        initial_state: The initial state.
        nelec: The number of alpha and beta electrons.
        n_steps: The number of Trotter steps.
        order: The order of the Trotter decomposition.

    Returns:
        The final state of the simulation.
    """
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}.")
    one_body_energies, one_body_basis_change = np.linalg.eigh(one_body_tensor)
    step_time = time / n_steps
    if copy:
        final_state = initial_state.copy()
    for _ in range(n_steps):
        final_state = _simulate_trotter_step_double_factorized(
            one_body_energies,
            one_body_basis_change,
            core_tensors,
            leaf_tensors,
            step_time,
            final_state,
            nelec,
            order,
        )
    return final_state


def _simulate_trotter_step_double_factorized(
    one_body_energies: np.ndarray,
    one_body_basis_change: np.ndarray,
    core_tensors: np.ndarray,
    leaf_tensors: np.ndarray,
    time: float,
    initial_state: np.ndarray,
    nelec: tuple[int, int],
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    final_state = initial_state
    norb = len(one_body_energies)
    for term_index, time in _simulate_trotter_step_iterator(
        1 + len(core_tensors), time, order
    ):
        if term_index == 0:
            final_state = apply_num_op_sum_evolution(
                one_body_energies,
                final_state,
                time,
                norb=norb,
                nelec=nelec,
                orbital_rotation=one_body_basis_change,
                copy=False,
            )
        else:
            final_state = apply_diag_coulomb_evolution(
                core_tensors[term_index - 1],
                final_state,
                time,
                norb=norb,
                nelec=nelec,
                orbital_rotation=leaf_tensors[term_index - 1],
                copy=False,
            )
    return final_state
