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

from fqcsim.gates import (
    apply_core_tensor_evolution,
    apply_num_op_sum_evolution,
    apply_orbital_rotation,
)


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
    final_state = initial_state.copy()
    norb, _ = one_body_tensor.shape
    current_basis_change = np.eye(norb)
    for _ in range(n_steps):
        final_state, current_basis_change = _simulate_trotter_step_double_factorized(
            current_basis_change,
            one_body_energies,
            one_body_basis_change,
            core_tensors,
            leaf_tensors,
            step_time,
            final_state,
            nelec,
            order,
        )
    final_state = apply_orbital_rotation(current_basis_change, final_state, norb, nelec)
    return final_state


def _simulate_trotter_step_double_factorized(
    current_basis_change: np.ndarray,
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
    for term_index, time in _simulate_trotter_step_iterator(
        1 + len(core_tensors), time, order
    ):
        if term_index == 0:
            final_state, current_basis_change = _apply_one_body_evolution(
                current_basis_change,
                one_body_energies,
                one_body_basis_change,
                final_state,
                time,
                nelec,
            )
        else:
            final_state, current_basis_change = _apply_two_body_term_evolution(
                current_basis_change,
                core_tensors[term_index - 1],
                leaf_tensors[term_index - 1],
                final_state,
                time,
                nelec,
            )
    return final_state, current_basis_change


def _apply_one_body_evolution(
    current_basis_change: np.ndarray,
    one_body_energies: np.ndarray,
    one_body_basis_change: np.ndarray,
    vec: np.ndarray,
    time: float,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    norb, _ = current_basis_change.shape
    vec = apply_orbital_rotation(
        one_body_basis_change.T.conj() @ current_basis_change,
        vec,
        norb=norb,
        nelec=nelec,
    )
    vec = apply_num_op_sum_evolution(
        one_body_energies,
        vec,
        time,
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec, one_body_basis_change


def _apply_two_body_term_evolution(
    current_basis_change: np.ndarray,
    core_tensor: np.ndarray,
    leaf_tensor: np.ndarray,
    vec: np.ndarray,
    time: float,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    norb, _ = current_basis_change.shape
    vec = apply_orbital_rotation(
        leaf_tensor.T.conj() @ current_basis_change,
        vec,
        norb=norb,
        nelec=nelec,
    )
    vec = apply_core_tensor_evolution(
        core_tensor,
        vec,
        time,
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec, leaf_tensor
