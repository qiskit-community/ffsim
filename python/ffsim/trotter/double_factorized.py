# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Trotter simulation for double-factorized Hamiltonian."""

from __future__ import annotations

import cmath

import numpy as np
import scipy.linalg

from ffsim.gates import (
    apply_diag_coulomb_evolution,
    apply_num_op_sum_evolution,
    apply_orbital_rotation,
)
from ffsim.hamiltonians import DoubleFactorizedHamiltonian
from ffsim.trotter._util import simulate_trotter_step_iterator


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
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        n_steps: The number of Trotter steps.
        order: The order of the Trotter decomposition.
        copy: Whether to copy the vector before operating on it.

            - If `copy=True` then this function always returns a newly allocated
              vector and the original vector is left untouched.
            - If `copy=False` then this function may still return a newly allocated
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

    one_body_energies, one_body_basis_change = scipy.linalg.eigh(
        hamiltonian.one_body_tensor
    )
    step_time = time / n_steps

    current_basis = np.eye(norb, dtype=complex)
    for _ in range(n_steps):
        vec, current_basis = _simulate_trotter_step_double_factorized(
            vec,
            current_basis,
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
    vec = apply_orbital_rotation(vec, current_basis, norb=norb, nelec=nelec, copy=False)
    vec *= cmath.exp(-1j * time * hamiltonian.constant)

    return vec


def _simulate_trotter_step_double_factorized(
    vec: np.ndarray,
    current_basis: np.ndarray,
    one_body_energies: np.ndarray,
    one_body_basis_change: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    order: int,
    z_representation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    for term_index, time in simulate_trotter_step_iterator(
        1 + len(diag_coulomb_mats), time, order
    ):
        if term_index == 0:
            vec = apply_orbital_rotation(
                vec,
                one_body_basis_change.T.conj() @ current_basis,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
            vec = apply_num_op_sum_evolution(
                vec,
                one_body_energies,
                time,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
            current_basis = one_body_basis_change
        else:
            orbital_rotation = orbital_rotations[term_index - 1]
            vec = apply_orbital_rotation(
                vec,
                orbital_rotation.T.conj() @ current_basis,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
            vec = apply_diag_coulomb_evolution(
                vec,
                diag_coulomb_mats[term_index - 1],
                time,
                norb=norb,
                nelec=nelec,
                z_representation=z_representation,
                copy=False,
            )
            current_basis = orbital_rotation
    return vec, current_basis
