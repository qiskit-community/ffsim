# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermionic quantum computation gates."""

from __future__ import annotations

import numpy as np
from scipy.special import comb

from ffsim.gates.orbital_rotation import _one_subspace_indices, apply_orbital_rotation


def _apply_phase_shift(
    vec: np.ndarray,
    phase: complex,
    target_orbs: tuple[tuple[int, ...], tuple[int, ...]],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    """Apply a phase shift controlled on target orbitals.

    Multiplies by the phase each coefficient corresponding to a string in which
    the target orbitals are all 1 (occupied).
    """
    if copy:
        vec = vec.copy()
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    vec = vec.reshape((dim_a, dim_b))
    target_orbs_a, target_orbs_b = target_orbs
    indices_a = _one_subspace_indices(norb, n_alpha, target_orbs_a)
    indices_b = _one_subspace_indices(norb, n_beta, target_orbs_b)
    vec[np.ix_(indices_a, indices_b)] *= phase
    return vec.reshape(-1)


def apply_givens_rotation(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a Givens rotation gate.

    The Givens rotation gate is

    .. math::

        G(\theta) = \exp(\theta (a^\dagger_i a_j - a\dagger_j a_i))

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals (i, j) to rotate.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(norb)
    mat[np.ix_(target_orbs, target_orbs)] = [[c, s], [-s, c]]
    return apply_orbital_rotation(vec, mat, norb=norb, nelec=nelec, copy=copy)


def apply_tunneling_interaction(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a tunneling interaction gate.

    The tunneling interaction gate is

    .. math::

        T(\theta) = \exp(i \theta (a^\dagger_i a_j + a\dagger_j a_i))

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals (i, j) on which to apply the interaction.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    vec = apply_num_interaction(
        vec, -np.pi / 2, target_orbs[0], norb=norb, nelec=nelec, copy=copy
    )
    vec = apply_givens_rotation(
        vec, theta, target_orbs, norb=norb, nelec=nelec, copy=False
    )
    vec = apply_num_interaction(
        vec, np.pi / 2, target_orbs[0], norb=norb, nelec=nelec, copy=False
    )
    return vec


def apply_num_interaction(
    vec: np.ndarray,
    theta: float,
    target_orb: int,
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply a number interaction gate.

    The number interaction gate is

    .. math::

        N(\theta) = \exp(i \theta a^\dagger_i a_i)

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orb: The orbital on which to apply the interaction.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if copy:
        vec = vec.copy()
    for sigma in range(2):
        vec = apply_num_op_prod_interaction(
            vec, theta, ((target_orb, sigma),), norb=norb, nelec=nelec, copy=False
        )
    return vec


def apply_num_op_prod_interaction(
    vec: np.ndarray,
    theta: float,
    target_orbs: tuple[tuple[int, int], ...],
    norb: int,
    nelec: tuple[int, int],
    *,
    copy: bool = True,
):
    r"""Apply interaction gate for product of number operators.

    The gate is

    .. math::

        NP(\theta) = \exp(i \theta \prod a^\dagger_{i, \sigma} a_{i, \sigma})

    Args:
        vec: The state vector to be transformed.
        theta: The rotation angle.
        target_orbs: The orbitals on which to apply the interaction. This should
            be a tuple of (orbital, spin) pairs.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        copy: Whether to copy the vector before operating on it.
            - If ``copy=True`` then this function always returns a newly allocated
            vector and the original vector is left untouched.
            - If ``copy=False`` then this function may still return a newly allocated
            vector, but the original vector may have its data overwritten.
            It is also possible that the original vector is returned,
            modified in-place.
    """
    if copy:
        vec = vec.copy()
    orbitals: list[set[int]] = [set(), set()]
    for i, spin_i in target_orbs:
        orbitals[spin_i].add(i)
    vec = _apply_phase_shift(
        vec,
        np.exp(1j * theta),
        (tuple(orbitals[0]), tuple(orbitals[1])),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec
