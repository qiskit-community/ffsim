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

import itertools
from functools import lru_cache

import numpy as np
from pyscf.fci import cistring
from scipy.special import comb
from fqcsim.linalg import (
    givens_decomposition,
    apply_matrix_to_slices,
)


def apply_orbital_rotation(
    mat: np.ndarray, vec: np.ndarray, n_orbitals: int, n_electrons: tuple[int, int]
) -> np.ndarray:
    """Apply an orbital rotation to a vector.

    An orbital rotation is also known as a single-particle basis change.

    Args:
        mat: Unitary matrix describing the orbital rotation.
        vec: Vector to be transformed.
        n_orbitals: Number of spatial orbitals.
        n_electrons: Number of alpha and beta electrons.
    """
    givens_rotations, phase_shifts = givens_decomposition(mat)
    for givens_mat, target_orbitals in givens_rotations:
        vec = apply_orbital_rotation_adjacent(
            givens_mat.conj(), vec, target_orbitals, n_orbitals, n_electrons
        )
    for i, phase_shift in enumerate(phase_shifts):
        vec = apply_phase_shift(
            phase_shift, vec, ((i,), ()), n_orbitals, n_electrons, copy=False
        )
        vec = apply_phase_shift(
            phase_shift, vec, ((), (i,)), n_orbitals, n_electrons, copy=False
        )
    return vec


def apply_orbital_rotation_adjacent(
    mat: np.ndarray,
    vec: np.ndarray,
    target_orbitals: tuple[int, int],
    n_orbitals: int,
    n_electrons: tuple[int, int],
):
    """Apply an orbital rotation to adjacent orbitals.

    Args:
        mat: A 2x2 unitary matrix describing the orbital rotation.
        vec: Vector to be transformed.
        target_orbitals: The orbitals to transform.
        n_orbitals: Number of spatial orbitals.
        n_electrons: Number of alpha and beta electrons.
    """
    i, j = target_orbitals
    assert i == j + 1 or i == j - 1, "Target orbitals must be adjacent."
    n_alpha, n_beta = n_electrons
    dim_a = comb(n_orbitals, n_alpha, exact=True)
    dim_b = comb(n_orbitals, n_beta, exact=True)
    vec = vec.reshape((dim_a, dim_b))
    indices = _zero_one_subspace_indices(n_orbitals, n_alpha, target_orbitals)
    slice1 = indices[: len(indices) // 2]
    slice2 = indices[len(indices) // 2 :]
    vec = apply_matrix_to_slices(mat, vec, [slice1, slice2])
    indices = _zero_one_subspace_indices(n_orbitals, n_beta, target_orbitals)
    slice1 = indices[: len(indices) // 2]
    slice2 = indices[len(indices) // 2 :]
    vec = apply_matrix_to_slices(mat, vec, [(Ellipsis, slice1), (Ellipsis, slice2)])
    return vec.reshape((dim_a * dim_b,))


@lru_cache(maxsize=None)
def _zero_one_subspace_indices(
    n_orbitals: int, n_occ: int, target_orbitals: tuple[int, int]
):
    """Return the indices where the target orbitals are 01 or 10."""
    orbitals = _shifted_orbitals(n_orbitals, target_orbitals)
    strings = cistring.make_strings(orbitals, n_occ)
    indices = np.argsort(strings)
    n00 = comb(n_orbitals - 2, n_occ, exact=True)
    n11 = comb(n_orbitals - 2, n_occ - 2, exact=True)
    return indices[n00 : len(indices) - n11]


def apply_phase_shift(
    phase: complex,
    vec: np.ndarray,
    target_orbitals: tuple[tuple[int], tuple[int]],
    n_orbitals: int,
    n_electrons: tuple[int, int],
    copy: bool = True,
):
    if copy:
        vec = vec.copy()
    n_alpha, n_beta = n_electrons
    dim_a = comb(n_orbitals, n_alpha, exact=True)
    dim_b = comb(n_orbitals, n_beta, exact=True)
    vec = vec.reshape((dim_a, dim_b))
    target_orbitals_a, target_orbitals_b = target_orbitals
    indices_a = _one_subspace_indices(n_orbitals, n_alpha, target_orbitals_a)
    indices_b = _one_subspace_indices(n_orbitals, n_beta, target_orbitals_b)
    vec[np.ix_(indices_a, indices_b)] *= phase
    return vec.reshape((dim_a * dim_b,))


@lru_cache(maxsize=None)
def _one_subspace_indices(
    n_orbitals: int, n_occ: int, target_orbitals: tuple[tuple[int], tuple[int]]
):
    """Return the indices where the target orbitals are 1."""
    orbitals = _shifted_orbitals(n_orbitals, target_orbitals)
    strings = cistring.make_strings(orbitals, n_occ)
    indices = np.argsort(strings)
    n0 = comb(n_orbitals, n_occ, exact=True) - comb(
        n_orbitals - len(target_orbitals), n_occ - len(target_orbitals), exact=True
    )
    return indices[n0:]


def _shifted_orbitals(n_orbitals: int, target_orbitals: tuple[int]):
    """Return orbital list with targeted orbitals shifted to the end."""
    orbitals = np.arange(n_orbitals - len(target_orbitals))
    values = sorted(
        zip(target_orbitals, range(n_orbitals - len(target_orbitals), n_orbitals))
    )
    for index, val in values:
        orbitals = np.insert(orbitals, index, val)
    return orbitals


def apply_num_op_sum_evolution(
    coeffs: np.ndarray,
    vec: np.ndarray,
    time: float,
    n_orbitals: int,
    n_electrons: tuple[int, int],
    copy: bool = True,
):
    """Apply a sum of number operators to a vector."""
    if copy:
        vec = vec.copy()
    for i, coeff in enumerate(coeffs):
        vec = apply_phase_shift(
            np.exp(-1j * coeff * time),
            vec,
            ((i,), ()),
            n_orbitals,
            n_electrons,
            copy=False,
        )
        vec = apply_phase_shift(
            np.exp(-1j * coeff * time),
            vec,
            ((), (i,)),
            n_orbitals,
            n_electrons,
            copy=False,
        )
    return vec


def apply_core_tensor_evolution(
    core_tensor: np.ndarray,
    vec: np.ndarray,
    time: float,
    n_orbitals: int,
    n_electrons: tuple[int, int],
    *,
    core_tensor_alpha_beta: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    # TODO n_orbitals does not need to be input here and some other places
    """Apply core tensor evolution."""
    if copy:
        vec = vec.copy()
    n_orbitals, _ = core_tensor.shape
    if core_tensor_alpha_beta is None:
        core_tensor_alpha_beta = core_tensor
    for i, j in itertools.combinations_with_replacement(range(n_orbitals), 2):
        coeff = 0.5 if i == j else 1.0
        vec = apply_phase_shift(
            np.exp(-1j * coeff * time * core_tensor[i, j]),
            vec,
            ((i,) if i == j else (i, j), ()),
            n_orbitals=n_orbitals,
            n_electrons=n_electrons,
            copy=False,
        )
        vec = apply_phase_shift(
            np.exp(-1j * coeff * time * core_tensor[i, j]),
            vec,
            ((), (i,) if i == j else (i, j)),
            n_orbitals=n_orbitals,
            n_electrons=n_electrons,
            copy=False,
        )
        vec = apply_phase_shift(
            np.exp(-1j * coeff * time * core_tensor_alpha_beta[i, j]),
            vec,
            ((i,), (j,)),
            n_orbitals=n_orbitals,
            n_electrons=n_electrons,
            copy=False,
        )
        vec = apply_phase_shift(
            np.exp(-1j * coeff * time * core_tensor_alpha_beta[i, j]),
            vec,
            ((j,), (i,)),
            n_orbitals=n_orbitals,
            n_electrons=n_electrons,
            copy=False,
        )
    return vec


def apply_givens_rotation_adjacent(
    theta: float,
    vec: np.ndarray,
    target_orbitals: tuple[int, int],
    n_orbitals: int,
    n_electrons: tuple[int, int],
):
    r"""Apply a Givens rotation gate to adjacent orbitals.

    The Givens rotation gate is

    .. math::
        G(\theta) = \exp(\theta (a^\dagger_i a_j - a\dagger_j a_i))

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbitals: The orbitals (i, j) to transform.
        n_orbitals: Number of spatial orbitals.
        n_electrons: Number of alpha and beta electrons.
    """
    i, j = target_orbitals
    assert i == j + 1 or i == j - 1, "Target orbitals must be adjacent."
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.array([[c, s], [-s, c]])
    return apply_orbital_rotation_adjacent(
        mat, vec, target_orbitals, n_orbitals=n_orbitals, n_electrons=n_electrons
    )


def apply_tunneling_interaction_adjacent(
    theta: float,
    vec: np.ndarray,
    target_orbitals: tuple[int, int],
    n_orbitals: int,
    n_electrons: tuple[int, int],
):
    r"""Apply a tunneling interaction gate to adjacent orbitals.

    The tunneling interaction gate is

    .. math::
        T(\theta) = \exp(i \theta (a^\dagger_i a_j + a\dagger_j a_i))

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbitals: The orbitals (i, j) on which to apply the interaction.
        n_orbitals: Number of spatial orbitals.
        n_electrons: Number of alpha and beta electrons.
    """
    i, j = target_orbitals
    assert i == j + 1 or i == j - 1, "Target orbitals must be adjacent."
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.array([[c, 1j * s], [1j * s, c]])
    return apply_orbital_rotation_adjacent(
        mat, vec, target_orbitals, n_orbitals=n_orbitals, n_electrons=n_electrons
    )


def apply_num_interaction(
    theta: float,
    vec: np.ndarray,
    target_orbital: int,
    n_orbitals: int,
    n_electrons: tuple[int, int],
    copy: bool = True,
):
    r"""Apply a number interaction gate.

    The number interaction gate is

    .. math::
        N(\theta) = \exp(i \theta a^\dagger_i a_i)

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbital: The orbital on which to apply the interaction.
        n_orbitals: Number of spatial orbitals.
        n_electrons: Number of alpha and beta electrons.
        copy: Whether to copy the input vector. If False, the operation is applied
            in-place.
    """
    if copy:
        vec = vec.copy()
    vec = apply_phase_shift(
        np.exp(1j * theta),
        vec,
        ((target_orbital,), ()),
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
        copy=False,
    )
    vec = apply_phase_shift(
        np.exp(1j * theta),
        vec,
        ((), (target_orbital,)),
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
        copy=False,
    )
    return vec


def apply_num_num_interaction(
    theta: float,
    vec: np.ndarray,
    target_orbitals: tuple[tuple[int, bool], tuple[int, bool]],
    n_orbitals: int,
    n_electrons: tuple[int, int],
    copy: bool = True,
):
    r"""Apply a number-number interaction gate.

    The number-number interaction gate is

    .. math::
        NN(\theta) = \exp(i \theta a^\dagger_{i, \sigma} a_{i, \sigma} a^\dagger_{j, \tau} a_{j, \tau})

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbitals: The orbitals on which to apply the interaction. This should
            be a pair of (orbital, spin) pairs.
        n_orbitals: Number of spatial orbitals.
        n_electrons: Number of alpha and beta electrons.
        copy: Whether to copy the input vector. If False, the operation is applied
            in-place.
    """
    if copy:
        vec = vec.copy()
    (i, spin_i), (j, spin_j) = target_orbitals
    orbitals = (set(), set())
    orbitals[spin_i].add(i)
    orbitals[spin_j].add(j)
    vec = apply_phase_shift(
        np.exp(1j * theta),
        vec,
        tuple(tuple(orbs) for orbs in orbitals),
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
        copy=False,
    )
    return vec
