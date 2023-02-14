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
    mat: np.ndarray, vec: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    """Apply an orbital rotation to a vector.

    An orbital rotation is also known as a single-particle basis change.

    Args:
        mat: Unitary matrix describing the orbital rotation.
        vec: Vector to be transformed.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
    """
    givens_rotations, phase_shifts = givens_decomposition(mat)
    buf1 = np.empty_like(vec)
    buf2 = np.empty_like(vec)
    buf1[...] = vec[...]
    for givens_mat, target_orbs in givens_rotations:
        vec = apply_orbital_rotation_adjacent(
            givens_mat.conj(), buf1, target_orbs, norb, nelec, out=buf2
        )
        buf1, buf2 = buf2, buf1
    for i, phase_shift in enumerate(phase_shifts):
        apply_phase_shift(phase_shift, vec, ((i,), ()), norb, nelec, copy=False)
        apply_phase_shift(phase_shift, vec, ((), (i,)), norb, nelec, copy=False)
    return vec


def apply_orbital_rotation_adjacent(
    mat: np.ndarray,
    vec: np.ndarray,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
    *,
    out: np.ndarray | None = None,
):
    """Apply an orbital rotation to adjacent orbitals.

    Args:
        mat: A 2x2 unitary matrix describing the orbital rotation.
        vec: Vector to be transformed.
        target_orbs: The orbitals to transform.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
    """
    if out is vec:
        raise ValueError("Output buffer cannot be the same as the input")
    if out is None:
        out = np.empty_like(vec)
    i, j = target_orbs
    assert i == j + 1 or i == j - 1, "Target orbitals must be adjacent."
    n_alpha, n_beta = nelec
    dim_a = comb(norb, n_alpha, exact=True)
    dim_b = comb(norb, n_beta, exact=True)
    vec = vec.reshape((dim_a, dim_b))
    out = out.reshape((dim_a, dim_b))
    indices = _zero_one_subspace_indices(norb, n_alpha, target_orbs)
    slice1 = indices[: len(indices) // 2]
    slice2 = indices[len(indices) // 2 :]
    buf = np.empty_like(vec)
    apply_matrix_to_slices(mat, vec, [slice1, slice2], out=buf)
    indices = _zero_one_subspace_indices(norb, n_beta, target_orbs)
    slice1 = indices[: len(indices) // 2]
    slice2 = indices[len(indices) // 2 :]
    apply_matrix_to_slices(mat, buf, [(Ellipsis, slice1), (Ellipsis, slice2)], out=out)
    return out.reshape((dim_a * dim_b,))


@lru_cache(maxsize=None)
def _zero_one_subspace_indices(norb: int, nocc: int, target_orbs: tuple[int, int]):
    """Return the indices where the target orbitals are 01 or 10."""
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n00 = comb(norb - 2, nocc, exact=True)
    n11 = comb(norb - 2, nocc - 2, exact=True)
    return indices[n00 : len(indices) - n11]


def apply_phase_shift(
    phase: complex,
    vec: np.ndarray,
    target_orbs: tuple[tuple[int], tuple[int]],
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
):
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
    return vec.reshape((dim_a * dim_b,))


@lru_cache(maxsize=None)
def _one_subspace_indices(
    norb: int, nocc: int, target_orbs: tuple[tuple[int], tuple[int]]
):
    """Return the indices where the target orbitals are 1."""
    orbitals = _shifted_orbitals(norb, target_orbs)
    strings = cistring.make_strings(orbitals, nocc)
    indices = np.argsort(strings)
    n0 = comb(norb, nocc, exact=True) - comb(
        norb - len(target_orbs), nocc - len(target_orbs), exact=True
    )
    return indices[n0:]


def _shifted_orbitals(norb: int, target_orbs: tuple[int]):
    """Return orbital list with targeted orbitals shifted to the end."""
    orbitals = np.arange(norb - len(target_orbs))
    values = sorted(zip(target_orbs, range(norb - len(target_orbs), norb)))
    for index, val in values:
        orbitals = np.insert(orbitals, index, val)
    return orbitals


def apply_num_op_sum_evolution(
    coeffs: np.ndarray,
    vec: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
):
    """Apply a sum of number operators to a vector."""
    if copy:
        vec = vec.copy()
    for i, coeff in enumerate(coeffs):
        vec = apply_num_interaction(
            -coeff * time,
            vec,
            i,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
    return vec


def apply_core_tensor_evolution(
    core_tensor: np.ndarray,
    vec: np.ndarray,
    time: float,
    norb: int,
    nelec: tuple[int, int],
    *,
    core_tensor_alpha_beta: np.ndarray | None = None,
    copy: bool = True,
) -> np.ndarray:
    # TODO norb does not need to be input here and some other places
    """Apply core tensor evolution."""
    if copy:
        vec = vec.copy()
    norb, _ = core_tensor.shape
    if core_tensor_alpha_beta is None:
        core_tensor_alpha_beta = core_tensor
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        coeff = 0.5 if i == j else 1.0
        for sigma in range(2):
            vec = apply_num_num_interaction(
                -coeff * time * core_tensor[i, j],
                vec,
                ((i, sigma), (j, sigma)),
                norb=norb,
                nelec=nelec,
                copy=False,
            )
            vec = apply_num_num_interaction(
                -coeff * time * core_tensor_alpha_beta[i, j],
                vec,
                ((i, sigma), (j, 1 - sigma)),
                norb=norb,
                nelec=nelec,
                copy=False,
            )
    return vec


def apply_givens_rotation(
    theta: float,
    vec: np.ndarray,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
):
    r"""Apply a Givens rotation gate.

    The Givens rotation gate is

    .. math::
        G(\theta) = \exp(\theta (a^\dagger_i a_j - a\dagger_j a_i))

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbs: The orbitals (i, j) to rotate.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.eye(norb)
    mat[np.ix_(target_orbs, target_orbs)] = [[c, s], [-s, c]]
    return apply_orbital_rotation(mat, vec, norb=norb, nelec=nelec)


def apply_tunneling_interaction(
    theta: float,
    vec: np.ndarray,
    target_orbs: tuple[int, int],
    norb: int,
    nelec: tuple[int, int],
):
    r"""Apply a tunneling interaction gate.

    The tunneling interaction gate is

    .. math::
        T(\theta) = \exp(i \theta (a^\dagger_i a_j + a\dagger_j a_i))

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbs: The orbitals (i, j) on which to apply the interaction.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
    """
    if len(set(target_orbs)) == 1:
        raise ValueError(f"The orbitals to rotate must be distinct. Got {target_orbs}.")
    vec = apply_num_interaction(
        -np.pi / 2,
        vec,
        target_orbs[0],
        norb=norb,
        nelec=nelec,
    )
    vec = apply_givens_rotation(theta, vec, target_orbs, norb=norb, nelec=nelec)
    vec = apply_num_interaction(
        np.pi / 2,
        vec,
        target_orbs[0],
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec


def apply_num_interaction(
    theta: float,
    vec: np.ndarray,
    target_orb: int,
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
):
    r"""Apply a number interaction gate.

    The number interaction gate is

    .. math::
        N(\theta) = \exp(i \theta a^\dagger_i a_i)

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orb: The orbital on which to apply the interaction.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
        copy: Whether to copy the input vector. If False, the operation is applied
            in-place.
    """
    if copy:
        vec = vec.copy()
    vec = apply_phase_shift(
        np.exp(1j * theta),
        vec,
        ((target_orb,), ()),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    vec = apply_phase_shift(
        np.exp(1j * theta),
        vec,
        ((), (target_orb,)),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec


def apply_num_num_interaction(
    theta: float,
    vec: np.ndarray,
    target_orbs: tuple[tuple[int, bool], tuple[int, bool]],
    norb: int,
    nelec: tuple[int, int],
    copy: bool = True,
):
    r"""Apply a number-number interaction gate.

    The number-number interaction gate is

    .. math::
        NN(\theta) = \exp(i \theta a^\dagger_{i, \sigma} a_{i, \sigma} a^\dagger_{j, \tau} a_{j, \tau})

    Args:
        theta: The rotation angle.
        vec: Vector to be transformed.
        target_orbs: The orbitals on which to apply the interaction. This should
            be a pair of (orbital, spin) pairs.
        norb: Number of spatial orbitals.
        nelec: Number of alpha and beta electrons.
        copy: Whether to copy the input vector. If False, the operation is applied
            in-place.
    """
    if copy:
        vec = vec.copy()
    (i, spin_i), (j, spin_j) = target_orbs
    orbitals = (set(), set())
    orbitals[spin_i].add(i)
    orbitals[spin_j].add(j)
    vec = apply_phase_shift(
        np.exp(1j * theta),
        vec,
        tuple(tuple(orbs) for orbs in orbitals),
        norb=norb,
        nelec=nelec,
        copy=False,
    )
    return vec
