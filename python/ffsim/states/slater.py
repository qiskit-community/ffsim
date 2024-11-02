# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for creating and manipulating Slater determinants."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, overload

import numpy as np
import scipy.linalg
from pyscf.fci import cistring

from ffsim import linalg
from ffsim.gates.orbital_rotation import apply_orbital_rotation
from ffsim.states.bitstring import (
    bitstring_to_occupied_orbitals,
)
from ffsim.states.states import dims


@overload
def slater_determinant(
    norb: int,
    occupied_orbitals: Sequence[int],
    orbital_rotation: np.ndarray | None = None,
) -> np.ndarray: ...
@overload
def slater_determinant(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
) -> np.ndarray: ...
def slater_determinant(
    norb: int,
    occupied_orbitals: Sequence[int] | tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
) -> np.ndarray:
    r"""Return a Slater determinant.

    A Slater determinant is a state of the form

    .. math::

        \mathcal{U} \lvert x \rangle,

    where :math:`\mathcal{U}` is an
    :doc:`orbital rotation </explanations/orbital-rotation>` and
    :math:`\lvert x \rangle` is an electronic configuration.

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: The occupied orbitals in the electronic configuration.
            This is either a list of integers specifying spinless orbitals, or a
            pair of lists, where the first list specifies the spin alpha orbitals and
            the second list specifies the spin beta orbitals.
        orbital_rotation: The optional orbital rotation.
            You can pass either a single Numpy array specifying the orbital rotation
            to apply to both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent orbital rotations for spin alpha and spin beta.
            If passing a pair, you can use ``None`` for one of the
            values in the pair to indicate that no operation should be applied to
            that spin sector.

    Returns:
        The Slater determinant as a state vector.
    """
    if norb == 0:
        return np.ones(1, dtype=complex)

    if not occupied_orbitals or isinstance(occupied_orbitals[0], (int, np.integer)):
        occupied_orbitals = (cast(Sequence[int], occupied_orbitals), [])

    alpha_orbitals, beta_orbitals = cast(
        tuple[Sequence[int], Sequence[int]], occupied_orbitals
    )
    n_alpha = len(alpha_orbitals)
    n_beta = len(beta_orbitals)
    nelec = (n_alpha, n_beta)
    dim1, dim2 = dims(norb, nelec)
    alpha_bits = np.zeros(norb, dtype=bool)
    alpha_bits[list(alpha_orbitals)] = 1
    alpha_string = int("".join("1" if b else "0" for b in alpha_bits[::-1]), base=2)
    alpha_index = cistring.str2addr(norb, n_alpha, alpha_string)
    beta_bits = np.zeros(norb, dtype=bool)
    beta_bits[list(beta_orbitals)] = 1
    beta_string = int("".join("1" if b else "0" for b in beta_bits[::-1]), base=2)
    beta_index = cistring.str2addr(norb, n_beta, beta_string)
    vec = linalg.one_hot(
        (dim1, dim2), (alpha_index, beta_index), dtype=complex
    ).reshape(-1)
    if orbital_rotation is not None:
        vec = apply_orbital_rotation(
            vec, orbital_rotation, norb=norb, nelec=nelec, copy=False
        )
    return vec


def hartree_fock_state(norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
    """Return the Hartree-Fock state.

    Args:
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.

    Returns:
        The Hartree-Fock state as a state vector.
    """
    if isinstance(nelec, int):
        return slater_determinant(norb, occupied_orbitals=range(nelec))

    n_alpha, n_beta = nelec
    return slater_determinant(norb, occupied_orbitals=(range(n_alpha), range(n_beta)))


@overload
def slater_determinant_rdms(
    norb: int,
    occupied_orbitals: Sequence[int],
    orbital_rotation: np.ndarray | None = None,
    *,
    rank: int = 1,
) -> np.ndarray: ...
@overload
def slater_determinant_rdms(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    *,
    rank: int = 1,
) -> np.ndarray: ...
def slater_determinant_rdms(
    norb: int,
    occupied_orbitals: Sequence[int] | tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    *,
    rank: int = 1,
) -> np.ndarray:
    """Return the reduced density matrices of a `Slater determinant`_.

    Note:
        Currently, only rank 1 is supported.

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: The occupied orbitals in the electronic configuration.
            This is either a list of integers specifying spinless orbitals, or a
            pair of lists, where the first list specifies the spin alpha orbitals and
            the second list specifies the spin beta orbitals.
        orbital_rotation: The optional orbital rotation.
            You can pass either a single Numpy array specifying the orbital rotation
            to apply to both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent orbital rotations for spin alpha and spin beta.
            If passing a pair, you can use ``None`` for one of the
            values in the pair to indicate that no operation should be applied to
            that spin sector.
        rank: The rank of the reduced density matrix. I.e., rank 1 corresponds to the
            one-particle RDM, rank 2 corresponds to the 2-particle RDM, etc.

    Returns:
        The reduced density matrices of the Slater determinant.
        All RDMs up to and including the specified rank are returned, in increasing
        order of rank. For example, if `rank=2` then a tuple `(rdm1, rdm2)` is returned.
        The representation of an RDM depends on whether `occupied_orbitals` is a
        sequence of integers (spinless case), or a pair of such sequences
        (spinful case). In the spinless case, the full RDM is returned.
        In the spinful case, each RDM is represented as a stacked Numpy
        array of sub-RDMs. For example, the 1-RDMs are: (alpha-alpha, alpha-beta), and
        the 2-RDMs are: (alpha-alpha, alpha-beta, beta-beta).

    .. _Slater determinant: ffsim.html#ffsim.slater_determinant
    """
    if not occupied_orbitals or isinstance(occupied_orbitals[0], (int, np.integer)):
        # Spinless case
        occupied_orbitals = list(cast(Sequence[int], occupied_orbitals))
        if rank == 1:
            rdm = np.zeros((norb, norb), dtype=complex)
            if occupied_orbitals:
                rdm[(occupied_orbitals, occupied_orbitals)] = 1
            if orbital_rotation is not None:
                orbital_rotation = cast(np.ndarray, orbital_rotation)
                rdm = orbital_rotation.conj() @ rdm @ orbital_rotation.T
            return rdm
        raise NotImplementedError(
            f"Returning the rank {rank} reduced density matrix is currently not "
            "supported."
        )
    else:
        # Spinful case
        alpha_orbitals, beta_orbitals = cast(
            tuple[Sequence[int], Sequence[int]], occupied_orbitals
        )
        alpha_orbitals = list(alpha_orbitals)
        beta_orbitals = list(beta_orbitals)
        if rank == 1:
            rdm_a = np.zeros((norb, norb))
            rdm_b = np.zeros((norb, norb))
            if alpha_orbitals:
                rdm_a[(alpha_orbitals, alpha_orbitals)] = 1
            if beta_orbitals:
                rdm_b[(beta_orbitals, beta_orbitals)] = 1
            if orbital_rotation is not None:
                if (
                    isinstance(orbital_rotation, np.ndarray)
                    and orbital_rotation.ndim == 2
                ):
                    orbital_rotation_a: np.ndarray | None = orbital_rotation
                    orbital_rotation_b: np.ndarray | None = orbital_rotation
                else:
                    orbital_rotation_a, orbital_rotation_b = orbital_rotation
                if orbital_rotation_a is not None:
                    rdm_a = orbital_rotation_a.conj() @ rdm_a @ orbital_rotation_a.T
                if orbital_rotation_b is not None:
                    rdm_b = orbital_rotation_b.conj() @ rdm_b @ orbital_rotation_b.T
            return np.stack([rdm_a, rdm_b])
        raise NotImplementedError(
            f"Returning the rank {rank} reduced density matrix is currently not "
            "supported."
        )


def slater_determinant_amplitudes(
    bitstrings: Sequence[int] | tuple[Sequence[int], Sequence[int]],
    norb: int,
    occupied_orbitals: Sequence[int] | tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray | tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Compute state vector amplitudes for a Slater determinant.

    Args:
        bitstrings: The bitstrings to return the amplitudes for, in integer
            representation. In the spinless case this is a list of integers. In the
            spinful case, this is a pair of lists of equal length specifying the
            alpha and beta parts of the bitstrings.
        norb: The number of spatial orbitals.
        occupied_orbitals: The occupied orbitals in the electronic configuration.
            This is either a list of integers specifying spinless orbitals, or a
            pair of lists, where the first list specifies the spin alpha orbitals and
            the second list specifies the spin beta orbitals.
        orbital_rotation: The orbital rotation describing the Slater determinant.
            You can pass either a single Numpy array specifying the orbital rotation
            to apply to both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent orbital rotations for spin alpha and spin beta.

    Returns:
        The amplitudes of the requested bitstrings.
    """
    if not occupied_orbitals or isinstance(occupied_orbitals[0], (int, np.integer)):
        # Spinless case
        vecs = cast(np.ndarray, orbital_rotation)[:, occupied_orbitals]
        amplitudes = []
        for bitstring in cast(Sequence[int], bitstrings):
            orbs = bitstring_to_occupied_orbitals(bitstring)
            amplitudes.append(scipy.linalg.det(vecs[orbs]))
        return np.array(amplitudes)

    # Spinful case
    occupied_orbitals_a, occupied_orbitals_b = cast(
        tuple[Sequence[int], Sequence[int]], occupied_orbitals
    )
    bitstrings_a, bitstrings_b = cast(tuple[Sequence[int], Sequence[int]], bitstrings)
    if isinstance(orbital_rotation, np.ndarray) and orbital_rotation.ndim == 2:
        orbital_rotation_a = orbital_rotation
        orbital_rotation_b = orbital_rotation
    else:
        orbital_rotation_a, orbital_rotation_b = orbital_rotation
    amplitudes_a = slater_determinant_amplitudes(
        bitstrings_a, norb, occupied_orbitals_a, orbital_rotation_a
    )
    amplitudes_b = slater_determinant_amplitudes(
        bitstrings_b, norb, occupied_orbitals_b, orbital_rotation_b
    )
    return amplitudes_a * amplitudes_b
