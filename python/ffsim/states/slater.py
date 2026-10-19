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

import itertools
from collections.abc import Sequence
from typing import cast, overload

import numpy as np
import scipy.linalg
from pyscf.fci import cistring

from ffsim import linalg
from ffsim.gates.orbital_rotation import apply_orbital_rotation
from ffsim.states.bitstring import bitstring_to_occupied_orbitals
from ffsim.states.dimensions import dims


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
    alpha_index = cistring.str2addr(norb, n_alpha, sum(1 << i for i in alpha_orbitals))
    beta_index = cistring.str2addr(norb, n_beta, sum(1 << i for i in beta_orbitals))
    vec = linalg.one_hot(
        (dim1, dim2), (alpha_index, beta_index), dtype=complex
    ).reshape(-1)
    if orbital_rotation is not None:
        vec = apply_orbital_rotation(
            vec, orbital_rotation, norb=norb, nelec=nelec, copy=False
        )
    sign = _permutation_sign(np.argsort(alpha_orbitals)) * _permutation_sign(
        np.argsort(beta_orbitals)
    )
    return sign * vec


def _permutation_sign(perm: np.ndarray) -> int:
    """Compute the sign of a permutation.

    Reference: https://stackoverflow.com/a/73511014
    """
    assert set(perm) == set(range(len(perm)))
    parity = False
    perm = perm.copy()
    for i in range(len(perm)):
        while perm[i] != i:
            parity = not parity
            j = perm[i]
            perm[i], perm[j] = perm[j], perm[i]
    return -1 if parity else 1


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
    reorder: bool = True,
) -> np.ndarray | tuple[np.ndarray, ...]: ...
@overload
def slater_determinant_rdms(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    *,
    rank: int = 1,
    reorder: bool = True,
) -> np.ndarray | tuple[np.ndarray, ...]: ...
def slater_determinant_rdms(
    norb: int,
    occupied_orbitals: Sequence[int] | tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    *,
    rank: int = 1,
    reorder: bool = True,
) -> np.ndarray | tuple[np.ndarray, ...]:
    """Return the reduced density matrices of a Slater determinant.

    The rank 1 RDM is defined as follows:

    .. code::

        rdm1[p, q] = ⟨p+ q⟩

    The definition of higher-rank RDMs depends on the ``reorder`` argument, which
    defaults to True.

    **reorder = True**

    The reordered RDMs are defined as follows:

    .. code::

        rdm2[p, q, r, s] = ⟨p+ r+ s q⟩
        rdm3[p, q, r, s, t, u] = ⟨p+ r+ t+ u s q⟩
        rdm4[p, q, r, s, t, u, v, w] = ⟨p+ r+ t+ v+ w u s q⟩

    and so on.

    **reorder = False**

    If reorder is set to False, the RDMs are defined as follows:

    .. code::

        rdm2[p, q, r, s] = ⟨p+ q r+ s⟩
        rdm3[p, q, r, s, t, u] = ⟨p+ q r+ s t+ u⟩
        rdm4[p, q, r, s, t, u, v, w] = ⟨p+ q r+ s t+ u v+ w⟩

    and so on.

    Note:
        The higher-rank RDMs are computed from the 1-RDM using Wick's theorem, so
        any rank is supported. A rank-``k`` tensor stores ``norb ** (2*k)`` entries,
        and computing it takes about ``k`` tensor products of that size.

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
        rank: The highest rank of reduced density matrix to return. I.e., rank 1
            corresponds to the one-particle RDM, rank 2 corresponds to the
            2-particle RDM, etc.
        reorder: Whether to reorder the indices of the reduced density matrix.

    Returns:
        The reduced density matrices of the Slater determinant.
        All RDMs up to and including the specified rank are returned, in increasing
        order of rank. For example, if ``rank=2`` then a tuple ``(rdm1, rdm2)`` is
        returned. For ``rank=1``, the 1-RDM is returned directly.
        The representation of an RDM depends on whether ``occupied_orbitals`` is a
        sequence of integers (spinless case), or a pair of such sequences
        (spinful case). In the spinless case, the full RDM is returned, with shape
        ``(norb,) * (2*k)`` for rank ``k``.
        In the spinful case, each rank-``k`` RDM is represented as a stacked Numpy
        array of sub-RDMs with shape ``(k+1,) + (norb,) * (2*k)``. The leading axis
        selects the spin blocks, with ``a`` denoting alpha and ``b`` beta for each
        creation-annihilation pair: ``(a, b)`` for rank 1, ``(aa, ab, bb)`` for
        rank 2, ``(aaa, aab, abb, bbb)`` for rank 3, and so on.
    """
    if rank < 1:
        raise ValueError(f"The rank must be at least 1. Got {rank}.")

    if not occupied_orbitals or isinstance(occupied_orbitals[0], (int, np.integer)):
        # Spinless case
        occupied_orbitals = list(cast(Sequence[int], occupied_orbitals))
        rdm1 = np.zeros((norb, norb))
        if occupied_orbitals:
            rdm1[(occupied_orbitals, occupied_orbitals)] = 1
        if orbital_rotation is not None:
            orbital_rotation = cast(np.ndarray, orbital_rotation)
            rdm1 = orbital_rotation.conj() @ rdm1 @ orbital_rotation.T
        # The helper returns transposed views, so copy them into contiguous memory
        rdms = [
            np.ascontiguousarray(rdm)
            for rdm in _slater_determinant_rdms(rdm1, rank, reorder)
        ]
    else:
        # Spinful case
        alpha_orbitals, beta_orbitals = cast(
            tuple[Sequence[int], Sequence[int]], occupied_orbitals
        )
        alpha_orbitals = list(alpha_orbitals)
        beta_orbitals = list(beta_orbitals)
        rdm_a = np.zeros((norb, norb))
        rdm_b = np.zeros((norb, norb))
        if alpha_orbitals:
            rdm_a[(alpha_orbitals, alpha_orbitals)] = 1
        if beta_orbitals:
            rdm_b[(beta_orbitals, beta_orbitals)] = 1
        if orbital_rotation is not None:
            if isinstance(orbital_rotation, np.ndarray) and orbital_rotation.ndim == 2:
                orbital_rotation_a: np.ndarray | None = orbital_rotation
                orbital_rotation_b: np.ndarray | None = orbital_rotation
            else:
                orbital_rotation_a, orbital_rotation_b = orbital_rotation
            if orbital_rotation_a is not None:
                rdm_a = orbital_rotation_a.conj() @ rdm_a @ orbital_rotation_a.T
            if orbital_rotation_b is not None:
                rdm_b = orbital_rotation_b.conj() @ rdm_b @ orbital_rotation_b.T
        # Index k holds the rank-k RDM of each spin sector, with rank 0 equal to 1
        rdms_a = [np.ones(()), *_slater_determinant_rdms(rdm_a, rank, reorder)]
        if np.array_equal(rdm_a, rdm_b):
            # Closed-shell case: reuse the alpha RDMs for beta
            rdms_b = rdms_a
        else:
            rdms_b = [np.ones(()), *_slater_determinant_rdms(rdm_b, rank, reorder)]
        # There are no contractions between alpha and beta operators, so each spin
        # block is a product of an alpha RDM and a beta RDM
        rdms = [
            np.stack(
                [
                    np.multiply.outer(rdms_a[k - n_beta], rdms_b[n_beta])
                    for n_beta in range(k + 1)
                ]
            )
            for k in range(1, rank + 1)
        ]
    return rdms[0] if rank == 1 else tuple(rdms)


def _slater_determinant_rdms(
    rdm1: np.ndarray, rank: int, reorder: bool
) -> list[np.ndarray]:
    """Return the RDMs of a Slater determinant up to rank ``rank`` from its 1-RDM.

    By Wick's theorem, the RDM entry is the determinant of the matrix whose
    ``(i, j)`` entry is the contraction of the ``i``-th creation operator with the
    ``j``-th annihilation operator. The determinant is computed by Laplace expansion
    along its rows, building each minor from the minors one row smaller so that every
    minor is computed only once. The rank-``k`` RDM is the minor of the first ``k``
    rows and columns, so all ranks come from a single expansion.

    The returned arrays are transposed views and are not contiguous in general.
    """
    norb, _ = rdm1.shape
    # Contracting an annihilator with a later creator gives ⟨q p+⟩ = δ - ⟨p+ q⟩
    anti_rdm1 = rdm1 - np.eye(norb)
    # minors[cols] is the minor of the first len(cols) rows and the columns in cols.
    # Its axes are the creation indices of those rows followed by the annihilation
    # indices of those columns.
    minors: dict[tuple[int, ...], np.ndarray] = {(): np.ones((), dtype=rdm1.dtype)}
    rdms = []
    for i in range(rank):
        next_minors: dict[tuple[int, ...], np.ndarray] = {}
        for cols in itertools.combinations(range(rank), i + 1):
            minor = np.zeros((norb,) * (2 * i + 2), dtype=rdm1.dtype)
            for pos, j in enumerate(cols):
                # With reorder=False, the annihilator of pair j precedes the creator of
                # pair i if j < i
                mat = anti_rdm1 if not reorder and j < i else rdm1
                term = np.multiply.outer(minors[cols[:pos] + cols[pos + 1 :]], mat)
                # Move the new creation and annihilation axes into place
                term = np.moveaxis(term, [-2, -1], [i, i + 1 + pos])
                if (i + pos) % 2:
                    minor -= term
                else:
                    minor += term
            next_minors[cols] = minor
        minors = next_minors
        # Interleave the axes so that pair j has its creation index on axis 2*j and
        # annihilation index on axis 2*j + 1
        k = i + 1
        axes = [axis for j in range(k) for axis in (j, k + j)]
        rdms.append(minors[tuple(range(k))].transpose(axes))
    return rdms


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
