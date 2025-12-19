# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Function to sample bitstrings from a Slater determinant."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import cast, overload

import numpy as np
import scipy.linalg
from typing_extensions import deprecated

from ffsim.states.bitstring import (
    BitstringType,
    concatenate_bitstrings,
    convert_bitstring_type,
    restrict_bitstrings,
)


@overload
def sample_slater(
    norb: int,
    occupied_orbitals: Sequence[int],
    orbital_rotation: np.ndarray | None = None,
    *,
    orbs: Sequence[int] | None = None,
    shots: int = 1,
    bitstring_type: BitstringType = BitstringType.STRING,
    seed: np.random.Generator | int | None = None,
) -> Sequence[int] | Sequence[str] | np.ndarray: ...
@overload
def sample_slater(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    *,
    orbs: tuple[Sequence[int], Sequence[int]] | None = None,
    shots: int = 1,
    concatenate: bool = True,
    bitstring_type: BitstringType = BitstringType.STRING,
    seed: np.random.Generator | int | None = None,
) -> Sequence[int] | Sequence[str] | np.ndarray: ...
def sample_slater(
    norb: int,
    occupied_orbitals: Sequence[int] | tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    *,
    orbs: Sequence[int] | tuple[Sequence[int], Sequence[int]] | None = None,
    shots: int = 1,
    concatenate: bool = True,
    bitstring_type: BitstringType = BitstringType.STRING,
    seed: np.random.Generator | int | None = None,
) -> Sequence[int] | Sequence[str] | np.ndarray:
    """Collect samples of electronic configurations from a Slater determinant.

    The Slater determinant is specified as an orbital rotation applied to the
    reference electronic configuration given by `occupied_orbitals`. The sampler
    draws independent samples from the corresponding determinantal point process
    using the projection-based sequential sampling algorithm of *Fermion Sampling
    Made More Efficient* (Phys. Rev. B 107, 035119, 2023).

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: The occupied orbitals in the electronic configuration.
            This is either a list of integers specifying spinless orbitals, or a
            pair of lists, where the first list specifies the spin alpha orbitals and
            the second list specifies the spin beta orbitals.
        orbital_rotation: The optional orbital rotation.
            Either a single Numpy array specifying the orbital rotation to apply to
            both spin sectors, or a pair of Numpy arrays specifying independent orbital
            rotations for spin alpha and spin beta. In the paired form, ``None``
            indicates that the identity operation is applied to the corresponding spin
            sector.
        orbs: The orbitals to sample.
            In the spinless case, this is a list of integers in ``range(norb)``.
            In the spinful case, this is a pair of such lists, where the first list
            stores spin-alpha orbitals and the second list stores spin-beta orbitals.
            If not specified, then all orbitals are sampled.
        shots: The number of bitstrings to sample.
        concatenate: Whether to concatenate the spin-alpha and spin-beta parts of the
            bitstrings. If True, then a single list of concatenated bitstrings is
            returned. The strings are concatenated in the order :math:`s_b s_a`,
            that is, the alpha string appears on the right.
            If False, then two lists are returned, ``(strings_a, strings_b)``, with the
            alpha strings listed first.
            In the spinless case (when `occupied_orbitals` is a sequence of integers),
            this argument is ignored.
        bitstring_type: The desired type of bitstring output.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled bitstrings.
    """
    rng = np.random.default_rng(seed)

    if not occupied_orbitals or isinstance(occupied_orbitals[0], (int, np.integer)):
        # Spinless case
        occupied_orbitals = cast(Sequence[int], occupied_orbitals)
        nelec = len(occupied_orbitals)
        if orbs is None:
            orbs = range(norb)
        orbs = cast(Sequence[int], orbs)
        if nelec == 0:
            strings = [0] * shots
        elif nelec == norb:
            strings = [(1 << norb) - 1] * shots
        else:
            if orbital_rotation is None:
                orbitals = np.eye(norb, dtype=complex)[:, occupied_orbitals]
            else:
                orbital_rotation = cast(np.ndarray, orbital_rotation)
                orbitals = orbital_rotation.conj()[:, occupied_orbitals]
            normals = _compute_projection_normals(orbitals)
            strings = [
                _sample_from_projection_normals(normals, rng) for _ in range(shots)
            ]
        strings = restrict_bitstrings(strings, orbs, bitstring_type=BitstringType.INT)
        return convert_bitstring_type(
            strings,
            BitstringType.INT,
            bitstring_type,
            length=len(orbs),
        )

    # Spinful case
    orbs = cast(tuple[Sequence[int], Sequence[int]] | None, orbs)
    if orbs is None:
        orbs = (range(norb), range(norb))
    orbs_a, orbs_b = cast(tuple[Sequence[int], Sequence[int]], orbs)
    orbs_a = cast(Sequence[int], orbs_a)
    orbs_b = cast(Sequence[int], orbs_b)

    occupied_orbitals_a, occupied_orbitals_b = cast(
        tuple[Sequence[int], Sequence[int]], occupied_orbitals
    )
    n_a = len(occupied_orbitals_a)
    n_b = len(occupied_orbitals_b)

    if orbital_rotation is None:
        orbital_rotation_a = None
        orbital_rotation_b = None
    elif isinstance(orbital_rotation, np.ndarray) and orbital_rotation.ndim == 2:
        orbital_rotation_a = orbital_rotation
        orbital_rotation_b = orbital_rotation
    else:
        orbital_rotation_a, orbital_rotation_b = cast(
            tuple[np.ndarray | None, np.ndarray | None], orbital_rotation
        )

    if n_a == 0:
        strings_a = [0] * shots
    elif n_a == norb:
        strings_a = [(1 << norb) - 1] * shots
    else:
        if orbital_rotation_a is None:
            orbitals_a = np.eye(norb, dtype=complex)[:, occupied_orbitals_a]
        else:
            orbitals_a = orbital_rotation_a.conj()[:, occupied_orbitals_a]
        normals_a = _compute_projection_normals(orbitals_a)
        strings_a = [
            _sample_from_projection_normals(normals_a, rng) for _ in range(shots)
        ]

    if n_b == 0:
        strings_b = [0] * shots
    elif n_b == norb:
        strings_b = [(1 << norb) - 1] * shots
    else:
        if orbital_rotation_b is None:
            orbitals_b = np.eye(norb, dtype=complex)[:, occupied_orbitals_b]
        else:
            orbitals_b = orbital_rotation_b.conj()[:, occupied_orbitals_b]
        normals_b = _compute_projection_normals(orbitals_b)
        strings_b = [
            _sample_from_projection_normals(normals_b, rng) for _ in range(shots)
        ]

    strings_a = restrict_bitstrings(strings_a, orbs_a, bitstring_type=BitstringType.INT)
    strings_b = restrict_bitstrings(strings_b, orbs_b, bitstring_type=BitstringType.INT)

    if concatenate:
        strings = concatenate_bitstrings(
            strings_a,
            strings_b,
            BitstringType.INT,
            length=len(orbs_a),
        )
        return convert_bitstring_type(
            strings,
            BitstringType.INT,
            bitstring_type,
            length=len(orbs_a) + len(orbs_b),
        )

    return convert_bitstring_type(
        strings_a,
        BitstringType.INT,
        bitstring_type,
        length=len(orbs_a),
    ), convert_bitstring_type(
        strings_b,
        BitstringType.INT,
        bitstring_type,
        length=len(orbs_b),
    )


@deprecated(
    "ffsim.sample_slater_determinant is deprecated. Instead, use ffsim.sample_slater."
)
def sample_slater_determinant(
    rdm: np.ndarray | tuple[np.ndarray, np.ndarray],
    norb: int,
    nelec: int | tuple[int, int],
    *,
    orbs: Sequence[int] | tuple[Sequence[int], Sequence[int]] | None = None,
    shots: int = 1,
    concatenate: bool = True,
    bitstring_type: BitstringType = BitstringType.STRING,
    seed: np.random.Generator | int | None = None,
) -> Sequence[int] | Sequence[str] | np.ndarray:
    """Collect samples of electronic configurations from a Slater determinant.

    .. warning::
        This function is deprecated. Use :func:`ffsim.sample_slater` instead.

    The Slater determinant is defined by its one-body reduced density matrix (RDM).
    The sampler uses a determinantal point process to auto-regressively produce
    uncorrelated samples.

    This sampling strategy is known as
    `determinantal point processes <https://arxiv.org/abs/1207.6083>`

    Args:
        rdm: The one-body reduced density matrix that defines the Slater determinant
            This is either a single Numpy array specifying the 1-RDM of a
            spin-polarized system, or a pair of Numpy arrays where each element
            of the pair contains the 1-RDM for each spin sector.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        shots: The number of bitstrings to sample.
        concatenate: Whether to concatenate the spin-alpha and spin-beta parts of the
            bitstrings. If True, then a single list of concatenated bitstrings is
            returned. The strings are concatenated in the order :math:`s_b s_a`,
            that is, the alpha string appears on the right.
            If False, then two lists are returned, ``(strings_a, strings_b)``. Note that
            the list of alpha strings appears first, that is, on the left.
            In the spinless case (when `nelec` is an integer), this argument is ignored.
        bitstring_type: The desired type of bitstring output.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        A 2D Numpy array with samples of electronic configurations.
        Each row is a sample.
    """
    rng = np.random.default_rng(seed)

    if isinstance(nelec, int):
        # Spinless case
        rdm = cast(np.ndarray, rdm)
        norb, _ = rdm.shape
        if orbs is None:
            orbs = range(norb)
        orbs = cast(Sequence[int], orbs)
        strings = _sample_slater_spinless(rdm, nelec, shots, rng)
        strings = restrict_bitstrings(strings, orbs, bitstring_type=BitstringType.INT)
        return convert_bitstring_type(
            strings,
            BitstringType.INT,
            bitstring_type,
            length=len(orbs),
        )

    # Spinful case
    rdm_a, rdm_b = rdm
    n_a, n_b = nelec
    norb, _ = rdm_a.shape
    if orbs is None:
        orbs = (range(norb), range(norb))
    orbs_a, orbs_b = orbs
    orbs_a = cast(Sequence[int], orbs_a)
    orbs_b = cast(Sequence[int], orbs_b)
    strings_a = _sample_slater_spinless(rdm_a, n_a, shots, rng)
    strings_b = _sample_slater_spinless(rdm_b, n_b, shots, rng)
    strings_a = restrict_bitstrings(strings_a, orbs_a, bitstring_type=BitstringType.INT)
    strings_b = restrict_bitstrings(strings_b, orbs_b, bitstring_type=BitstringType.INT)

    if concatenate:
        strings = concatenate_bitstrings(
            strings_a,
            strings_b,
            BitstringType.INT,
            length=len(orbs_a),
        )
        return convert_bitstring_type(
            strings,
            BitstringType.INT,
            bitstring_type,
            length=len(orbs_a) + len(orbs_b),
        )

    return convert_bitstring_type(
        strings_a,
        BitstringType.INT,
        bitstring_type,
        length=len(orbs_a),
    ), convert_bitstring_type(
        strings_b,
        BitstringType.INT,
        bitstring_type,
        length=len(orbs_b),
    )


def _sample_slater_spinless(
    rdm: np.ndarray,
    nelec: int,
    shots: int,
    seed: np.random.Generator | int | None = None,
) -> list[int]:
    """Collect samples of electronic configurations from a Slater determinant.

    The Slater determinant is defined by its one-body reduced density matrix (RDM).
    The sampler uses a determinantal point process to auto-regressively produce
    uncorrelated samples.

    Args:
        rdm: The one-body reduced density matrix that defines the Slater determinant.
        shots: Number of samples to collect.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled bitstrings in integer format.
    """
    norb, _ = rdm.shape

    if nelec == 0:
        return [0] * shots

    if nelec == norb:
        return [(1 << norb) - 1] * shots

    rng = np.random.default_rng(seed)
    samples = [_autoregressive_slater(rdm, norb, nelec, rng) for _ in range(shots)]
    return [sum(1 << orb for orb in sample) for sample in samples]


def _autoregressive_slater(
    rdm: np.ndarray,
    norb: int,
    nelec: int,
    seed: np.random.Generator | int | None = None,
) -> set[int]:
    """Autoregressively sample occupied orbitals for a Slater determinant.

    Args:
        rdm: The one-body reduced density matrix.
        norb: The number of orbitals.
        nelec: The number of electrons.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        A set containing the occupied orbitals.
    """
    rng = np.random.default_rng(seed)
    orb = rng.choice(norb, p=np.diag(rdm).real / nelec)
    sample = {orb}
    for i in range(nelec - 1):
        index = rng.choice(norb - 1 - i, p=_generate_probs(rdm, sample, norb))
        empty_orbitals = (orb for orb in range(norb) if orb not in sample)
        sample.add(next(itertools.islice(empty_orbitals, index, None)))
    return sample


def _generate_probs(rdm: np.ndarray, sample: set[int], norb: int) -> np.ndarray:
    """Computes the probabilities for the next occupied orbital.

    This is a step of the autoregressive sampling, and uses Bayes's rule.

    Args:
        rdm: A Numpy array with the one-body reduced density matrix.
        sample: The list of already occupied orbitals.
        norb: The number of orbitals.

    Returns:
        The probabilities for the next empty orbital to occupy.
    """
    probs = np.zeros(norb - len(sample))
    empty_orbitals = (orb for orb in range(norb) if orb not in sample)
    for i, orb in enumerate(empty_orbitals):
        indices = list(sample | {orb})
        probs[i] = np.linalg.det(rdm[indices][:, indices]).real
    return probs / np.sum(probs)


def _compute_projection_normals(orbitals: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute the normal vectors needed for fast projection sampling.

    Given an orthonormal orbital matrix for a projector 1-RDM, precomputes dual
    normal vectors via reduced QR factorization and triangular solves for reuse
    during sampling.

    Args:
        orbitals: A 2D array whose columns are orthonormal orbitals spanning the
            occupied subspace. Shape ``(norb, nelec)``.
        tol: Tolerance for detecting ill-conditioned QR factorizations.

    Returns:
        A 2D array of normal vectors with shape ``(norb, nelec)``.

    Raises:
        ValueError: If the input is not 2D, has zero occupied orbitals, or is
            ill-conditioned for the QR-based precomputation.
    """
    if orbitals.ndim != 2:
        raise ValueError("Projector orbitals must be a 2D array.")
    _, nelec = orbitals.shape
    if nelec == 0:
        raise ValueError("Fast sampler requires at least one occupied orbital.")
    q, r = np.linalg.qr(orbitals, mode="reduced")
    diag = np.abs(np.diag(r))
    if np.any(diag <= tol):
        raise ValueError(
            "Projector factorization is ill-conditioned for fast sampling."
        )
    inv_rt = scipy.linalg.solve_triangular(
        r.T, np.eye(nelec), lower=True, check_finite=False
    )
    return q @ inv_rt


def _sample_from_projection_normals(
    normals: np.ndarray, rng: np.random.Generator, tol: float = 1e-12
) -> int:
    """Draw one configuration using cached projection normals and supplied RNG.

    Permutes the precomputed normal vectors, then iteratively picks an orbital
    with probability proportional to the squared magnitude of the leading normal,
    performs a Gaussian-elimination pivot to update the remaining normals, and
    repeats until all electrons are placed.

    Args:
        normals: A 2D array of precomputed normal vectors with shape ``(norb, nelec)``.
        rng: NumPy Generator to use for randomness.
        tol: Tolerance for detecting pivot breakdown in elimination.

    Returns:
        Integer-encoded bitstring for the sampled configuration.
    """
    norb, nelec = normals.shape
    perm = rng.permutation(nelec)
    # Working set of normals after applying the random permutation.
    active_normals = normals[:, perm].copy()
    selected: list[int] = []
    active = nelec
    # Iterate until all electrons are placed
    while active:
        # Pivot column used for sampling and elimination.
        pivot_normal = active_normals[:, 0]
        row_norms = np.abs(pivot_normal) ** 2
        total = float(np.sum(row_norms))
        if total <= 0:
            raise ValueError("Failed to compute sampling probabilities.")
        probs = row_norms / total
        orb = int(rng.choice(norb, p=probs))
        selected.append(orb)
        if active == 1:
            # No further elimination needed
            break
        pivot_val = pivot_normal[orb]
        if np.abs(pivot_val) <= tol:
            raise ValueError("Numerical pivot breakdown in fast sampler.")
        # Update remaining normals via Gaussian elimination (O(N^2))
        for j in range(1, active):
            active_normals[:, j] = (
                active_normals[:, j]
                - (active_normals[orb, j] / pivot_val) * pivot_normal
            )
        # Drop the first normal vector
        active_normals = active_normals[:, 1:active]
        active -= 1
    return sum(1 << orb for orb in selected)
