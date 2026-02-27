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

from collections.abc import Sequence
from typing import cast, overload

import numpy as np
import scipy.linalg

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
    concatenate: bool = True,
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
    using the projection-based sequential sampling algorithm described in the
    reference below.

    References:
        - `Sun, Zou, and Li, "Fermion Sampling Made More Efficient" (2023)`_

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

    .. _Sun, Zou, and Li, "Fermion Sampling Made More Efficient" (2023): https://arxiv.org/abs/2109.07358
    """
    rng = np.random.default_rng(seed)

    if not occupied_orbitals or isinstance(occupied_orbitals[0], (int, np.integer)):
        # Spinless case
        nelec = len(occupied_orbitals)
        if orbs is None:
            orbs = range(norb)
        strings = _sample_strings(
            norb,
            nelec,
            cast(Sequence[int], occupied_orbitals),
            cast(np.ndarray | None, orbital_rotation),
            shots,
            rng,
        )
        strings = restrict_bitstrings(
            strings, cast(Sequence[int], orbs), bitstring_type=BitstringType.INT
        )
        return convert_bitstring_type(
            strings,
            BitstringType.INT,
            bitstring_type,
            length=len(orbs),
        )

    # Spinful case
    if orbs is None:
        orbs = (range(norb), range(norb))
    orbs_a, orbs_b = cast(tuple[Sequence[int], Sequence[int]], orbs)
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
        orbital_rotation_a, orbital_rotation_b = orbital_rotation

    strings_a = _sample_strings(
        norb, n_a, occupied_orbitals_a, orbital_rotation_a, shots, rng
    )
    strings_b = _sample_strings(
        norb, n_b, occupied_orbitals_b, orbital_rotation_b, shots, rng
    )
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


def _sample_strings(
    norb: int,
    nocc: int,
    occupied_orbitals: Sequence[int],
    orbital_rotation: np.ndarray | None,
    shots: int,
    rng: np.random.Generator,
) -> list[int]:
    if nocc == 0:
        return [0] * shots
    if nocc == norb:
        return [(1 << norb) - 1] * shots
    if orbital_rotation is None:
        orbital_rotation = np.eye(norb)
    if 2 * nocc > norb:
        # high filling: sample holes instead of particles
        occupied_orbitals_set = set(occupied_orbitals)
        unoccupied = [i for i in range(norb) if i not in occupied_orbitals_set]
        orbitals = orbital_rotation.conj()[:, unoccupied]
        normals = _compute_projection_normals(orbitals)
        full_mask = (1 << norb) - 1
        return [
            full_mask ^ _sample_from_projection_normals(normals, rng)
            for _ in range(shots)
        ]
    else:
        orbitals = orbital_rotation.conj()[:, occupied_orbitals]
        normals = _compute_projection_normals(orbitals)
        return [_sample_from_projection_normals(normals, rng) for _ in range(shots)]


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
    if np.isrealobj(normals):
        blas_ger = scipy.linalg.blas.dger
    else:
        blas_ger = scipy.linalg.blas.zgeru
    norb, nelec = normals.shape
    perm = rng.permutation(nelec)
    # Working set of normals after applying the random permutation.
    active_normals = np.asfortranarray(normals[:, perm])
    selected: list[int] = []
    active = nelec
    # Iterate until all electrons are placed
    while active:
        # Pivot column used for sampling and elimination.
        pivot_normal = active_normals[:, 0]
        probs = np.abs(pivot_normal) ** 2
        probs /= probs.sum()
        orb = rng.choice(norb, p=probs)
        selected.append(orb)
        if active == 1:
            # No further elimination needed
            break
        pivot_val = pivot_normal[orb]
        # Update remaining normals via Gaussian elimination (rank-1 update)
        active_normals[:, 1:active] = blas_ger(
            -1 / pivot_val,
            pivot_normal,
            active_normals[orb, 1:active],
            a=active_normals[:, 1:active],
        )
        # Drop the first normal vector
        active_normals = active_normals[:, 1:active]
        active -= 1
    return sum(1 << orb for orb in selected)
