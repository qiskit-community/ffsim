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
from typing import cast

import numpy as np

from ffsim.states.bitstring import (
    BitstringType,
    concatenate_bitstrings,
    convert_bitstring_type,
    restrict_bitstrings,
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
