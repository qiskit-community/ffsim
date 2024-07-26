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
from typing import cast

import numpy as np

from ffsim.states.bitstring import (
    BitstringType,
    concatenate_bitstrings,
    convert_bitstring_type,
    restrict_bitstrings,
)


def sample_slater(
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
        # spinless case
        rdm = cast(np.ndarray, rdm)
        norb, _ = rdm.shape

        if orbs is None:
            orbs = range(norb)
        orbs = cast(Sequence[int], orbs)

        strings = _sample_spinless_direct(rdm, nelec, shots, rng)
        strings = restrict_bitstrings(strings, orbs, bitstring_type=BitstringType.INT)

        return convert_bitstring_type(
            strings,
            BitstringType.INT,
            bitstring_type,
            length=len(orbs),
        )

    else:
        rdm_a, rdm_b = rdm

        n_a, n_b = nelec
        norb, _ = rdm_a.shape

        if orbs is None:
            orbs = (range(norb), range(norb))
        orbs_a, orbs_b = orbs
        orbs_a = cast(Sequence[int], orbs_a)
        orbs_b = cast(Sequence[int], orbs_b)

        strings_a = _sample_spinless_direct(rdm_a, n_a, shots, rng)
        strings_b = _sample_spinless_direct(rdm_b, n_b, shots, rng)
        strings_a = restrict_bitstrings(
            strings_a, orbs_a, bitstring_type=BitstringType.INT
        )
        strings_b = restrict_bitstrings(
            strings_b, orbs_b, bitstring_type=BitstringType.INT
        )

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


def _generate_marginals(
    rdm: np.ndarray, sample: list[int], empty_orbitals: list[int], marginal: float
) -> np.ndarray:
    """Computes the marginal probabilities for adding a particle.

    This is a step of the autoregressive sampling, and uses Bayes's rule.

    Args:
        rdm: A Numpy array with the one-body reduced density matrix.
        pos_array: A Numpy array with the positions of the particles.
        empty_orbitals: A Numpy array with the empty orbitals that a new particle
            may occupy. The sorted union of ``pos_array`` and ``empty_orbitals``
            must be equal to ``numpy.arange(num_orbitals)``.
        marginal: The marginal probability associated to having the particles
            in the positions determined by ``pos_array``.

    Returns:
        A Numpy array with the marginal corresponding to having the particles in the
        position array and one extra in all possible empty orbitals.

    """
    marginals = np.zeros(len(empty_orbitals), dtype=float)
    for i, orbital in enumerate(empty_orbitals):
        new_sample = sample.copy()
        new_sample = sample + [orbital]
        rest_rdm = rdm[np.ix_(new_sample, new_sample)]
        marginals[i] = np.linalg.det(rest_rdm).real
    return marginals


def _autoregressive_slater(
    rdm: np.ndarray,
    norb: int,
    nelec: int,
    seed: np.random.Generator | int | None = None,
) -> list[int]:
    """Autoregressively sample positions of particles for a Slater determinant wave
    function using a determinantal point process.

    Args:
        rdm: A Numpy array with the one-body reduced density matrix.
        norb: Number of orbitals.
        nelec: Number of electrons.
        seed: Either a Numpy random generator, an integer seed for the random number
            generator or ``None``.

    Returns:
        A Numpy array with the position of the sampled electrons.
    """
    rng = np.random.default_rng(seed)

    probs = np.diag(rdm).real / nelec
    sample = [rng.choice(norb, p=probs)]
    marginal = [probs[sample[0]]]
    all_orbs = set(range(norb))
    empty_orbitals = list(all_orbs.difference(sample))
    for k in range(nelec - 1):
        marginals = _generate_marginals(rdm, sample, empty_orbitals, marginal[-1])
        conditionals = marginals / marginal[-1]
        conditionals /= np.sum(conditionals)
        index = rng.choice(len(empty_orbitals), p=conditionals)
        sample.append(empty_orbitals[index])
        marginal.append(marginals[index])
        empty_orbitals.pop(index)
    return sample


def _sample_spinless_direct(
    rdm: np.ndarray,
    nelec: int,
    shots: int,
    seed: np.random.Generator | int | None = None,
) -> list[int]:
    """Collect samples of electronic configurations from a Slater determinant for
    spin-polarized systems.

    The Slater determinat is defined by its one-body reduced density matrix (RDM).
    The sampler uses a determinantal point process to auto-regressively produce
    uncorrelated samples.

    Args:
        rdm: The one-body reduced density matrix that defines the Slater determinant
            This is either a single Numpy array specifying the 1-RDM of a
            spin-polarized system, or a pair of Numpy arrays where each element
            of the pair contains the 1-RDM for each spin sector.
        shots: Number of samples to collect.
        seed: Either a Numpy random generator, an integer seed for the random number
            generator or ``None``.

    Returns:
        A 2D Numpy array with samples of electronic configurations.
        Each row is a sample.
    """
    norb, _ = rdm.shape

    if nelec == 0:
        return [0] * shots
    if nelec == norb:
        return [(1 << norb) - 1] * shots
    rng = np.random.default_rng(seed)
    samples = []
    samples = [_autoregressive_slater(rdm, norb, nelec, rng) for _ in range(shots)]
    return [sum(1 << orb for orb in sample) for sample in samples]
