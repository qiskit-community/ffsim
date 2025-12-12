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
from typing import cast

import numpy as np
import scipy.linalg

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
    method: str = "fast",
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
        method: Which sampler to use.
            ``"fast"`` (default) runs the projection-based algorithm from
            *Fermion Sampling Made More Efficient* (Phys. Rev. B 107, 035119),
            including the marginal mode when the projector rank exceeds ``nelec``,
            and falls back internally when the 1-RDM is not an exact projector.
            ``"determinant"`` uses the determinant-based autoregressive sampler.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        A 2D Numpy array with samples of electronic configurations.
        Each row is a sample.
    """
    method = method.lower()
    if method not in {"determinant", "fast"}:
        raise ValueError(f"Unsupported sampling method: {method}")

    rng = np.random.default_rng(seed)

    if isinstance(nelec, int):
        # Spinless case
        rdm = cast(np.ndarray, rdm)
        norb, _ = rdm.shape
        if orbs is None:
            orbs = range(norb)
        orbs = cast(Sequence[int], orbs)
        if method == "fast":
            try:
                strings = _sample_slater_spinless_fast(rdm, nelec, shots, rng)
            except ValueError:
                strings = _sample_slater_spinless(rdm, nelec, shots, rng)
        else:
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
    if method == "fast":
        try:
            strings_a = _sample_slater_spinless_fast(rdm_a, n_a, shots, rng)
            strings_b = _sample_slater_spinless_fast(rdm_b, n_b, shots, rng)
        except ValueError:
            strings_a = _sample_slater_spinless(rdm_a, n_a, shots, rng)
            strings_b = _sample_slater_spinless(rdm_b, n_b, shots, rng)
    else:
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


def _sample_slater_spinless_fast(
    rdm: np.ndarray,
    nelec: int,
    shots: int,
    seed: np.random.Generator | int | None = None,
) -> list[int]:
    """Collect samples of electronic configurations from a Slater determinant.

    Uses the projection-based fast sampler: it builds an orthonormal orbital
    basis from a Hermitian projector 1-RDM and iteratively samples rows via
    Gaussian-elimination updates. If the projector rank exceeds ``nelec``, it
    samples the marginal distribution by uniformly selecting occupied orbitals
    inside the projector subspace on every draw. This routine raises a
    ``ValueError`` when given a nonprojector 1-RDM or when ``nelec`` exceeds
    the projector rank; callers are expected to catch and fall back to the
    determinant-based sampler when needed.

    Args:
        rdm: The one-body reduced density matrix that defines the Slater determinant.
        nelec: Number of electrons.
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
    if not np.allclose(rdm, rdm.conj().T, atol=1e-10):
        raise ValueError("Fast sampler requires a Hermitian projector 1-RDM.")
    tol = 1e-8
    eigvals, eigvecs = np.linalg.eigh(rdm)
    rank = int(round(float(np.clip(eigvals, 0.0, 1.0).sum())))
    if not _is_projection_eigvals(eigvals, rank, tol=tol):
        raise ValueError("Fast sampler requires a projector 1-RDM.")
    order = np.argsort(eigvals)[::-1]
    orbitals = eigvecs[:, order[:rank]]
    if nelec > rank:
        raise ValueError("Fast sampler requires at least nelec occupied orbitals.")

    if nelec == rank:
        # Full-rank projector
        sampler = _FastProjectionSampler(orbitals)
        return [sampler.sample(rng) for _ in range(shots)]

    def sample_once() -> int:
        """Draw one bitstring from a rank-``rank`` projector marginal."""
        cols = rng.choice(rank, size=nelec, replace=False)
        sampler = _FastProjectionSampler(orbitals[:, cols])
        return sampler.sample(rng)

    return [sample_once() for _ in range(shots)]


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


class _FastProjectionSampler:
    """Projection-based fast sampler core with cached precomputation.

    Given an orthonormal orbital matrix for a projector 1-RDM, the sampler
    precomputes the normal vectors needed for the Gaussian-elimination sampling
    routine and exposes a ``sample`` method that generates integer-encoded
    bitstrings using a provided RNG.
    """

    def __init__(self, orbitals: np.ndarray):
        """Initialize the sampler with occupied orbitals from a projector 1-RDM.

        Precomputes dual normal vectors via reduced QR factorization and
        triangular solves for reuse during sampling.

        Args:
            orbitals: A 2D array whose columns are orthonormal orbitals spanning
                the occupied subspace. Shape ``(norb, nelec)``.

        Raises:
            ValueError: If the input is not 2D, has zero occupied orbitals, or is
                ill-conditioned for the QR-based precomputation.
        """
        if orbitals.ndim != 2:
            raise ValueError("Projector orbitals must be a 2D array.")
        self._norb, self._nelec = orbitals.shape
        if self._nelec == 0:
            raise ValueError("Fast sampler requires at least one occupied orbital.")
        # Precompute normal vectors h_t from QR
        q, r = np.linalg.qr(orbitals, mode="reduced")
        diag = np.abs(np.diag(r))
        tol = 1e-12
        if np.any(diag <= tol):
            raise ValueError(
                "Projector factorization is ill-conditioned for fast sampling."
            )
        inv_rt = scipy.linalg.solve_triangular(
            r.T, np.eye(self._nelec), lower=True, check_finite=False
        )
        self._normals = q @ inv_rt

    @property
    def norb(self) -> int:
        return self._norb

    @property
    def nelec(self) -> int:
        return self._nelec

    def sample(self, rng: np.random.Generator) -> int:
        """Draw one configuration using the cached normals and supplied RNG.

        Permutes the precomputed normal vectors, then iteratively picks an
        orbital with probability proportional to the squared magnitude of the
        leading normal, performs a Gaussian-elimination pivot to update the
        remaining normals, and repeats until all electrons are placed.

        Args:
            rng: NumPy Generator to use for randomness.

        Returns:
            Integer-encoded bitstring for the sampled configuration.
        """
        perm = rng.permutation(self._nelec)
        h = self._normals[:, perm].copy()
        selected: list[int] = []
        active = self._nelec
        tol = 1e-12
        # Iterate until all electrons are placed
        while active:
            h0 = h[:, 0]
            row_norms = np.abs(h0) ** 2
            total = float(np.sum(row_norms))
            if total <= 0:
                raise ValueError("Failed to compute sampling probabilities.")
            probs = row_norms / total
            orb = int(rng.choice(self._norb, p=probs))
            selected.append(orb)
            if active == 1:
                # No further elimination needed
                break
            pivot_val = h0[orb]
            if np.abs(pivot_val) <= tol:
                raise ValueError("Numerical pivot breakdown in fast sampler.")
            # Update remaining normals via Gaussian elimination (O(N^2))
            for j in range(1, active):
                h[:, j] = h[:, j] - (h[orb, j] / pivot_val) * h0
            # Drop the first normal vector
            h = h[:, 1:active]
            active -= 1
        return sum(1 << orb for orb in selected)


def _is_projection_eigvals(eigvals: np.ndarray, nelec: int, tol: float = 1e-8) -> bool:
    """Return True if eigenvalues describe a rank-``nelec`` projector.

    Args:
        eigvals: Eigenvalues of the 1-RDM.
        nelec: Expected rank (number of occupied orbitals).
        tol: Absolute tolerance for checking idempotency and trace.

    Returns:
        True when the spectrum is within ``tol`` of a {0, 1} projector with trace
        ``nelec``; False otherwise.
    """
    eigvals = np.clip(eigvals, 0.0, 1.0)
    trace_ok = np.isclose(np.sum(eigvals), nelec, atol=tol)
    idempotent_ok = np.all(
        np.isclose(eigvals, 0.0, atol=tol) | np.isclose(eigvals, 1.0, atol=tol)
    )
    return trace_ok and idempotent_ok
