# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fermionic quantum states."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import scipy.linalg
from pyscf.fci.spin_op import contract_ss

from ffsim.states.bitstring import (
    BitstringType,
    addresses_to_strings,
    concatenate_bitstrings,
    restrict_bitstrings,
    strings_to_addresses,
)


@dataclass
class StateVector:
    """A state vector in the FCI representation.

    Attributes:
        vec: Array of state vector coefficients.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
    """

    vec: np.ndarray
    norb: int
    nelec: int | tuple[int, int]

    def __array__(self, dtype=None, copy=None):
        # TODO in Numpy 2.0 this can be simplified to
        # return np.array(self.vec, dtype=dtype, copy=copy)
        if copy:
            if dtype is None:
                return self.vec.copy()
            else:
                return self.vec.astype(dtype, copy=True)
        if dtype is None:
            return self.vec
        return self.vec.astype(dtype, copy=False)


def dims(norb: int, nelec: tuple[int, int]) -> tuple[int, int]:
    """Get the dimensions of the FCI space.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        A pair of integers (dim_a, dim_b) representing the dimensions of the
        alpha- and beta- FCI space.
    """
    n_alpha, n_beta = nelec
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    return dim_a, dim_b


def dim(norb: int, nelec: int | tuple[int, int]) -> int:
    """Get the dimension of the FCI space.

    Args:
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.

    Returns:
        The dimension of the FCI space.
    """
    if isinstance(nelec, int):
        return math.comb(norb, nelec)
    n_alpha, n_beta = nelec
    return math.comb(norb, n_alpha) * math.comb(norb, n_beta)


# source: pyscf.fci.spin_op.spin_square0
# modified to support complex wavefunction
def spin_square(fcivec: np.ndarray, norb: int, nelec: tuple[int, int]):
    """Expectation value of spin squared operator on a state vector."""
    if np.iscomplexobj(fcivec):
        ci1 = contract_ss(fcivec.real, norb, nelec).astype(complex)
        ci1 += 1j * contract_ss(fcivec.imag, norb, nelec)
    else:
        ci1 = contract_ss(fcivec, norb, nelec)
    return np.einsum("ij,ij->", fcivec.reshape(ci1.shape), ci1.conj()).real


def sample_state_vector(
    vec: np.ndarray | StateVector,
    *,
    norb: int | None = None,
    nelec: int | tuple[int, int] | None = None,
    orbs: Sequence[int] | tuple[Sequence[int], Sequence[int]] | None = None,
    shots: int = 1,
    concatenate: bool = True,
    bitstring_type: BitstringType = BitstringType.STRING,
    seed: np.random.Generator | int | None = None,
):
    """Sample bitstrings from a state vector.

    Args:
        vec: The state vector to sample from.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        orbs: The spin-orbitals to sample.
            In the spinless case (when `nelec` is an integer), this is a list of
            integers ranging from ``0`` to ``norb``.
            In the spinful case, this is a pair of lists of such integers, with the
            first list storing the spin-alpha orbitals and the second list storing
            the spin-beta orbitals.
            If not specified, then all spin-orbitals are sampled.
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
        The sampled bitstrings, as a list of strings of length `shots`.

    Raises:
        TypeError: When passing vec as a Numpy array, norb and nelec must be specified.
        TypeError: When passing vec as a StateVector, norb and nelec must both be None.
    """
    vec, norb, nelec = canonicalize_vec_norb_nelec(vec, norb, nelec)
    if isinstance(nelec, int):
        return _sample_state_vector_spinless(
            vec,
            norb=norb,
            nelec=nelec,
            orbs=cast(Optional[Sequence[int]], orbs),
            shots=shots,
            bitstring_type=bitstring_type,
            seed=seed,
        )
    return _sample_state_vector_spinful(
        vec,
        norb=norb,
        nelec=nelec,
        orbs=cast(Optional[tuple[Sequence[int], Sequence[int]]], orbs),
        shots=shots,
        concatenate=concatenate,
        bitstring_type=bitstring_type,
        seed=seed,
    )


def _sample_state_vector_spinless(
    vec: np.ndarray,
    *,
    norb: int,
    nelec: int,
    orbs: Sequence[int] | None,
    shots: int,
    bitstring_type: BitstringType,
    seed: np.random.Generator | int | None,
):
    if orbs is None:
        orbs = range(norb)
    rng = np.random.default_rng(seed)
    probabilities = np.abs(vec) ** 2
    samples = rng.choice(len(vec), size=shots, p=probabilities)
    strings = addresses_to_strings(samples, norb, nelec, bitstring_type=bitstring_type)
    if list(orbs) == list(range(norb)):
        return strings
    return restrict_bitstrings(strings, orbs, bitstring_type=bitstring_type)


def _sample_state_vector_spinful(
    vec: np.ndarray,
    *,
    norb: int,
    nelec: tuple[int, int],
    orbs: tuple[Sequence[int], Sequence[int]] | None,
    shots: int,
    concatenate: bool = True,
    bitstring_type: BitstringType,
    seed: np.random.Generator | int | None,
):
    if orbs is None:
        orbs = range(norb), range(norb)
    rng = np.random.default_rng(seed)
    probabilities = np.abs(vec) ** 2
    samples = rng.choice(len(vec), size=shots, p=probabilities)
    orbs_a, orbs_b = orbs

    if list(orbs_a) == list(orbs_b) == list(range(norb)):
        # All orbitals are sampled, so we can simply call addresses_to_strings
        return addresses_to_strings(
            samples, norb, nelec, concatenate=concatenate, bitstring_type=bitstring_type
        )

    strings_a, strings_b = addresses_to_strings(
        samples, norb, nelec, concatenate=False, bitstring_type=bitstring_type
    )
    strings_a = restrict_bitstrings(strings_a, orbs_a, bitstring_type=bitstring_type)
    strings_b = restrict_bitstrings(strings_b, orbs_b, bitstring_type=bitstring_type)
    if concatenate:
        return concatenate_bitstrings(
            strings_a, strings_b, bitstring_type=bitstring_type, length=len(orbs_a)
        )
    return strings_a, strings_b


def canonicalize_vec_norb_nelec(
    vec: np.ndarray | StateVector, norb: int | None, nelec: int | tuple[int, int] | None
) -> tuple[np.ndarray, int, int | tuple[int, int]]:
    """Canonicalize a state vector, number of orbitals, and number(s) of electrons.

    Args:
        vec: The state vector.
        norb: The number of spatial orbitals, or None if `vec` is a StateVector.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions. If `vec` is a StateVector then this should be None.

    Returns:
        The state vector as a Numpy array, the number of spatial orbitals, and the
        number or numbers of electrons.

    Raises:
        TypeError: When passing vec as a Numpy array, norb and nelec must be specified.
        TypeError: When passing vec as a StateVector, norb and nelec must both be None.
    """
    if isinstance(vec, np.ndarray):
        if norb is None or nelec is None:
            raise TypeError(
                "When passing vec as a Numpy array, norb and nelec must be specified."
            )
    else:
        if norb is not None or nelec is not None:
            raise TypeError(
                "When passing vec as a StateVector, norb and nelec must both be None."
            )
        norb = vec.norb
        nelec = vec.nelec
        vec = vec.vec
    return vec, norb, nelec


def spinful_to_spinless_vec(
    vec: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    """Convert a spinful state vector to a spinless state vector.

    Args:
        vec: The state vector in spinful format.
        norb: The number of spatial orbitals.
        nelec: The numbers of spin alpha and spin beta electrons.

    Returns:
        The state vector in spinless format, with norb = norb and nocc = sum(nelec).
    """
    nocc = sum(nelec)
    old_dim = dim(norb, nelec)
    new_dim = dim(2 * norb, nocc)
    new_vec = np.zeros(new_dim, dtype=vec.dtype)
    strings = addresses_to_strings(range(old_dim), norb=norb, nelec=nelec)
    addresses = strings_to_addresses(strings, norb=2 * norb, nelec=nocc)
    new_vec[addresses] = vec
    return new_vec


def spinful_to_spinless_rdm1(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Convert a spinful 1-RDM to a spinless 1-RDM.

    Args:
        mat_a: The spin alpha part of the 1-RDM.
        mat_b: The spin beta part of the 1-RDM.

    Returns:
        The 1-RDM as a single matrix in spinless format.
    """
    return scipy.linalg.block_diag(mat_a, mat_b)


def spinful_to_spinless_rdm2(
    mat_aa: np.ndarray, mat_ab: np.ndarray, mat_bb: np.ndarray
) -> np.ndarray:
    """Convert a spinful 2-RDM to a spinless 2-RDM.

    Args:
        mat_aa: The alpha-alpha part of the 2-RDM.
        mat_ab: The alpha-beta part of the 2-RDM.
        mat_bb: The beta-beta part of the 2-RDM.

    Returns:
        The 2-RDM as a single tensor in spinless format.
    """
    norb, _, _, _ = mat_aa.shape
    new_mat = np.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=mat_aa.dtype)
    new_mat[:norb, :norb, :norb, :norb] = mat_aa
    new_mat[:norb, :norb, norb:, norb:] = mat_ab
    new_mat[:norb, norb:, norb:, :norb] = -mat_ab.transpose(0, 3, 2, 1)
    new_mat[norb:, :norb, :norb, norb:] = -mat_ab.transpose(2, 1, 0, 3)
    new_mat[norb:, norb:, :norb, :norb] = mat_ab.transpose(2, 3, 0, 1)
    new_mat[norb:, norb:, norb:, norb:] = mat_bb
    return new_mat
