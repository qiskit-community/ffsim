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
from typing import cast, overload

import numpy as np
import scipy.linalg
from pyscf.fci import cistring
from pyscf.fci.spin_op import contract_ss
from typing_extensions import deprecated

from ffsim import linalg
from ffsim.gates.orbital_rotation import apply_orbital_rotation


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


@deprecated(
    "Using one_hot from the ffsim namespace is deprecated. "
    "Instead, use ffsim.linalg.one_hot."
)
def one_hot(shape: int | tuple[int, ...], index, *, dtype=complex):
    """Return an array of all zeros except for a one at a specified index.

    .. warning::
        This function is deprecated. Use :func:`ffsim.linalg.one_hot` instead.

    Args:
        shape: The desired shape of the array.
        index: The index at which to place a one.

    Returns:
        The one-hot vector.
    """
    vec = np.zeros(shape, dtype=dtype)
    vec[index] = 1
    return vec


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


@deprecated(
    "ffsim.slater_determinant_rdm is deprecated. "
    "Instead, use ffsim.slater_determinant_rdms."
)
def slater_determinant_rdm(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
    rank: int = 1,
    spin_summed: bool = True,
) -> np.ndarray:
    """Return the reduced density matrix of a `Slater determinant`_.

    .. warning::
        This function is deprecated. Use :func:`ffsim.slater_determinant_rdms` instead.

    Note:
        Currently, only rank 1 is supported.

    Args:
        norb: The number of spatial orbitals.
        occupied_orbitals: The occupied orbitals in the electronic configuration.
            This is a pair of lists of integers, where the first list specifies the
            spin alpha orbitals and the second list specifies the spin beta
            orbitals.
        orbital_rotation: The optional orbital rotation.
            You can pass either a single Numpy array specifying the orbital rotation
            to apply to both spin sectors, or you can pass a pair of Numpy arrays
            specifying independent orbital rotations for spin alpha and spin beta.
            If passing a pair, you can use ``None`` for one of the
            values in the pair to indicate that no operation should be applied to that
            spin sector.
        rank: The rank of the reduced density matrix. I.e., rank 1 corresponds to the
            one-particle RDM, rank 2 corresponds to the 2-particle RDM, etc.
        spin_summed: Whether to sum over the spin index.

    Returns:
        The reduced density matrix of the Slater determinant.

    .. _Slater determinant: ffsim.html#ffsim.slater_determinant
    """
    if rank == 1:
        rdm_a = np.zeros((norb, norb), dtype=complex)
        rdm_b = np.zeros((norb, norb), dtype=complex)
        alpha_orbitals = np.array(occupied_orbitals[0])
        beta_orbitals = np.array(occupied_orbitals[1])
        if len(alpha_orbitals):
            rdm_a[(alpha_orbitals, alpha_orbitals)] = 1
        if len(beta_orbitals):
            rdm_b[(beta_orbitals, beta_orbitals)] = 1
        if orbital_rotation is not None:
            if isinstance(orbital_rotation, np.ndarray):
                orbital_rotation_a: np.ndarray | None = orbital_rotation
                orbital_rotation_b: np.ndarray | None = orbital_rotation
            else:
                orbital_rotation_a, orbital_rotation_b = orbital_rotation
            if orbital_rotation_a is not None:
                rdm_a = orbital_rotation_a.conj() @ rdm_a @ orbital_rotation_a.T
            if orbital_rotation_b is not None:
                rdm_b = orbital_rotation_b.conj() @ rdm_b @ orbital_rotation_b.T
        if spin_summed:
            return rdm_a + rdm_b
        return scipy.linalg.block_diag(rdm_a, rdm_b)
    raise NotImplementedError(
        f"Returning the rank {rank} reduced density matrix is currently not supported."
    )


@overload
def slater_determinant_rdms(
    norb: int,
    occupied_orbitals: Sequence[int],
    orbital_rotation: np.ndarray | None = None,
) -> np.ndarray: ...
@overload
def slater_determinant_rdms(
    norb: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    orbital_rotation: np.ndarray
    | tuple[np.ndarray | None, np.ndarray | None]
    | None = None,
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


def indices_to_strings(
    indices: Sequence[int] | np.ndarray, norb: int, nelec: int | tuple[int, int]
) -> list[str]:
    """Convert state vector indices to bitstrings.

    Example:

    .. code::

        import ffsim

        norb = 3
        nelec = (2, 1)
        dim = ffsim.dim(norb, nelec)
        ffsim.indices_to_strings(range(dim), norb, nelec)
        # output:
        # ['001011',
        #  '010011',
        #  '100011',
        #  '001101',
        #  '010101',
        #  '100101',
        #  '001110',
        #  '010110',
        #  '100110']
    """
    if isinstance(nelec, int):
        strings = cistring.addrs2str(norb=norb, nelec=nelec, addrs=indices)
        return [f"{string:0{norb}b}" for string in strings]

    n_alpha, n_beta = nelec
    dim_b = math.comb(norb, n_beta)
    indices_a, indices_b = np.divmod(indices, dim_b)
    strings_a = cistring.addrs2str(norb=norb, nelec=n_alpha, addrs=indices_a)
    strings_b = cistring.addrs2str(norb=norb, nelec=n_beta, addrs=indices_b)
    return [
        f"{string_b:0{norb}b}{string_a:0{norb}b}"
        for string_a, string_b in zip(strings_a, strings_b)
    ]


def strings_to_indices(
    strings: Sequence[str] | np.ndarray, norb: int, nelec: int | tuple[int, int]
) -> np.ndarray:
    """Convert bitstrings to state vector indices.

    Example:

    .. code::

        import ffsim

        norb = 3
        nelec = (2, 1)
        dim = ffsim.dim(norb, nelec)
        ffsim.strings_to_indices(
            [
                "001011",
                "010011",
                "100011",
                "001101",
                "010101",
                "100101",
                "001110",
                "010110",
                "100110",
            ],
            norb,
            nelec,
        )
        # output:
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)
    """
    if isinstance(nelec, int):
        return cistring.strs2addr(
            norb=norb, nelec=nelec, strings=[int(s, base=2) for s in strings]
        )

    n_alpha, n_beta = nelec
    strings_a = [int(s[norb:], base=2) for s in strings]
    strings_b = [int(s[:norb], base=2) for s in strings]
    addrs_a = cistring.strs2addr(norb=norb, nelec=n_alpha, strings=strings_a)
    addrs_b = cistring.strs2addr(norb=norb, nelec=n_beta, strings=strings_b)
    dim_b = math.comb(norb, n_beta)
    return addrs_a * dim_b + addrs_b


def addresses_to_strings(
    addresses: Sequence[int] | np.ndarray, norb: int, nelec: int | tuple[int, int]
) -> np.ndarray:
    """Convert state vector addresses to bitstrings.

    Example:

    .. code::

        import ffsim

        norb = 3
        nelec = (2, 1)
        dim = ffsim.dim(norb, nelec)
        strings = ffsim.addresses_to_strings(range(dim), norb, nelec)
        [format(s, f"06b") for s in strings]
        # output:
        # ['001011',
        #  '010011',
        #  '100011',
        #  '001101',
        #  '010101',
        #  '100101',
        #  '001110',
        #  '010110',
        #  '100110']
    """
    if isinstance(nelec, int):
        return cistring.addrs2str(norb=norb, nelec=nelec, addrs=addresses)
    if norb >= 32:
        raise NotImplementedError(
            "addresses_to_strings currently does not support norb >= 32."
        )
    n_alpha, n_beta = nelec
    dim_b = math.comb(norb, n_beta)
    addresses_a, addresses_b = np.divmod(addresses, dim_b)
    strings_a = cistring.addrs2str(norb=norb, nelec=n_alpha, addrs=addresses_a)
    strings_b = cistring.addrs2str(norb=norb, nelec=n_beta, addrs=addresses_b)
    return (strings_b << norb) + strings_a


def strings_to_addresses(
    strings: Sequence[int] | np.ndarray, norb: int, nelec: int | tuple[int, int]
) -> np.ndarray:
    """Convert bitstrings to state vector addresses.

    Example:

    .. code::

        import ffsim

        norb = 3
        nelec = (2, 1)
        dim = ffsim.dim(norb, nelec)
        ffsim.strings_to_addresses(
            [
                0b001011,
                0b010011,
                0b100011,
                0b001101,
                0b010101,
                0b100101,
                0b001110,
                0b010110,
                0b100110,
            ],
            norb,
            nelec,
        )
        # output:
        # array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)
    """
    if isinstance(nelec, int):
        return cistring.strs2addr(norb=norb, nelec=nelec, strings=strings)

    n_alpha, n_beta = nelec
    strings = np.asarray(strings)
    strings_a = strings & ((1 << norb) - 1)
    strings_b = strings >> norb
    addrs_a = cistring.strs2addr(norb=norb, nelec=n_alpha, strings=strings_a)
    addrs_b = cistring.strs2addr(norb=norb, nelec=n_beta, strings=strings_b)
    dim_b = math.comb(norb, n_beta)
    return addrs_a * dim_b + addrs_b


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
    orbs: list[int],
    shots: int,
    norb: int | None = None,
    nelec: int | tuple[int, int] | None = None,
    seed: np.random.Generator | int | None = None,
) -> list[str]:
    """Sample bitstrings from a state vector.

    Args:
        vec: The state vector to sample from.
        orbs: The spin-orbitals to sample from. These are integers ranging from
            ``0`` to ``2 * norb - 1``, with the first half of the range representing
            the spin alpha orbitals, and the second half representing the spin beta
            orbitals.
        shots: The number of bitstrings to sample.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled bitstrings, as a list of strings of length `shots`.

    Raises:
        TypeError: When passing vec as a Numpy array, norb and nelec must be specified.
        TypeError: When passing vec as a StateVector, norb and nelec must both be None.
    """
    vec, norb, nelec = canonicalize_vec_norb_nelec(vec, norb, nelec)
    rng = np.random.default_rng(seed)
    probabilities = np.abs(vec) ** 2
    samples = rng.choice(len(vec), size=shots, p=probabilities)
    bitstrings = indices_to_strings(samples, norb, nelec)
    if orbs == list(range(2 * norb)):
        return bitstrings
    return ["".join(bitstring[-1 - i] for i in orbs[::-1]) for bitstring in bitstrings]


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
