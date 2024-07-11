# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for handling bitstrings."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum, auto
from typing import cast

import numpy as np
from pyscf.fci import cistring


class BitstringType(Enum):
    """Enumeration for indicating the data type of bitstrings.

    String:
        ["0101", "0110"]

    Integer:
        [5, 6]

    Bit array:
        [[False, True, False, True],
         [False, True, True, False]]
    """

    STRING = auto()
    """String."""

    INT = auto()
    """Integer."""

    BIT_ARRAY = auto()
    """Bit array."""


def indices_to_strings(
    indices: Sequence[int] | np.ndarray,
    norb: int,
    nelec: int | tuple[int, int],
    concatenate: bool = True,
    bitstring_type: BitstringType = BitstringType.STRING,
):
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

    Args:
        indices: The state vector indices to convert to bitstrings.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        bitstring_type: The desired type of bitstring output.
        concatenate: Whether to concatenate the spin-alpha and spin-beta parts of the
            bitstrings. If True, then a single list of concatenated bitstrings is
            returned. The strings are concatenated in the order :math:`s_b s_a`,
            that is, the alpha string appears on the right.
            If False, then two lists are returned, ``(strings_a, strings_b)``. Note that
            the list of alpha strings appears first, that is, on the left.
            In the spinless case (when `nelec` is an integer), this argument is ignored.
    """
    if isinstance(nelec, int):
        # Spinless case
        return convert_bitstring_type(
            # TODO convert to python int
            list(cistring.addrs2str(norb=norb, nelec=nelec, addrs=indices)),
            input_type=BitstringType.INT,
            output_type=bitstring_type,
            length=norb,
        )

    # Spinful case
    n_alpha, n_beta = nelec
    dim_b = math.comb(norb, n_beta)
    indices_a, indices_b = np.divmod(indices, dim_b)
    strings_a = convert_bitstring_type(
        # TODO convert to python int
        list(cistring.addrs2str(norb=norb, nelec=n_alpha, addrs=indices_a)),
        input_type=BitstringType.INT,
        output_type=bitstring_type,
        length=norb,
    )
    strings_b = convert_bitstring_type(
        # TODO convert to python int
        list(cistring.addrs2str(norb=norb, nelec=n_beta, addrs=indices_b)),
        input_type=BitstringType.INT,
        output_type=bitstring_type,
        length=norb,
    )
    if concatenate:
        return concatenate_bitstrings(
            strings_a, strings_b, bitstring_type=bitstring_type, length=norb
        )
    return strings_a, strings_b


def convert_bitstring_type(
    strings: list[str] | list[int] | np.ndarray,
    input_type: BitstringType,
    output_type: BitstringType,
    length: int,
):
    """Convert bitstrings from one type to another.

    Args:
        strings: The bitstrings.
        input_type: The bitstring type of the input.
        output_type: The desired bitstring type of the output.
        length: The length of a bitstring.

    Returns:
        The converted bitstrings.
    """
    if input_type is output_type:
        return strings

    if input_type is BitstringType.STRING:
        strings = cast(list[str], strings)
        if output_type is BitstringType.INT:
            return [int(s, base=2) for s in strings]

        if output_type is BitstringType.BIT_ARRAY:
            return np.array([[b == "1" for b in s] for s in strings])

    if input_type is BitstringType.INT:
        strings = cast(list[int], strings)
        if output_type is BitstringType.STRING:
            return [f"{string:0{length}b}" for string in strings]

        if output_type is BitstringType.BIT_ARRAY:
            return np.array([[b == "1" for b in f"{s:0{length}b}"] for s in strings])

    if input_type is BitstringType.BIT_ARRAY:
        strings = cast(np.ndarray, strings)
        raise NotImplementedError


def restrict_bitstrings(
    strings: list[str] | list[int] | np.ndarray,
    indices: Sequence[int],
    bitstring_type: BitstringType,
):
    """Restrict bitstrings to a subset of bitstring indices.

    Args:
        strings: The bitstrings.
        indices: The indices of the bitstrings to restrict to.
        bitstring_type: The bitstring type of the input.

    Returns:
        The restricted bitstrings.
    """
    if bitstring_type is BitstringType.STRING:
        return [
            "".join(s[-1 - i] for i in indices[::-1]) for s in cast(list[str], strings)
        ]
    if bitstring_type is BitstringType.INT:
        return [
            sum((s >> index & 1) * (1 << i) for i, index in enumerate(indices))
            for s in cast(list[int], strings)
        ]
    if bitstring_type is BitstringType.BIT_ARRAY:
        return cast(np.ndarray, strings)[:, [-1 - i for i in indices[::-1]]]


def concatenate_bitstrings(
    strings_a: list[str] | list[int] | np.ndarray,
    strings_b: list[str] | list[int] | np.ndarray,
    bitstring_type: BitstringType,
    length: int,
):
    """Concatenate bitstrings.

    Bitstrings are concatenated in "little endian" order. That is, a bitstring from
    the first list will appear on the right side of the concatenated bitstring.

    Args:
        strings_a: The first list of bitstrings.
        strings_b: The second list of bitstrings.
        bitstring_type: The bitstring type of the input.
        length: The length of a bitstring in the first list of bitstrings.

    Returns:
        The concatenated bitstrings.
    """
    if bitstring_type is BitstringType.STRING:
        return ["".join((s_b, s_a)) for s_a, s_b in zip(strings_a, strings_b)]
    if bitstring_type is BitstringType.INT:
        return [(s_b << length) + s_a for s_a, s_b in zip(strings_a, strings_b)]
    if bitstring_type is BitstringType.BIT_ARRAY:
        return np.concatenate([strings_b, strings_a], axis=1)


def strings_to_indices(
    strings: Sequence[str], norb: int, nelec: int | tuple[int, int]
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
    addresses: Sequence[int] | np.ndarray,
    norb: int,
    nelec: int | tuple[int, int],
    concatenate: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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

    Args:
        addresses: The state vector addresses to convert to bitstrings.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        concatenate: Whether to concatenate the spin-alpha and spin-beta parts of the
            bitstrings. If True, then a single list of concatenated bitstrings is
            returned. The strings are concatenated in the order :math:`s_b s_a`,
            that is, the alpha string appears on the right.
            If False, then two lists are returned, ``(strings_a, strings_b)``. Note that
            the list of alpha strings appears first, that is, on the left.
            In the spinless case (when `nelec` is an integer), this argument is ignored.
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
    if concatenate:
        return (strings_b << norb) + strings_a
    return strings_a, strings_b


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
