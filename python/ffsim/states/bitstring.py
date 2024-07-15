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
from typing_extensions import deprecated


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


@deprecated(
    "ffsim.indices_to_strings is deprecated. "
    "Instead, use ffsim.addresses_to_strings."
)
def indices_to_strings(
    indices: Sequence[int] | np.ndarray,
    norb: int,
    nelec: int | tuple[int, int],
    concatenate: bool = True,
    bitstring_type: BitstringType = BitstringType.STRING,
):
    """Convert state vector indices to bitstrings.

    .. warning::
        This function is deprecated. Use :class:`ffsim.addresses_to_strings` instead.

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
        list(cistring.addrs2str(norb=norb, nelec=n_alpha, addrs=indices_a)),
        input_type=BitstringType.INT,
        output_type=bitstring_type,
        length=norb,
    )
    strings_b = convert_bitstring_type(
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
    strings: Sequence[str] | Sequence[int] | np.ndarray,
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
        strings = cast(Sequence[int], strings)

        if output_type is BitstringType.STRING:
            return [f"{string:0{length}b}" for string in strings]

        if output_type is BitstringType.BIT_ARRAY:
            return np.array(
                [[s >> i & 1 for i in range(length - 1, -1, -1)] for s in strings],
                dtype=bool,
            )

    if input_type is BitstringType.BIT_ARRAY:
        strings = cast(np.ndarray, strings)

        if output_type is BitstringType.STRING:
            return ["".join("1" if bit else "0" for bit in bits) for bits in strings]

        if output_type is BitstringType.INT:
            return [
                sum(bit * (1 << i) for i, bit in enumerate(bits[::-1]))
                for bits in strings
            ]


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


@deprecated(
    "ffsim.strings_to_indices is deprecated. "
    "Instead, use ffsim.strings_to_addresses."
)
def strings_to_indices(
    strings: Sequence[str], norb: int, nelec: int | tuple[int, int]
) -> np.ndarray:
    """Convert bitstrings to state vector indices.

    .. warning::
        This function is deprecated. Use :class:`ffsim.strings_to_addresses` instead.

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
    bitstring_type: BitstringType = BitstringType.INT,
):
    """Convert state vector addresses to bitstrings.

    Example:

    .. code::

        import ffsim

        norb = 3
        nelec = (2, 1)
        dim = ffsim.dim(norb, nelec)

        strings = ffsim.addresses_to_strings(range(5), norb, nelec)
        print(strings)  # prints [11, 19, 35, 13, 21]

        strings = ffsim.addresses_to_strings(
            range(5), norb, nelec, bitstring_type=ffsim.BitstringType.STRING
        )
        print(strings)  # prints ['001011', '010011', '100011', '001101', '010101']

        strings = ffsim.addresses_to_strings(
            range(5), norb, nelec, bitstring_type=ffsim.BitstringType.BIT_ARRAY
        )
        print(strings)
        # prints
        # [[False False  True False  True  True]
        #  [False  True False False  True  True]
        #  [ True False False False  True  True]
        #  [False False  True  True False  True]
        #  [False  True False  True False  True]]

    Args:
        addresses: The state vector addresses to convert to bitstrings.
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

    Returns:
        The bitstrings. The type of the output depends on `bitstring_type` and
        `concatenate`.
    """
    if isinstance(nelec, int):
        # Spinless case
        return convert_bitstring_type(
            [
                int(s)
                for s in cistring.addrs2str(norb=norb, nelec=nelec, addrs=addresses)
            ],
            input_type=BitstringType.INT,
            output_type=bitstring_type,
            length=norb,
        )

    # Spinful case
    n_alpha, n_beta = nelec
    dim_b = math.comb(norb, n_beta)
    indices_a, indices_b = np.divmod(addresses, dim_b)
    strings_a = convert_bitstring_type(
        [int(s) for s in cistring.addrs2str(norb=norb, nelec=n_alpha, addrs=indices_a)],
        input_type=BitstringType.INT,
        output_type=bitstring_type,
        length=norb,
    )
    strings_b = convert_bitstring_type(
        [int(s) for s in cistring.addrs2str(norb=norb, nelec=n_beta, addrs=indices_b)],
        input_type=BitstringType.INT,
        output_type=bitstring_type,
        length=norb,
    )
    if concatenate:
        return concatenate_bitstrings(
            strings_a, strings_b, bitstring_type=bitstring_type, length=norb
        )
    return strings_a, strings_b


def strings_to_addresses(
    strings: Sequence[int] | Sequence[str] | np.ndarray,
    norb: int,
    nelec: int | tuple[int, int],
) -> np.ndarray:
    """Convert bitstrings to state vector addresses.

    Example:

    .. code::

        import numpy as np

        import ffsim

        norb = 3
        nelec = (2, 1)
        dim = ffsim.dim(norb, nelec)

        addresses = ffsim.strings_to_addresses(
            [
                0b001011,
                0b010101,
                0b100101,
                0b010110,
                0b100110,
            ],
            norb,
            nelec,
        )
        print(addresses)  # prints [0 4 5 7 8]

        addresses = ffsim.strings_to_addresses(
            [
                "001011",
                "010101",
                "100101",
                "010110",
                "100110",
            ],
            norb,
            nelec,
        )
        print(addresses)  # prints [0 4 5 7 8]

        addresses = ffsim.strings_to_addresses(
            np.array(
                [
                    [False, False, True, False, True, True],
                    [False, True, False, True, False, True],
                    [True, False, False, True, False, True],
                    [False, True, False, True, True, False],
                    [True, False, False, True, True, False],
                ]
            ),
            norb,
            nelec,
        )
        print(addresses)  # prints [0 4 5 7 8]

    Args:
        strings: The bitstrings to convert to state vector addresses.
            Can be a list of strings, a list of integers, or a Numpy array of bits.
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.

    Returns:
        The state vector addresses, as a Numpy array of integers.
    """
    if not len(strings):
        return np.array([])
    if isinstance(strings, np.ndarray):
        bitstring_type = BitstringType.BIT_ARRAY
    elif isinstance(strings[0], str):
        bitstring_type = BitstringType.STRING
    else:
        bitstring_type = BitstringType.INT
    strings = convert_bitstring_type(
        strings, input_type=bitstring_type, output_type=BitstringType.INT, length=norb
    )
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
