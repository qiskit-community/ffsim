# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test bitstring utilities."""

from __future__ import annotations

import numpy as np
import pytest

import ffsim


def test_indices_to_strings_string():
    """Test converting statevector indices to strings, output type string."""
    norb = 3
    nelec = 2
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.indices_to_strings(range(dim), norb, nelec)
    assert strings == [
        "011",
        "101",
        "110",
    ]
    strings = ffsim.indices_to_strings(range(dim), norb, nelec, concatenate=False)
    assert strings == [
        "011",
        "101",
        "110",
    ]

    norb = 3
    nelec = (2, 1)
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.indices_to_strings(range(dim), norb, nelec)
    assert strings == [
        "001011",
        "010011",
        "100011",
        "001101",
        "010101",
        "100101",
        "001110",
        "010110",
        "100110",
    ]
    strings_a, strings_b = ffsim.indices_to_strings(
        range(dim), norb, nelec, concatenate=False
    )
    assert strings_a == [
        "011",
        "011",
        "011",
        "101",
        "101",
        "101",
        "110",
        "110",
        "110",
    ]
    assert strings_b == [
        "001",
        "010",
        "100",
        "001",
        "010",
        "100",
        "001",
        "010",
        "100",
    ]


def test_indices_to_strings_int():
    """Test converting statevector indices to strings, output type int."""
    norb = 3
    nelec = 2
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.indices_to_strings(
        range(dim), norb, nelec, bitstring_type=ffsim.BitstringType.INT
    )
    assert strings == [
        0b011,
        0b101,
        0b110,
    ]
    strings = ffsim.indices_to_strings(
        range(dim),
        norb,
        nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.INT,
    )
    assert strings == [
        0b011,
        0b101,
        0b110,
    ]

    norb = 3
    nelec = (2, 1)
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.indices_to_strings(
        range(dim), norb, nelec, bitstring_type=ffsim.BitstringType.INT
    )
    assert strings == [
        0b001011,
        0b010011,
        0b100011,
        0b001101,
        0b010101,
        0b100101,
        0b001110,
        0b010110,
        0b100110,
    ]
    strings_a, strings_b = ffsim.indices_to_strings(
        range(dim),
        norb,
        nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.INT,
    )
    assert strings_a == [
        0b011,
        0b011,
        0b011,
        0b101,
        0b101,
        0b101,
        0b110,
        0b110,
        0b110,
    ]
    assert strings_b == [
        0b001,
        0b010,
        0b100,
        0b001,
        0b010,
        0b100,
        0b001,
        0b010,
        0b100,
    ]


def test_indices_to_strings_bit_array():
    """Test converting statevector indices to strings, output type bit array."""
    norb = 3
    nelec = 2
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.indices_to_strings(
        range(dim), norb, nelec, bitstring_type=ffsim.BitstringType.BIT_ARRAY
    )
    np.testing.assert_array_equal(
        strings,
        np.array(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ]
        ),
    )
    strings = ffsim.indices_to_strings(
        range(dim),
        norb,
        nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    np.testing.assert_array_equal(
        strings,
        np.array(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ]
        ),
    )

    norb = 3
    nelec = (2, 1)
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.indices_to_strings(
        range(dim), norb, nelec, bitstring_type=ffsim.BitstringType.BIT_ARRAY
    )
    np.testing.assert_array_equal(
        strings,
        np.array(
            [
                [False, False, True, False, True, True],
                [False, True, False, False, True, True],
                [True, False, False, False, True, True],
                [False, False, True, True, False, True],
                [False, True, False, True, False, True],
                [True, False, False, True, False, True],
                [False, False, True, True, True, False],
                [False, True, False, True, True, False],
                [True, False, False, True, True, False],
            ]
        ),
    )
    strings_a, strings_b = ffsim.indices_to_strings(
        range(dim),
        norb,
        nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    np.testing.assert_array_equal(
        strings_a,
        np.array(
            [
                [False, True, True],
                [False, True, True],
                [False, True, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, True, False],
                [True, True, False],
                [True, True, False],
            ]
        ),
    )
    np.testing.assert_array_equal(
        strings_b,
        np.array(
            [
                [False, False, True],
                [False, True, False],
                [True, False, False],
                [False, False, True],
                [False, True, False],
                [True, False, False],
                [False, False, True],
                [False, True, False],
                [True, False, False],
            ]
        ),
    )


def test_strings_to_indices():
    """Test converting statevector strings to indices."""
    norb = 3
    nelec = 2
    dim = ffsim.dim(norb, nelec)
    indices = ffsim.strings_to_indices(
        [
            "011",
            "101",
            "110",
        ],
        norb,
        nelec,
    )
    np.testing.assert_array_equal(indices, np.arange(dim))

    norb = 3
    nelec = (2, 1)
    dim = ffsim.dim(norb, nelec)
    indices = ffsim.strings_to_indices(
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
    np.testing.assert_array_equal(indices, np.arange(dim))


def test_addresses_to_strings():
    """Test converting statevector addresses to strings."""
    norb = 3
    nelec = 2
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.addresses_to_strings(range(dim), norb, nelec)
    np.testing.assert_array_equal(strings, [0b011, 0b101, 0b110])
    strings = ffsim.addresses_to_strings(range(dim), norb, nelec, concatenate=False)
    np.testing.assert_array_equal(strings, [0b011, 0b101, 0b110])

    norb = 3
    nelec = (2, 1)
    dim = ffsim.dim(norb, nelec)
    strings = ffsim.addresses_to_strings(range(dim), norb, nelec)
    np.testing.assert_array_equal(
        strings,
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
    )
    strings_a, strings_b = ffsim.addresses_to_strings(
        range(dim), norb, nelec, concatenate=False
    )
    np.testing.assert_array_equal(
        strings_a,
        [
            0b011,
            0b011,
            0b011,
            0b101,
            0b101,
            0b101,
            0b110,
            0b110,
            0b110,
        ],
    )
    np.testing.assert_array_equal(
        strings_b,
        [
            0b001,
            0b010,
            0b100,
            0b001,
            0b010,
            0b100,
            0b001,
            0b010,
            0b100,
        ],
    )


def test_strings_to_addresses():
    """Test converting statevector strings to addresses."""
    norb = 3
    nelec = 2
    dim = ffsim.dim(norb, nelec)
    indices = ffsim.strings_to_addresses(
        [0b011, 0b101, 0b110],
        norb,
        nelec,
    )
    np.testing.assert_array_equal(indices, np.arange(dim))

    norb = 3
    nelec = (2, 1)
    dim = ffsim.dim(norb, nelec)
    indices = ffsim.strings_to_addresses(
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
    np.testing.assert_array_equal(indices, np.arange(dim))


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(1, 6)))
def test_indices_and_strings_roundtrip_spinless(norb: int, nelec: int):
    """Test converting statevector indices to strings."""
    rng = np.random.default_rng(26390)
    dim = ffsim.dim(norb, nelec)
    indices = rng.choice(dim, size=10)
    strings = ffsim.indices_to_strings(indices, norb=norb, nelec=nelec)
    indices_again = ffsim.strings_to_indices(strings, norb=norb, nelec=nelec)
    np.testing.assert_array_equal(indices_again, indices)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 6)))
def test_indices_and_strings_roundtrip_spinful(norb: int, nelec: tuple[int, int]):
    """Test converting statevector indices to strings."""
    rng = np.random.default_rng(26390)
    dim = ffsim.dim(norb, nelec)
    indices = rng.choice(dim, size=10)
    strings = ffsim.indices_to_strings(
        indices, norb=norb, nelec=nelec, concatenate=True
    )
    indices_again = ffsim.strings_to_indices(strings, norb=norb, nelec=nelec)
    np.testing.assert_array_equal(indices_again, indices)
