# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test states."""

from __future__ import annotations

import numpy as np

import ffsim


def test_sample_state_vector_spinful_string():
    """Test sampling state vector, spinful, output type string."""
    norb = 5
    nelec = (3, 2)
    index = ffsim.strings_to_addresses(["1000101101"], norb=norb, nelec=nelec)[0]
    vec = ffsim.linalg.one_hot(ffsim.dim(norb, nelec), index)

    samples = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec)
    assert samples == ["1000101101"]

    samples = ffsim.sample_state_vector(vec, shots=10, norb=norb, nelec=nelec)
    assert samples == ["1000101101"] * 10

    samples = ffsim.sample_state_vector(
        vec, orbs=([0, 1, 2], [0, 1, 3]), shots=10, norb=norb, nelec=nelec
    )
    assert samples == ["001101"] * 10

    samples = ffsim.sample_state_vector(
        vec,
        orbs=([0, 1, 2], [0, 1, 3]),
        shots=10,
        norb=norb,
        nelec=nelec,
        concatenate=False,
    )
    assert samples == (["101"] * 10, ["001"] * 10)


def test_sample_state_vector_spinful_int():
    """Test sampling state vector, spinful, output type int."""
    norb = 5
    nelec = (3, 2)
    index = ffsim.strings_to_addresses(["1000101101"], norb=norb, nelec=nelec)[0]
    vec = ffsim.linalg.one_hot(ffsim.dim(norb, nelec), index)

    samples = ffsim.sample_state_vector(
        vec, norb=norb, nelec=nelec, bitstring_type=ffsim.BitstringType.INT
    )
    assert samples == [0b1000101101]

    samples = ffsim.sample_state_vector(
        vec, shots=10, norb=norb, nelec=nelec, bitstring_type=ffsim.BitstringType.INT
    )
    assert samples == [0b1000101101] * 10

    samples = ffsim.sample_state_vector(
        vec,
        orbs=([0, 1, 2], [0, 1, 3]),
        shots=10,
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.INT,
    )
    assert samples == [0b001101] * 10

    samples = ffsim.sample_state_vector(
        vec,
        orbs=([0, 1, 2], [0, 1, 3]),
        shots=10,
        norb=norb,
        nelec=nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.INT,
    )
    assert samples == ([0b101] * 10, [0b001] * 10)


def test_sample_state_vector_spinful_bit_array():
    """Test sampling state vector, spinful, output type bit array."""
    norb = 5
    nelec = (3, 2)
    index = ffsim.strings_to_addresses(["1000101101"], norb=norb, nelec=nelec)[0]
    vec = ffsim.linalg.one_hot(ffsim.dim(norb, nelec), index)

    samples = ffsim.sample_state_vector(
        vec, norb=norb, nelec=nelec, bitstring_type=ffsim.BitstringType.BIT_ARRAY
    )
    np.testing.assert_array_equal(
        samples,
        np.array([[True, False, False, False, True, False, True, True, False, True]]),
    )

    samples = ffsim.sample_state_vector(
        vec,
        shots=10,
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    np.testing.assert_array_equal(
        samples,
        np.array(
            [
                [True, False, False, False, True, False, True, True, False, True],
                [True, False, False, False, True, False, True, True, False, True],
                [True, False, False, False, True, False, True, True, False, True],
                [True, False, False, False, True, False, True, True, False, True],
                [True, False, False, False, True, False, True, True, False, True],
                [True, False, False, False, True, False, True, True, False, True],
                [True, False, False, False, True, False, True, True, False, True],
                [True, False, False, False, True, False, True, True, False, True],
                [True, False, False, False, True, False, True, True, False, True],
                [True, False, False, False, True, False, True, True, False, True],
            ]
        ),
    )

    samples = ffsim.sample_state_vector(
        vec,
        orbs=([0, 1, 2], [0, 1, 3]),
        shots=10,
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    np.testing.assert_array_equal(
        samples,
        np.array(
            [
                [False, False, True, True, False, True],
                [False, False, True, True, False, True],
                [False, False, True, True, False, True],
                [False, False, True, True, False, True],
                [False, False, True, True, False, True],
                [False, False, True, True, False, True],
                [False, False, True, True, False, True],
                [False, False, True, True, False, True],
                [False, False, True, True, False, True],
                [False, False, True, True, False, True],
            ]
        ),
    )

    samples_a, samples_b = ffsim.sample_state_vector(
        vec,
        orbs=([0, 1, 2], [0, 1, 3]),
        shots=10,
        norb=norb,
        nelec=nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    np.testing.assert_array_equal(
        samples_a,
        np.array(
            [
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
            ]
        ),
    )
    np.testing.assert_array_equal(
        samples_b,
        np.array(
            [
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
                [False, False, True],
            ]
        ),
    )


def test_sample_state_vector_spinless_string():
    """Test sampling state vector, spinless, output type string."""
    norb = 5
    nelec = 3
    index = ffsim.strings_to_addresses(["01101"], norb=norb, nelec=nelec)[0]
    vec = ffsim.linalg.one_hot(ffsim.dim(norb, nelec), index)

    samples = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec)
    assert samples == ["01101"]

    samples = ffsim.sample_state_vector(vec, norb=norb, nelec=nelec, concatenate=False)
    assert samples == ["01101"]

    samples = ffsim.sample_state_vector(vec, shots=10, norb=norb, nelec=nelec)
    assert samples == ["01101"] * 10

    samples = ffsim.sample_state_vector(
        vec, orbs=[0, 1, 3], shots=10, norb=norb, nelec=nelec
    )
    assert samples == ["101"] * 10

    samples = ffsim.sample_state_vector(
        vec, orbs=[0, 1, 3], shots=10, norb=norb, nelec=nelec, concatenate=False
    )
    assert samples == ["101"] * 10


def test_sample_state_vector_spinless_int():
    """Test sampling state vector, spinless, output type int."""
    norb = 5
    nelec = 3
    index = ffsim.strings_to_addresses(["01101"], norb=norb, nelec=nelec)[0]
    vec = ffsim.linalg.one_hot(ffsim.dim(norb, nelec), index)

    samples = ffsim.sample_state_vector(
        vec, norb=norb, nelec=nelec, bitstring_type=ffsim.BitstringType.INT
    )
    assert samples == [0b01101]

    samples = ffsim.sample_state_vector(
        vec,
        norb=norb,
        nelec=nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.INT,
    )
    assert samples == [0b01101]

    samples = ffsim.sample_state_vector(
        vec, shots=10, norb=norb, nelec=nelec, bitstring_type=ffsim.BitstringType.INT
    )
    assert samples == [0b01101] * 10

    samples = ffsim.sample_state_vector(
        vec,
        orbs=[0, 1, 3],
        shots=10,
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.INT,
    )
    assert samples == [0b101] * 10

    samples = ffsim.sample_state_vector(
        vec,
        orbs=[0, 1, 3],
        shots=10,
        norb=norb,
        nelec=nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.INT,
    )
    assert samples == [0b101] * 10


def test_sample_state_vector_spinless_bit_array():
    """Test sampling state vector, spinless, output type bit_array."""
    norb = 5
    nelec = 3
    index = ffsim.strings_to_addresses(["01101"], norb=norb, nelec=nelec)[0]
    vec = ffsim.linalg.one_hot(ffsim.dim(norb, nelec), index)

    samples = ffsim.sample_state_vector(
        vec, norb=norb, nelec=nelec, bitstring_type=ffsim.BitstringType.BIT_ARRAY
    )
    np.testing.assert_array_equal(
        samples,
        np.array([[False, True, True, False, True]]),
    )

    samples = ffsim.sample_state_vector(
        vec,
        norb=norb,
        nelec=nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    np.testing.assert_array_equal(
        samples,
        np.array([[False, True, True, False, True]]),
    )

    samples = ffsim.sample_state_vector(
        vec,
        shots=10,
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    np.testing.assert_array_equal(
        samples,
        np.array(
            [
                [False, True, True, False, True],
                [False, True, True, False, True],
                [False, True, True, False, True],
                [False, True, True, False, True],
                [False, True, True, False, True],
                [False, True, True, False, True],
                [False, True, True, False, True],
                [False, True, True, False, True],
                [False, True, True, False, True],
                [False, True, True, False, True],
            ]
        ),
    )

    samples = ffsim.sample_state_vector(
        vec,
        orbs=[0, 1, 3],
        shots=10,
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    np.testing.assert_array_equal(
        samples,
        np.array(
            [
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
            ]
        ),
    )

    samples = ffsim.sample_state_vector(
        vec,
        orbs=[0, 1, 3],
        shots=10,
        norb=norb,
        nelec=nelec,
        concatenate=False,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    np.testing.assert_array_equal(
        samples,
        np.array(
            [
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
                [True, False, True],
            ]
        ),
    )


def test_state_vector_array():
    """Test StateVector's __array__ method."""
    norb = 5
    nelec = (3, 2)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=3556)
    state_vec = ffsim.StateVector(vec, norb, nelec)
    assert np.array_equal(np.abs(state_vec), np.abs(vec))
