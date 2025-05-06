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
import pytest

import ffsim

RNG = np.random.default_rng(169376298574962428809204458160326579828)


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


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (1, (0, 0)),
        (1, (0, 1)),
        (4, (2, 2)),
        (4, (2, 4)),
        (5, (3, 2)),
    ],
)
def test_state_vector_array(norb: int, nelec: tuple[int, int]):
    """Test StateVector's __array__ method."""
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    state_vec = ffsim.StateVector(vec, norb, nelec)
    assert np.array_equal(np.abs(state_vec), np.abs(vec))


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (1, (0, 0)),
        (1, (0, 1)),
        (4, (2, 2)),
        (4, (2, 4)),
        (5, (3, 2)),
    ],
)
def test_spinful_to_spinless_vec_rdm(norb: int, nelec: tuple[int, int]):
    """Test converting spinful state vector and RDMs to spinless format."""
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    spinless_vec = ffsim.spinful_to_spinless_vec(vec, norb, nelec)
    rdm1, rdm2 = ffsim.rdms(vec, norb=norb, nelec=nelec, rank=2)

    actual_rdm1 = ffsim.spinful_to_spinless_rdm1(*rdm1)
    actual_rdm2 = ffsim.spinful_to_spinless_rdm2(*rdm2)

    expected_rdm1, expected_rdm2 = ffsim.rdms(
        spinless_vec, norb=2 * norb, nelec=(sum(nelec), 0), rank=2
    )
    expected_rdm1 = expected_rdm1[0]
    expected_rdm2 = expected_rdm2[0]

    np.testing.assert_allclose(actual_rdm1, expected_rdm1, atol=1e-8)
    np.testing.assert_allclose(actual_rdm2, expected_rdm2, atol=1e-8)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (1, (0, 0)),
        (1, (0, 1)),
        (4, (2, 2)),
        (4, (2, 4)),
        (5, (3, 2)),
    ],
)
def test_spinful_to_spinless_rdm_energy(norb: int, nelec: tuple[int, int]):
    """Test converting RDMs to spinless format preserves energy."""
    mol_ham = ffsim.random.random_molecular_hamiltonian(norb, nelec)
    linop = ffsim.linear_operator(mol_ham, norb=norb, nelec=nelec)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    energy = np.vdot(vec, linop @ vec)

    rdm1, rdm2 = ffsim.rdms(vec, norb=norb, nelec=nelec, rank=2, spin_summed=True)
    energy_spinful = (
        np.einsum("pq,pq->", rdm1, mol_ham.one_body_tensor)
        + 0.5 * np.einsum("pqrs,pqrs->", rdm2, mol_ham.two_body_tensor)
        + mol_ham.constant
    )
    np.testing.assert_allclose(energy_spinful, energy, atol=1e-8)

    rdm1, rdm2 = ffsim.rdms(vec, norb=norb, nelec=nelec, rank=2, spin_summed=False)
    rdm1_spinless = ffsim.spinful_to_spinless_rdm1(*rdm1)
    rdm2_spinless = ffsim.spinful_to_spinless_rdm2(*rdm2)
    energy_spinless = (
        np.einsum("pq,pq->", rdm1_spinless, mol_ham.one_body_tensor_spinless)
        + 0.5
        * np.einsum("pqrs,pqrs->", rdm2_spinless, mol_ham.two_body_tensor_spinless)
        + mol_ham.constant
    )
    np.testing.assert_allclose(energy_spinless, energy, atol=1e-8)
