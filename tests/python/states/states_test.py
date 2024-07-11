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

import itertools

import numpy as np
import pytest

import ffsim


@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(5)))
def test_slater_determinant_spinless(norb: int, nocc: int):
    """Test Slater determinant with same rotation for both spins."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nocc, seed=rng)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    eigs, orbital_rotation = np.linalg.eigh(one_body_tensor)
    eig = sum(eigs[occupied_orbitals])
    state = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    hamiltonian = ffsim.contract.one_body_linop(
        one_body_tensor, norb=norb, nelec=(nocc, 0)
    )
    np.testing.assert_allclose(hamiltonian @ state, eig * state)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_slater_determinant_same_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant with same rotation for both spins."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    occ_a, occ_b = occupied_orbitals

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=rng)
    eigs, orbital_rotation = np.linalg.eigh(one_body_tensor)
    eig = sum(eigs[occ_a]) + sum(eigs[occ_b])
    state = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    hamiltonian = ffsim.contract.one_body_linop(one_body_tensor, norb=norb, nelec=nelec)
    np.testing.assert_allclose(hamiltonian @ state, eig * state)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_slater_determinant_diff_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant with different rotations for each spin."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    occ_a, occ_b = occupied_orbitals

    orbital_rotation_a = ffsim.random.random_unitary(norb, seed=rng)
    orbital_rotation_b = ffsim.random.random_unitary(norb, seed=rng)

    state = ffsim.slater_determinant(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
    )
    state_a = ffsim.slater_determinant(
        norb,
        (occ_a, []),
        orbital_rotation=orbital_rotation_a,
    )
    state_b = ffsim.slater_determinant(
        norb,
        ([], occ_b),
        orbital_rotation=orbital_rotation_b,
    )

    np.testing.assert_allclose(state, np.kron(state_a, state_b))


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_hartree_fock_state_spinful(norb: int, nelec: tuple[int, int]):
    """Test Hartree-Fock state."""
    vec = ffsim.hartree_fock_state(norb, nelec)
    dim = ffsim.dim(norb, nelec)
    assert vec.shape == (dim,)
    assert vec[0] == 1
    assert all(vec[1:] == 0)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(5)))
def test_hartree_fock_state_spinless(norb: int, nelec: int):
    """Test Hartree-Fock state."""
    vec = ffsim.hartree_fock_state(norb, nelec)
    dim = ffsim.dim(norb, nelec)
    assert vec.shape == (dim,)
    assert vec[0] == 1
    assert all(vec[1:] == 0)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_slater_determinant_rdm1s_same_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant 1-RDM."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )
    rdm = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=orbital_rotation,
    )
    expected = ffsim.rdms(vec, norb, nelec)

    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(5)))
def test_slater_determinant_rdm1s_diff_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant 1-RDM."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    orbital_rotation_a = ffsim.random.random_unitary(norb, seed=rng)
    orbital_rotation_b = ffsim.random.random_unitary(norb, seed=rng)

    vec = ffsim.slater_determinant(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
    )
    expected = ffsim.rdms(vec, norb, nelec)

    rdm = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
    )
    np.testing.assert_allclose(rdm, expected, atol=1e-12)

    rdm = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=np.stack([orbital_rotation_a, orbital_rotation_b]),
    )
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nocc(range(5)))
def test_slater_determinant_rdm1s_spinless(norb: int, nelec: int):
    """Test Slater determinant 1-RDM, spinless."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )
    rdm = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=orbital_rotation,
    )
    expected = ffsim.rdms(vec, norb, (nelec, 0), spin_summed=True)

    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec, spin_summed",
    [
        (norb, nelec, spin_summed)
        for (norb, nelec), spin_summed in itertools.product(
            ffsim.testing.generate_norb_nelec(range(5)), [False, True]
        )
    ],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_slater_determinant_one_rdm_same_rotation(
    norb: int, nelec: tuple[int, int], spin_summed: bool
):
    """Test Slater determinant 1-RDM."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )
    rdm = ffsim.slater_determinant_rdm(
        norb,
        occupied_orbitals,
        orbital_rotation=orbital_rotation,
        spin_summed=spin_summed,
    )
    expected = ffsim.rdm(vec, norb, nelec, spin_summed=spin_summed)

    np.testing.assert_allclose(rdm, expected, atol=1e-12)


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
    "norb, nelec, spin_summed",
    [
        (norb, nelec, spin_summed)
        for (norb, nelec), spin_summed in itertools.product(
            ffsim.testing.generate_norb_nelec(range(5)), [False, True]
        )
    ],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_slater_determinant_one_rdm_diff_rotation(
    norb: int, nelec: tuple[int, int], spin_summed: bool
):
    """Test Slater determinant 1-RDM."""
    rng = np.random.default_rng()

    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)
    orbital_rotation_a = ffsim.random.random_unitary(norb, seed=rng)
    orbital_rotation_b = ffsim.random.random_unitary(norb, seed=rng)

    vec = ffsim.slater_determinant(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
    )
    rdm = ffsim.slater_determinant_rdm(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
        spin_summed=spin_summed,
    )
    expected = ffsim.rdm(vec, norb, nelec, spin_summed=spin_summed)

    np.testing.assert_allclose(rdm, expected, atol=1e-12)


def test_state_vector_array():
    """Test StateVector's __array__ method."""
    norb = 5
    nelec = (3, 2)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=3556)
    state_vec = ffsim.StateVector(vec, norb, nelec)
    assert np.array_equal(np.abs(state_vec), np.abs(vec))
