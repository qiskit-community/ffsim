# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the quantum Schur transform gate."""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

import ffsim
from ffsim.qiskit.nonconserving import PaldusTransformJW, SchurTransform

RNG = np.random.default_rng(118704377312585236384627633567914839521)


def _clebsch_gordan_spin_half(two_j1: int, two_m: int, two_m2: int, two_j: int):
    """Clebsch-Gordan coefficient <j1, m - m2; 1/2, m2 | j, m>."""
    j1 = two_j1 / 2
    m = two_m / 2
    if abs(two_m - two_m2) > two_j1 or abs(two_m) > two_j:
        return 0.0
    if two_j == two_j1 + 1:
        if two_m2 == 1:
            return math.sqrt((j1 + m + 0.5) / (2 * j1 + 1))
        return math.sqrt((j1 - m + 0.5) / (2 * j1 + 1))
    if two_m2 == 1:
        return -math.sqrt((j1 - m + 0.5) / (2 * j1 + 1))
    return math.sqrt((j1 + m + 0.5) / (2 * j1 + 1))


def _running_two_s(yamanouchi: tuple[int, ...]) -> int:
    """Return 2S after coupling spins according to a Yamanouchi symbol."""
    return sum(1 - 2 * y for y in yamanouchi)


@functools.cache
def _schur_state(yamanouchi: tuple[int, ...], two_m: int) -> np.ndarray:
    """Return a Schur state as a vector, with qubit i holding spin i.

    Spin up is |0> and spin down is |1>. Yamanouchi bit 0 raises the total spin by
    1/2 and bit 1 lowers it by 1/2.
    """
    if not yamanouchi:
        return np.array([1.0 if two_m == 0 else 0.0])
    prefix = yamanouchi[:-1]
    two_s = _running_two_s(prefix)
    new_two_s = _running_two_s(yamanouchi)
    vec = np.zeros(1 << len(yamanouchi))
    for two_m2, spin in [(1, np.array([1.0, 0.0])), (-1, np.array([0.0, 1.0]))]:
        cg = _clebsch_gordan_spin_half(two_s, two_m, two_m2, new_two_s)
        if cg:
            # The new spin is the most significant qubit
            vec += cg * np.kron(spin, _schur_state(prefix, two_m - two_m2))
    return vec


def _schur_labels(num_spins: int) -> list[tuple[tuple[int, ...], int]]:
    """Return all valid (Yamanouchi symbol, 2M) labels."""
    labels: list[tuple[tuple[int, ...], int]] = []
    for yamanouchi in itertools.product(range(2), repeat=num_spins):
        if all(_running_two_s(yamanouchi[: i + 1]) >= 0 for i in range(num_spins)):
            two_s = _running_two_s(yamanouchi)
            labels.extend((yamanouchi, two_m) for two_m in range(-two_s, two_s + 1, 2))
    return labels


def _schur_index(gate: SchurTransform, yamanouchi: tuple[int, ...], two_m: int) -> int:
    """Return the computational basis index of a Schur basis state."""
    index = sum(y << i for i, y in enumerate(yamanouchi))
    index |= _running_two_s(yamanouchi) << gate.num_spins
    index |= (two_m % (1 << gate.num_qubits_m)) << (gate.num_spins + gate.num_qubits_s)
    return index


def _decode(index: int, gate: SchurTransform) -> tuple[tuple[int, ...], int, int]:
    """Return (Yamanouchi symbol, 2S, 2M) for a computational basis index."""
    yamanouchi = tuple((index >> i) & 1 for i in range(gate.num_spins))
    index >>= gate.num_spins
    two_s = index & ((1 << gate.num_qubits_s) - 1)
    two_m = index >> gate.num_qubits_s
    if two_m >= 1 << (gate.num_qubits_m - 1):
        two_m -= 1 << gate.num_qubits_m
    return yamanouchi, two_s, two_m


def _embed(vec: np.ndarray, num_qubits: int) -> np.ndarray:
    """Pad a vector with ancilla registers set to zero."""
    result = np.zeros(1 << num_qubits, dtype=complex)
    result[: len(vec)] = vec
    return result


def _spin_operators(num_spins: int) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Return the total spin S^2 and S_z operators."""
    s_x, s_y, s_z = (
        SparsePauliOp.from_sparse_list(
            [(pauli, [i], 0.5) for i in range(num_spins)], num_qubits=num_spins
        )
        for pauli in "XYZ"
    )
    s_squared = (s_x @ s_x + s_y @ s_y + s_z @ s_z).simplify()
    return s_squared, s_z


@pytest.mark.parametrize("num_spins", range(1, 6))
def test_schur_states(num_spins: int):
    """Test that the reference Schur states are an orthonormal basis."""
    labels = _schur_labels(num_spins)
    assert len(labels) == 2**num_spins
    mat = np.stack([_schur_state(*label) for label in labels], axis=1)
    np.testing.assert_allclose(mat.T @ mat, np.eye(2**num_spins), atol=1e-12)
    s_squared, s_z = _spin_operators(num_spins)
    for yamanouchi, two_m in labels:
        vec = _schur_state(yamanouchi, two_m)
        two_s = _running_two_s(yamanouchi)
        np.testing.assert_allclose(
            s_squared.to_matrix() @ vec, 0.25 * two_s * (two_s + 2) * vec, atol=1e-12
        )
        np.testing.assert_allclose(s_z.to_matrix() @ vec, 0.5 * two_m * vec, atol=1e-12)


@pytest.mark.parametrize("num_spins", range(1, 5))
def test_schur_transform_basis_states(num_spins: int):
    """Test that the gate maps each Schur state to a computational basis state."""
    gate = SchurTransform(num_spins)
    assert gate.num_qubits == num_spins + gate.num_qubits_s + gate.num_qubits_m
    for yamanouchi, two_m in _schur_labels(num_spins):
        vec = _embed(_schur_state(yamanouchi, two_m), gate.num_qubits)
        result = np.array(Statevector(vec).evolve(gate))
        expected = np.zeros_like(result)
        expected[_schur_index(gate, yamanouchi, two_m)] = 1
        np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("num_spins", range(1, 5))
def test_schur_transform_superposition(num_spins: int):
    """Test the gate and its inverse on a random superposition of Schur states."""
    gate = SchurTransform(num_spins)
    labels = _schur_labels(num_spins)
    coeffs = ffsim.random.random_state_vector(len(labels), seed=RNG)
    vec = _embed(
        sum(c * _schur_state(*label) for c, label in zip(coeffs, labels)),
        gate.num_qubits,
    )
    expected = np.zeros(1 << gate.num_qubits, dtype=complex)
    for c, (yamanouchi, two_m) in zip(coeffs, labels):
        expected[_schur_index(gate, yamanouchi, two_m)] = c
    result = np.array(Statevector(vec).evolve(gate))
    np.testing.assert_allclose(result, expected, atol=1e-12)
    result = np.array(Statevector(expected).evolve(gate.inverse()))
    np.testing.assert_allclose(result, vec, atol=1e-12)


@pytest.mark.parametrize("num_spins", range(1, 4))
def test_schur_transform_matches_paldus_transform(num_spins: int):
    """Test that the Paldus transform acts as the Schur transform on single occupancy.

    Spin up and spin down in orbital i correspond to occupying the spin alpha and
    spin beta orbital i, and Yamanouchi bits 0 and 1 correspond to the steps that
    raise and lower the total spin.
    """
    norb = num_spins
    schur = SchurTransform(num_spins)
    paldus = PaldusTransformJW(norb)
    for x in itertools.product(range(2), repeat=num_spins):
        # Schur transform of the computational basis state
        schur_vec = _embed(
            np.eye(1 << num_spins)[sum(b << i for i, b in enumerate(x))],
            schur.num_qubits,
        )
        schur_result = np.array(Statevector(schur_vec).evolve(schur))
        # Paldus transform of the corresponding determinant. Creating the electrons
        # in orbital order gives a sign relative to ffsim's determinant, which
        # creates all spin alpha electrons before all spin beta electrons.
        occ_a = [i for i, b in enumerate(x) if b == 0]
        occ_b = [i for i, b in enumerate(x) if b == 1]
        nelec = (len(occ_a), len(occ_b))
        sign = (-1) ** sum(
            x[i] == 1 and x[j] == 0 for i in range(norb) for j in range(i + 1, norb)
        )
        fock_vec = sign * ffsim.slater_determinant(norb, (occ_a, occ_b))
        paldus_vec = _embed(
            ffsim.qiskit.ffsim_vec_to_qiskit_vec(fock_vec, norb=norb, nelec=nelec),
            paldus.num_qubits,
        )
        paldus_result = np.array(Statevector(paldus_vec).evolve(paldus))
        # Relabel the Schur transform output as Paldus transform output
        expected = np.zeros_like(paldus_result)
        for index in np.flatnonzero(np.abs(schur_result) > 1e-12):
            yamanouchi, two_s, two_m = _decode(int(index), schur)
            paldus_index = sum(
                1 << (i if y == 0 else norb + i) for i, y in enumerate(yamanouchi)
            )
            offset = 2 * norb
            paldus_index |= num_spins << offset
            offset += paldus.num_qubits_n
            paldus_index |= two_s << offset
            offset += paldus.num_qubits_s
            paldus_index |= (two_m % (1 << paldus.num_qubits_m)) << offset
            expected[paldus_index] = schur_result[index]
        np.testing.assert_allclose(paldus_result, expected, atol=1e-12)


def test_schur_transform_invalid_num_spins():
    """Test that a non-positive number of spins raises an error."""
    with pytest.raises(ValueError, match="num_spins"):
        _ = SchurTransform(0)
