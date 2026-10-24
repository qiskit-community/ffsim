# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the quantum Paldus transform gate."""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

import ffsim
from ffsim.qiskit.nonconserving import PaldusTransformJW

RNG = np.random.default_rng(301868447427305116383364536052233740717)


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


def _gt_operator(steps: tuple[int, ...], two_m: int) -> ffsim.FermionOperator:
    """Operator that creates a Gelfand-Tsetlin state from the vacuum.

    The orbitals are coupled one at a time in order. Step 0 is an empty orbital,
    step 1 raises the spin by 1/2, step 2 lowers the spin by 1/2, and step 3 is a
    doubly occupied orbital.
    """
    # Each partial term is (creation operators, intermediate 2M) -> coefficient
    terms: dict[tuple[tuple[tuple[bool, bool, int], ...], int], float] = {((), 0): 1.0}
    two_s = 0
    for orb, step in enumerate(steps):
        new_terms: dict = {}
        if step in (0, 3):
            ops = () if step == 0 else (ffsim.cre_a(orb), ffsim.cre_b(orb))
            for (prev_ops, prev_two_m), coeff in terms.items():
                new_terms[prev_ops + ops, prev_two_m] = coeff
        else:
            new_two_s = two_s + 1 if step == 1 else two_s - 1
            for (prev_ops, prev_two_m), coeff in terms.items():
                for two_m2, op in [(1, ffsim.cre_a(orb)), (-1, ffsim.cre_b(orb))]:
                    new_two_m = prev_two_m + two_m2
                    cg = _clebsch_gordan_spin_half(two_s, new_two_m, two_m2, new_two_s)
                    if cg:
                        key = ((*prev_ops, op), new_two_m)
                        new_terms[key] = new_terms.get(key, 0.0) + coeff * cg
            two_s = new_two_s
        terms = new_terms
    return ffsim.FermionOperator(
        {
            ops: coeff
            for (ops, this_two_m), coeff in terms.items()
            if this_two_m == two_m
        }
    )


def _uga_labels(norb: int) -> list[tuple[tuple[int, ...], int]]:
    """Return all valid (step vector, 2M) labels."""
    labels: list[tuple[tuple[int, ...], int]] = []
    for steps in itertools.product(range(4), repeat=norb):
        two_s = 0
        for step in steps:
            two_s += (step == 1) - (step == 2)
            if two_s < 0:
                break
        else:
            labels.extend((steps, two_m) for two_m in range(-two_s, two_s + 1, 2))
    return labels


def _quantum_numbers(steps: tuple[int, ...]) -> tuple[int, int]:
    """Return N and 2S for a step vector."""
    n = sum((step in (1, 2)) + 2 * (step == 3) for step in steps)
    two_s = sum((step == 1) - (step == 2) for step in steps)
    return n, two_s


def _uga_index(gate: PaldusTransformJW, steps: tuple[int, ...], two_m: int) -> int:
    """Return the computational basis index of a UGA basis state."""
    norb = gate.norb
    n, two_s = _quantum_numbers(steps)
    index = 0
    for orb, step in enumerate(steps):
        index |= (step in (1, 3)) << orb
        index |= (step in (2, 3)) << (norb + orb)
    offset = 2 * norb
    index |= n << offset
    offset += gate.num_qubits_n
    index |= two_s << offset
    offset += gate.num_qubits_s
    index |= (two_m % (1 << gate.num_qubits_m)) << offset
    return index


@functools.cache
def _gt_states(norb: int) -> dict[tuple[tuple[int, ...], int], np.ndarray]:
    """Return the Gelfand-Tsetlin states as vectors on 2 * norb qubits."""
    vacuum = np.zeros(1 << (2 * norb), dtype=complex)
    vacuum[0] = 1
    states = {}
    for steps, two_m in _uga_labels(norb):
        op = ffsim.qiskit.jordan_wigner(_gt_operator(steps, two_m), norb)
        states[steps, two_m] = op.to_matrix(sparse=True) @ vacuum
    return states


def _embed(vec: np.ndarray, gate: PaldusTransformJW) -> np.ndarray:
    """Pad a vector on the fermionic qubits with ancilla registers set to zero."""
    result = np.zeros(1 << gate.num_qubits, dtype=complex)
    result[: len(vec)] = vec
    return result


@pytest.mark.parametrize("norb", range(1, 4))
def test_gt_states(norb: int):
    """Test that the reference Gelfand-Tsetlin states are an orthonormal basis."""
    states = _gt_states(norb)
    assert len(states) == 4**norb
    mat = np.stack(list(states.values()), axis=1)
    np.testing.assert_allclose(mat.T.conj() @ mat, np.eye(4**norb), atol=1e-12)
    s_squared = ffsim.qiskit.jordan_wigner(ffsim.s_squared_operator(norb), norb)
    s_z = ffsim.qiskit.jordan_wigner(ffsim.s_z_operator(norb), norb)
    for (steps, two_m), vec in states.items():
        _, two_s = _quantum_numbers(steps)
        np.testing.assert_allclose(
            s_squared.to_matrix(sparse=True) @ vec,
            0.25 * two_s * (two_s + 2) * vec,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            s_z.to_matrix(sparse=True) @ vec, 0.5 * two_m * vec, atol=1e-12
        )


@pytest.mark.parametrize("norb", range(1, 4))
def test_paldus_transform_basis_states(norb: int):
    """Test that the gate maps each Gelfand-Tsetlin state to a UGA basis state."""
    gate = PaldusTransformJW(norb)
    assert gate.num_qubits == (
        2 * norb + gate.num_qubits_n + gate.num_qubits_s + gate.num_qubits_m
    )
    for (steps, two_m), vec in _gt_states(norb).items():
        result = np.array(Statevector(_embed(vec, gate)).evolve(gate))
        expected = np.zeros_like(result)
        expected[_uga_index(gate, steps, two_m)] = 1
        np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("norb", range(1, 4))
def test_paldus_transform_superposition(norb: int):
    """Test the gate and its inverse on a random superposition of GT states."""
    gate = PaldusTransformJW(norb)
    states = _gt_states(norb)
    coeffs = ffsim.random.random_state_vector(len(states), seed=RNG)
    vec = sum(c * v for c, v in zip(coeffs, states.values()))
    expected = np.zeros(1 << gate.num_qubits, dtype=complex)
    for c, (steps, two_m) in zip(coeffs, states):
        expected[_uga_index(gate, steps, two_m)] = c
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, range(gate.num_qubits))
    result = np.array(Statevector(_embed(vec, gate)).evolve(circuit))
    np.testing.assert_allclose(result, expected, atol=1e-12)
    result = np.array(Statevector(expected).evolve(gate.inverse()))
    np.testing.assert_allclose(result, _embed(vec, gate), atol=1e-12)


def test_paldus_transform_paper_examples():
    """Test the norb = 2 examples of mixed occupation from the paper."""
    norb = 2
    gate = PaldusTransformJW(norb)
    vacuum = np.zeros(1 << (2 * norb), dtype=complex)
    vacuum[0] = 1
    r = 1 / math.sqrt(2)
    cases: list[
        tuple[dict[tuple[tuple[bool, bool, int], ...], complex], tuple[int, ...], int]
    ] = [
        # Singlet, N = 2, S = 0, d = (1, 2)
        (
            {(ffsim.cre_a(0), ffsim.cre_b(1)): r, (ffsim.cre_b(0), ffsim.cre_a(1)): -r},
            (1, 2),
            0,
        ),
        # Triplet, N = 2, S = 1, M = 0, d = (1, 1)
        (
            {(ffsim.cre_a(0), ffsim.cre_b(1)): r, (ffsim.cre_b(0), ffsim.cre_a(1)): r},
            (1, 1),
            0,
        ),
        # N = 3, S = 1/2, M = -1/2, d = (3, 1)
        ({(ffsim.cre_a(0), ffsim.cre_b(0), ffsim.cre_b(1)): 1}, (3, 1), -1),
    ]
    for coeffs, steps, two_m in cases:
        op = ffsim.FermionOperator(coeffs)
        vec = ffsim.qiskit.jordan_wigner(op, norb).to_matrix() @ vacuum
        result = np.array(Statevector(_embed(vec, gate)).evolve(gate))
        expected = np.zeros_like(result)
        expected[_uga_index(gate, steps, two_m)] = 1
        np.testing.assert_allclose(result, expected, atol=1e-12)


def test_paldus_transform_invalid_norb():
    """Test that a non-positive number of orbitals raises an error."""
    with pytest.raises(ValueError, match="norb"):
        _ = PaldusTransformJW(0)
