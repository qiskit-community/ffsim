# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for parameterized Qiskit LUCJ ansatz circuits."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator

import ffsim

RNG = np.random.default_rng(211287762716487520063937912823535598337)


def _merged_ucj_circuit(ucj_op: ffsim.UCJOpSpinBalanced) -> QuantumCircuit:
    norb = ucj_op.norb
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    return ffsim.qiskit.PRE_INIT.run(circuit).decompose()


def test_lucj_spin_balanced_ansatz_parameter_count():
    norb = 3
    n_reps = 2
    pairs_aa = [(0, 1), (1, 2)]
    pairs_ab = [(0, 0), (0, 1), (2, 2)]

    ansatz = ffsim.qiskit.lucj_spin_balanced_ansatz(
        norb, n_reps, interaction_pairs=(pairs_aa, pairs_ab)
    )

    n_orbital_rotation_params = 2 * norb**2
    n_diag_params = 2 * len(pairs_aa) + sum(1 if i == j else 2 for i, j in pairs_ab)
    expected = (n_reps + 1) * n_orbital_rotation_params + n_reps * n_diag_params
    assert ansatz.num_parameters == expected


@pytest.mark.parametrize(
    "interaction_pairs",
    [
        None,
        ([(0, 1), (1, 2)], [(0, 0), (0, 2), (2, 2)]),
        ([], [(0, 0)]),
        ([(0, 2)], []),
    ],
)
def test_lucj_spin_balanced_parameters_match_merged_ucj(interaction_pairs):
    norb = 3
    n_reps = 2
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=True,
        seed=RNG,
    )

    ansatz = ffsim.qiskit.lucj_spin_balanced_ansatz(
        norb, n_reps, interaction_pairs=interaction_pairs
    )
    values = ffsim.qiskit.lucj_spin_balanced_parameters(
        ucj_op, interaction_pairs=interaction_pairs
    )

    bound = ansatz.assign_parameters(values)
    expected = _merged_ucj_circuit(ucj_op)
    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Operator(bound)), np.array(Operator(expected))
    )


def test_lucj_spin_balanced_parameters_from_t2_match_merged_ucj():
    norb = 4
    nocc = 2
    n_reps = 2
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=float)
    t1 = RNG.standard_normal((nocc, norb - nocc))
    interaction_pairs = ([(0, 1), (2, 3)], [(0, 0), (1, 2), (3, 3)])

    ansatz = ffsim.qiskit.lucj_spin_balanced_ansatz(
        norb, n_reps, interaction_pairs=interaction_pairs
    )
    values = ffsim.qiskit.lucj_spin_balanced_parameters_from_t2(
        t2, t1=t1, n_reps=n_reps, interaction_pairs=interaction_pairs
    )

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2, t1=t1, n_reps=n_reps, interaction_pairs=interaction_pairs
    )
    bound = ansatz.assign_parameters(values)
    expected = _merged_ucj_circuit(ucj_op)
    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Operator(bound)), np.array(Operator(expected))
    )


def test_lucj_spin_balanced_parameters_from_zero_t2_keeps_full_length():
    norb = 4
    nocc = 2
    n_reps = 3
    t2 = np.zeros((nocc, nocc, norb - nocc, norb - nocc))
    t1 = np.zeros((nocc, norb - nocc))
    interaction_pairs = ([(0, 1)], [(0, 0), (1, 2)])

    ansatz = ffsim.qiskit.lucj_spin_balanced_ansatz(
        norb, n_reps, interaction_pairs=interaction_pairs
    )
    values = ffsim.qiskit.lucj_spin_balanced_parameters_from_t2(
        t2, t1=t1, n_reps=n_reps, interaction_pairs=interaction_pairs
    )

    assert len(values) == ansatz.num_parameters
    bound = ansatz.assign_parameters(values)
    np.testing.assert_allclose(np.array(Operator(bound)), np.eye(2 ** (2 * norb)))


def test_lucj_spin_balanced_ansatz_validates_inputs():
    with pytest.raises(ValueError, match="norb"):
        ffsim.qiskit.lucj_spin_balanced_ansatz(0, 1)
    with pytest.raises(ValueError, match="n_reps"):
        ffsim.qiskit.lucj_spin_balanced_ansatz(2, 0)
    with pytest.raises(ValueError, match="triangular"):
        ffsim.qiskit.lucj_spin_balanced_ansatz(2, 1, interaction_pairs=([(1, 0)], []))
    with pytest.raises(ValueError, match="Duplicate"):
        ffsim.qiskit.lucj_spin_balanced_ansatz(
            2, 1, interaction_pairs=([], [(0, 1), (0, 1)])
        )


def test_lucj_public_exports():
    names = set(ffsim.qiskit.__all__)
    for name in [
        "lucj_spin_balanced_ansatz",
        "lucj_spin_balanced_parameters",
        "lucj_spin_balanced_parameters_from_t2",
    ]:
        assert name in names

    assert ffsim.qiskit.lucj_spin_balanced_ansatz is not None
    assert ffsim.qiskit.lucj_spin_balanced_parameters is not None
    assert ffsim.qiskit.lucj_spin_balanced_parameters_from_t2 is not None
