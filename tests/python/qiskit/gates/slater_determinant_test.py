# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Slater determinant preparation gate."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

import ffsim

RNG = np.random.default_rng(86887010805322231956738698580997269570)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_prepare_hartree_fock_jw(norb: int, nelec: tuple[int, int]):
    """Test preparing Hartree-Fock state."""
    gate = ffsim.qiskit.PrepareHartreeFockJW(norb, nelec)

    statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
    result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        np.array(statevec), norb=norb, nelec=nelec
    )

    expected = ffsim.hartree_fock_state(norb, nelec)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nocc", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_prepare_hartree_fock_spinless_jw(norb: int, nocc: int):
    """Test preparing Hartree-Fock state."""
    gate = ffsim.qiskit.PrepareHartreeFockSpinlessJW(norb, nocc)

    nelec = (nocc, 0)
    statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
    result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        np.array(statevec), norb=norb, nelec=nelec
    )

    expected = ffsim.hartree_fock_state(norb, nelec)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_random_slater_determinant_symmetric_spin(norb: int, nelec: tuple[int, int]):
    """Test random Slater determinant circuit gives correct output state."""
    for _ in range(3):
        occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)
        orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
        gate = ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        )

        statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.slater_determinant(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        )

        ffsim.testing.assert_allclose_up_to_global_phase(result, expected)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_random_slater_determinant_asymmetric_spin(norb: int, nelec: tuple[int, int]):
    """Test random Slater determinant circuit with independent orbital rotations."""
    for _ in range(3):
        occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)
        orbital_rotation_a = ffsim.random.random_unitary(norb, seed=RNG)
        orbital_rotation_b = ffsim.random.random_unitary(norb, seed=RNG)

        # (mat_a, mat_b)
        gate = ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb,
            occupied_orbitals,
            orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
        )
        statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.slater_determinant(
            norb,
            occupied_orbitals,
            orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
        )
        ffsim.testing.assert_allclose_up_to_global_phase(result, expected)

        # (mat_a, None)
        gate = ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb,
            occupied_orbitals,
            orbital_rotation=(orbital_rotation_a, None),
        )
        statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.slater_determinant(
            norb,
            occupied_orbitals,
            orbital_rotation=(orbital_rotation_a, None),
        )
        ffsim.testing.assert_allclose_up_to_global_phase(result, expected)

        # (None, mat_b)
        gate = ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb,
            occupied_orbitals,
            orbital_rotation=(None, orbital_rotation_b),
        )
        statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.slater_determinant(
            norb,
            occupied_orbitals,
            orbital_rotation=(None, orbital_rotation_b),
        )
        ffsim.testing.assert_allclose_up_to_global_phase(result, expected)

        # (None, None)
        gate = ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb,
            occupied_orbitals,
            orbital_rotation=(None, None),
        )
        statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        expected = ffsim.slater_determinant(
            norb,
            occupied_orbitals,
            orbital_rotation=(None, None),
        )
        ffsim.testing.assert_allclose_up_to_global_phase(result, expected)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_slater_determinant_no_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant circuit with no orbital rotation."""
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)

    gate = ffsim.qiskit.PrepareSlaterDeterminantJW(norb, occupied_orbitals)

    # Check state vector is correct
    statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
    result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
        np.array(statevec), norb=norb, nelec=nelec
    )
    expected = ffsim.slater_determinant(norb, occupied_orbitals)
    np.testing.assert_allclose(result, expected)

    # Check that the circuit only contains X gates
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(gate, qubits)
    assert circuit.decompose().count_ops().keys() in [
        set(),
        {"global_phase"},
        {"x", "global_phase"},
    ]


@pytest.mark.parametrize(
    "norb, nocc", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_random_slater_determinant_spinless(norb: int, nocc: int):
    """Test random Slater determinant circuit, spinless."""
    nelec = (nocc, 0)
    for _ in range(3):
        occupied_orbitals, _ = ffsim.testing.random_occupied_orbitals(norb, nelec)
        orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)

        gate = ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
            norb,
            occupied_orbitals,
            orbital_rotation=orbital_rotation,
        )

        statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )

        expected = ffsim.slater_determinant(
            norb,
            (occupied_orbitals, []),
            orbital_rotation=orbital_rotation,
        )

        ffsim.testing.assert_allclose_up_to_global_phase(result, expected)


def test_slater_determinant_tol():
    """Test passing tol."""
    norb = 6
    generator = 1e-8j * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)

    # Spinful
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb, (range(norb // 2), range(norb // 2)), orbital_rotation, tol=1e-7
        ),
        qubits,
    )
    assert circuit.decompose().count_ops() == {"x": norb, "global_phase": 1}

    # Spinless
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
            norb, range(norb // 2), orbital_rotation, tol=1e-7
        ),
        qubits,
    )
    assert circuit.decompose().count_ops() == {"x": norb // 2, "global_phase": 1}


@pytest.mark.parametrize("norb", [6, 7])
def test_compressed_max_givens(norb: int):
    """Test that max_givens caps the number of XXPlusYY gates."""
    n_a, n_b = 3, 2
    generator = 0.1j * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    occ = (range(n_a), range(n_b))
    max_givens = 2

    # Spinful: each spin sector contributes up to max_givens Givens rotations.
    gate = ffsim.qiskit.PrepareSlaterDeterminantJW(
        norb, occ, orbital_rotation, max_givens=max_givens
    )
    assert gate.definition.count_ops()["xx_plus_yy"] == 2 * max_givens

    # Spinless: a single sector.
    gate = ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
        norb, range(n_a), orbital_rotation, max_givens=max_givens
    )
    assert gate.definition.count_ops()["xx_plus_yy"] == max_givens


@pytest.mark.parametrize("norb", [6, 7])
def test_compressed_max_layers(norb: int):
    """Test that max_layers caps the number of XXPlusYY gates."""
    n_a, n_b = 3, 2
    generator = 0.1j * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    max_layers = 2

    # The number of retained Givens rotations per sector matches the compressed
    # decomposition.
    coeffs_a = orbital_rotation.T[list(range(n_a))]
    coeffs_b = orbital_rotation.T[list(range(n_b))]
    n_a_expected = len(
        ffsim.linalg.givens_decomposition_slater(coeffs_a, max_layers=max_layers)
    )
    n_b_expected = len(
        ffsim.linalg.givens_decomposition_slater(coeffs_b, max_layers=max_layers)
    )

    gate = ffsim.qiskit.PrepareSlaterDeterminantJW(
        norb, (range(n_a), range(n_b)), orbital_rotation, max_layers=max_layers
    )
    assert gate.definition.count_ops()["xx_plus_yy"] == n_a_expected + n_b_expected
    assert (
        gate.definition.depth(lambda instruction: instruction.operation.num_qubits == 2)
        <= max_layers
    )

    gate = ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
        norb, range(n_a), orbital_rotation, max_layers=max_layers
    )
    assert gate.definition.count_ops()["xx_plus_yy"] == n_a_expected
    assert (
        gate.definition.depth(lambda instruction: instruction.operation.num_qubits == 2)
        <= max_layers
    )


@pytest.mark.parametrize("norb, nelec", [(6, (3, 2)), (7, (2, 4))])
def test_compressed_state_fidelity(norb: int, nelec: tuple[int, int]):
    """Test that compression approximates the exact Slater determinant."""
    n_a, n_b = nelec
    occ = (range(n_a), range(n_b))
    # Near-identity rotation so even aggressive compression approximates it well.
    generator = 0.02j * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)

    expected = ffsim.slater_determinant(norb, occ, orbital_rotation=orbital_rotation)

    prev_fidelity = 0.0
    for max_layers in [1, 2, norb]:
        gate = ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb, occ, orbital_rotation, max_layers=max_layers
        )
        statevec = Statevector.from_int(0, 2 ** (2 * norb)).evolve(gate)
        result = ffsim.qiskit.qiskit_vec_to_ffsim_vec(
            np.array(statevec), norb=norb, nelec=nelec
        )
        fidelity = abs(np.vdot(expected, result)) ** 2
        # Fidelity improves (or saturates) as more layers are retained.
        assert fidelity >= prev_fidelity - 1e-9
        prev_fidelity = fidelity
    # The full decomposition reproduces the exact Slater determinant.
    assert prev_fidelity == pytest.approx(1.0)
