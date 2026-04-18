# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, Statevector

import ffsim

RNG = np.random.default_rng(49392490163314280547973100414356125766)


@pytest.mark.parametrize("norb", range(1, 4))
def test_yields_equivalent_circuit_spinful(norb: int):
    """Test merging orbital rotations results in an equivalent circuit, spinful."""
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb, ffsim.random.random_unitary(norb, seed=RNG)
            ),
            qubits,
        )
    for _ in range(4):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb,
                (
                    ffsim.random.random_unitary(norb, seed=RNG),
                    ffsim.random.random_unitary(norb, seed=RNG),
                ),
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert circuit.count_ops()["orb_rot_jw"] == 7
    assert transpiled.count_ops()["orb_rot_jw"] == 1
    np.testing.assert_allclose(
        np.array(Operator(circuit)), np.array(Operator(transpiled))
    )


@pytest.mark.parametrize("norb", range(1, 4))
def test_yields_equivalent_circuit_spinless(norb: int):
    """Test merging orbital rotations results in an equivalent circuit, spinless."""
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationSpinlessJW(
                norb, ffsim.random.random_unitary(norb, seed=RNG)
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert circuit.count_ops()["orb_rot_spinless_jw"] == 3
    assert transpiled.count_ops()["orb_rot_spinless_jw"] == 1
    np.testing.assert_allclose(
        np.array(Operator(circuit)), np.array(Operator(transpiled))
    )


@pytest.mark.parametrize(
    "norb, nelec",
    ffsim.testing.generate_norb_nelec(exhaustive=False, include_norb_zero=False),
)
def test_merge_slater_spinful(norb: int, nelec: tuple[int, int]):
    """Test merging orbital rotations into Slater determinant preparation, spinful."""
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb,
            ffsim.testing.random_occupied_orbitals(norb, nelec),
            ffsim.random.random_unitary(norb, seed=RNG),
        ),
        qubits,
    )
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb, ffsim.random.random_unitary(norb, seed=RNG)
            ),
            qubits,
        )
    for _ in range(4):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb,
                (
                    ffsim.random.random_unitary(norb, seed=RNG),
                    ffsim.random.random_unitary(norb, seed=RNG),
                ),
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert circuit.count_ops() == {"slater_jw": 1, "orb_rot_jw": 7}
    assert transpiled.count_ops() == {"slater_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


@pytest.mark.parametrize(
    "norb, nocc",
    ffsim.testing.generate_norb_nocc(exhaustive=False, include_norb_zero=False),
)
def test_merge_slater_spinless(norb: int, nocc: int):
    """Test merging orbital rotations into Slater determinant preparation, spinless."""
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
            norb,
            ffsim.testing.random_occupied_orbitals(norb, (nocc, 0))[0],
            ffsim.random.random_unitary(norb, seed=RNG),
        ),
        qubits,
    )
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationSpinlessJW(
                norb, ffsim.random.random_unitary(norb, seed=RNG)
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert circuit.count_ops() == {"slater_spinless_jw": 1, "orb_rot_spinless_jw": 3}
    assert transpiled.count_ops() == {"slater_spinless_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


@pytest.mark.parametrize(
    "norb, nelec",
    ffsim.testing.generate_norb_nelec(exhaustive=False, include_norb_zero=False),
)
def test_merge_hartree_fock_spinful(norb: int, nelec: tuple[int, int]):
    """Test merging orbital rotations into Hartree-Fock state preparation, spinful."""
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),
        qubits,
    )
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb, ffsim.random.random_unitary(norb, seed=RNG)
            ),
            qubits,
        )
    for _ in range(4):
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(
                norb,
                (
                    ffsim.random.random_unitary(norb, seed=RNG),
                    ffsim.random.random_unitary(norb, seed=RNG),
                ),
            ),
            qubits,
        )
    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert circuit.count_ops() == {"hartree_fock_jw": 1, "orb_rot_jw": 7}
    assert transpiled.count_ops() == {"slater_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


@pytest.mark.parametrize(
    "norb, nocc",
    ffsim.testing.generate_norb_nocc(exhaustive=False, include_norb_zero=False),
)
def test_merge_hartree_fock_spinless(norb: int, nocc: int):
    """Test merging orbital rotations into Hartree-Fock state preparation, spinless."""
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)

    circuit.append(
        ffsim.qiskit.PrepareHartreeFockSpinlessJW(norb, nocc),
        qubits,
    )
    for _ in range(3):
        circuit.append(
            ffsim.qiskit.OrbitalRotationSpinlessJW(
                norb, ffsim.random.random_unitary(norb, seed=RNG)
            ),
            qubits,
        )

    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert circuit.count_ops() == {
        "hartree_fock_spinless_jw": 1,
        "orb_rot_spinless_jw": 3,
    }
    assert transpiled.count_ops() == {"slater_spinless_jw": 1}

    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


@pytest.mark.parametrize(
    "norb, nelec",
    ffsim.testing.generate_norb_nelec(exhaustive=False, include_norb_zero=False),
)
def test_merge_ucj_spinful(norb: int, nelec: tuple[int, int]):
    """Test merging orbital rotations in UCJ operator, spinful."""
    qubits = QuantumRegister(2 * norb)
    n_reps = 3

    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),
        qubits,
    )
    ucj_op_unbalanced = ffsim.random.random_ucj_op_spin_unbalanced(
        norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=RNG
    )
    circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op_unbalanced), qubits)
    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert circuit.count_ops() == {"hartree_fock_jw": 1, "ucj_unbalanced_jw": 1}
    assert transpiled.count_ops()["slater_jw"] == 1
    assert transpiled.count_ops()["orb_rot_jw"] == n_reps
    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )

    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(norb, nelec),
        qubits,
    )
    ucj_op_balanced = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=RNG
    )
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op_balanced), qubits)
    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert circuit.count_ops() == {"hartree_fock_jw": 1, "ucj_balanced_jw": 1}
    assert transpiled.count_ops()["slater_jw"] == 1
    assert transpiled.count_ops()["orb_rot_jw"] == n_reps
    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


@pytest.mark.parametrize(
    "norb, nocc",
    ffsim.testing.generate_norb_nocc(exhaustive=False, include_norb_zero=False),
)
def test_merge_ucj_spinless(norb: int, nocc: int):
    """Test merging orbital rotations in UCJ operator, spinless."""
    qubits = QuantumRegister(norb)
    n_reps = 3

    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockSpinlessJW(norb, nocc), qubits)
    ucj_op_spinless = ffsim.random.random_ucj_op_spinless(
        norb, n_reps=n_reps, with_final_orbital_rotation=True, seed=RNG
    )
    circuit.append(ffsim.qiskit.UCJOpSpinlessJW(ucj_op_spinless), qubits)
    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert circuit.count_ops() == {"hartree_fock_spinless_jw": 1, "ucj_spinless_jw": 1}
    assert transpiled.count_ops()["slater_spinless_jw"] == 1
    assert transpiled.count_ops()["orb_rot_spinless_jw"] == n_reps
    ffsim.testing.assert_allclose_up_to_global_phase(
        np.array(Statevector(circuit)), np.array(Statevector(transpiled))
    )


def test_tol_spinful():
    """Test passing tol, spinful."""
    norb = 6
    tol = 1e-8

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    for _ in range(3):
        generator = 0.1j * tol * ffsim.random.random_hermitian(norb, seed=RNG)
        orbital_rotation = scipy.linalg.expm(generator)
        circuit.append(
            ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation, tol=tol), qubits
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert next(iter(transpiled.data)).operation.tol == tol
    assert "xx_plus_yy" not in transpiled.decompose().count_ops()


def test_tol_spinless():
    """Test passing tol, spinless."""
    norb = 6
    tol = 1e-8

    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    for _ in range(3):
        generator = 0.1j * tol * ffsim.random.random_hermitian(norb, seed=RNG)
        orbital_rotation = scipy.linalg.expm(generator)
        circuit.append(
            ffsim.qiskit.OrbitalRotationSpinlessJW(norb, orbital_rotation, tol=tol),
            qubits,
        )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert next(iter(transpiled.data)).operation.tol == tol
    assert "xx_plus_yy" not in transpiled.decompose().count_ops()


def test_tol_slater_spinful():
    """Test passing tol, Slater determinant spinful."""
    norb = 6
    nelec = (3, 3)
    slater_tol = 1e-6
    orb_rot_tol = 1e-8
    tol = max(slater_tol, orb_rot_tol)

    generator = 0.1j * tol * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb, (range(nelec[0]), range(nelec[1])), tol=slater_tol
        ),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation, tol=orb_rot_tol),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert transpiled.count_ops() == {"slater_jw": 1}
    assert next(iter(transpiled.data)).operation.tol == tol
    assert "xx_plus_yy" not in transpiled.decompose().count_ops()


def test_tol_slater_spinless():
    """Test passing tol, Slater determinant spinless."""
    norb = 6
    nocc = 3
    slater_tol = 1e-6
    orb_rot_tol = 1e-8
    tol = max(slater_tol, orb_rot_tol)

    generator = 0.1j * tol * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
            norb, range(nocc), tol=slater_tol
        ),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(norb, orbital_rotation, tol=orb_rot_tol),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert transpiled.count_ops() == {"slater_spinless_jw": 1}
    assert next(iter(transpiled.data)).operation.tol == tol
    assert "xx_plus_yy" not in transpiled.decompose().count_ops()


def test_tol_hartree_fock_spinful():
    """Test passing tol, Hartree-Fock spinful."""
    norb = 6
    nelec = (3, 3)
    slater_tol = 1e-12  # default tol from PrepareHartreeFockJW decomposition
    orb_rot_tol = 1e-8
    tol = max(slater_tol, orb_rot_tol)

    generator = 0.1j * tol * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(
        ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation, tol=orb_rot_tol), qubits
    )
    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert transpiled.count_ops() == {"slater_jw": 1}
    assert next(iter(transpiled.data)).operation.tol == tol
    assert "xx_plus_yy" not in transpiled.decompose().count_ops()


def test_tol_hartree_fock_spinless():
    """Test passing tol, Hartree-Fock spinless."""
    norb = 6
    nocc = 3
    slater_tol = 1e-12  # default tol from PrepareHartreeFockSpinlessJW decomposition
    orb_rot_tol = 1e-8
    tol = max(slater_tol, orb_rot_tol)

    generator = 0.1j * tol * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockSpinlessJW(norb, nocc), qubits)
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(norb, orbital_rotation, tol=orb_rot_tol),
        qubits,
    )
    transpiled = ffsim.qiskit.PRE_INIT.run(circuit)
    assert transpiled.count_ops() == {"slater_spinless_jw": 1}
    assert next(iter(transpiled.data)).operation.tol == tol
    assert "xx_plus_yy" not in transpiled.decompose().count_ops()
