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


def test_compression_preserved_spinful():
    """Test that compression settings survive merging, spinful."""
    norb = 6
    max_givens = 4
    max_layers = 2

    # A single orbital rotation forms a run of length one; the merged gate must still
    # carry the compression settings through the pass.
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    generator = 0.1j * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    circuit.append(
        ffsim.qiskit.OrbitalRotationJW(
            norb, orbital_rotation, max_givens=max_givens, max_layers=max_layers
        ),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    merged_op = next(iter(transpiled.data)).operation
    assert merged_op.max_givens == max_givens
    assert merged_op.max_layers == max_layers
    # The compressed decomposition uses at most max_givens Givens rotations per sector.
    assert transpiled.decompose().count_ops()["xx_plus_yy"] <= 2 * max_givens

    # When merging gates with different budgets, the tightest one wins (min, ignoring
    # None), mirroring how tol is combined with max.
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation), qubits)
    circuit.append(
        ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation, max_layers=max_layers),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.OrbitalRotationJW(
            norb, orbital_rotation, max_layers=max_layers + 3
        ),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    merged_op = next(iter(transpiled.data)).operation
    assert merged_op.max_givens is None
    assert merged_op.max_layers == max_layers


def test_compression_preserved_spinless():
    """Test that compression settings survive merging, spinless."""
    norb = 6
    max_givens = 4
    max_layers = 2

    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    generator = 0.1j * ffsim.random.random_hermitian(norb, seed=RNG)
    orbital_rotation = scipy.linalg.expm(generator)
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb, orbital_rotation, max_givens=max_givens, max_layers=max_layers
        ),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    merged_op = next(iter(transpiled.data)).operation
    assert merged_op.max_givens == max_givens
    assert merged_op.max_layers == max_layers
    assert transpiled.decompose().count_ops()["xx_plus_yy"] <= max_givens

    # When merging gates with different budgets, the tightest one wins.
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(norb, orbital_rotation), qubits
    )
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb, orbital_rotation, max_layers=max_layers
        ),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb, orbital_rotation, max_layers=max_layers + 3
        ),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    merged_op = next(iter(transpiled.data)).operation
    assert merged_op.max_givens is None
    assert merged_op.max_layers == max_layers


def test_compression_preserved_slater_spinful():
    """Test that compression settings survive Slater merging, spinful."""
    norb = 6
    max_layers = 2
    occ = (range(3), range(2))

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb,
            occ,
            ffsim.random.random_unitary(norb, seed=RNG),
            max_layers=max_layers,
        ),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.OrbitalRotationJW(
            norb, ffsim.random.random_unitary(norb, seed=RNG)
        ),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert transpiled.count_ops() == {"slater_jw": 1}
    merged_op = next(iter(transpiled.data)).operation
    assert merged_op.max_layers == max_layers


def test_compression_preserved_slater_spinless():
    """Test that compression settings survive Slater merging, spinless."""
    norb = 6
    max_layers = 2

    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
            norb,
            range(3),
            ffsim.random.random_unitary(norb, seed=RNG),
            max_layers=max_layers,
        ),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb, ffsim.random.random_unitary(norb, seed=RNG)
        ),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert transpiled.count_ops() == {"slater_spinless_jw": 1}
    merged_op = next(iter(transpiled.data)).operation
    assert merged_op.max_layers == max_layers


def test_slater_merge_ignores_orbital_rotation_budget_spinful():
    """Test that Slater merging keeps the Slater budget and ignores the rotation's."""
    norb = 6
    slater_max_layers = 2
    orb_rot_max_layers = 4
    occ = (range(3), range(2))

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb,
            occ,
            ffsim.random.random_unitary(norb, seed=RNG),
            max_layers=slater_max_layers,
        ),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.OrbitalRotationJW(
            norb,
            ffsim.random.random_unitary(norb, seed=RNG),
            max_layers=orb_rot_max_layers,
        ),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert transpiled.count_ops() == {"slater_jw": 1}
    merged_op = next(iter(transpiled.data)).operation
    # The absorbed orbital rotation's (brickwork-scale) budget is discarded; the
    # Slater determinant preparation's own (diamond-scale) budget is retained.
    assert merged_op.max_layers == slater_max_layers


def test_slater_merge_ignores_orbital_rotation_budget_spinless():
    """Test that Slater merging keeps the Slater budget and ignores the rotation's."""
    norb = 6
    slater_max_layers = 2
    orb_rot_max_layers = 4

    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
            norb,
            range(3),
            ffsim.random.random_unitary(norb, seed=RNG),
            max_layers=slater_max_layers,
        ),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(
            norb,
            ffsim.random.random_unitary(norb, seed=RNG),
            max_layers=orb_rot_max_layers,
        ),
        qubits,
    )
    transpiled = ffsim.qiskit.MergeOrbitalRotations()(circuit)
    assert transpiled.count_ops() == {"slater_spinless_jw": 1}
    merged_op = next(iter(transpiled.data)).operation
    # The absorbed orbital rotation's (brickwork-scale) budget is discarded; the
    # Slater determinant preparation's own (diamond-scale) budget is retained.
    assert merged_op.max_layers == slater_max_layers
