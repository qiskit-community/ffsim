# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit circuit simulation utilities."""

from __future__ import annotations

from typing import cast

from qiskit.circuit import CircuitInstruction, QuantumCircuit
from qiskit.circuit.library import Barrier, Measure

from ffsim import gates, protocols, states, trotter
from ffsim.qiskit.gates import (
    DiagCoulombEvolutionJW,
    DiagCoulombEvolutionSpinlessJW,
    GivensAnsatzOperatorJW,
    GivensAnsatzOperatorSpinlessJW,
    OrbitalRotationJW,
    OrbitalRotationSpinlessJW,
    PrepareHartreeFockJW,
    PrepareHartreeFockSpinlessJW,
    PrepareSlaterDeterminantJW,
    PrepareSlaterDeterminantSpinlessJW,
    SimulateTrotterDoubleFactorizedJW,
    UCJOperatorJW,
    UCJOpSpinBalancedJW,
    UCJOpSpinlessJW,
    UCJOpSpinUnbalancedJW,
)


def final_state_vector(circuit: QuantumCircuit) -> states.StateVector:
    """Return the final state vector of a fermionic quantum circuit.

    Args:
        circuit: The circuit composed of fermionic gates.

    Returns:
        The final state vector that results from applying the circuit to the vacuum
        state.
    """
    if not circuit.data:
        raise ValueError("Circuit must contain at least one instruction.")
    state_vector = _prepare_state_vector(circuit.data[0], circuit)
    evolve_func = (
        _evolve_state_vector_spinless
        if isinstance(state_vector.nelec, int)
        else _evolve_state_vector_spinful
    )
    for instruction in circuit.data[1:]:
        state_vector = evolve_func(state_vector, instruction, circuit)
    return state_vector


def _prepare_state_vector(
    instruction: CircuitInstruction, circuit: QuantumCircuit
) -> states.StateVector:
    op = instruction.operation
    qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
    consecutive_sorted = qubit_indices == list(
        range(min(qubit_indices), max(qubit_indices) + 1)
    )

    if isinstance(op, (PrepareHartreeFockJW, PrepareHartreeFockSpinlessJW)):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        norb = op.norb
        nelec = op.nelec
        vec = states.hartree_fock_state(norb, nelec)
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, PrepareSlaterDeterminantJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        norb = op.norb
        occ_a, occ_b = op.occupied_orbitals
        nelec = (len(occ_a), len(occ_b))
        vec = states.slater_determinant(
            norb,
            occupied_orbitals=op.occupied_orbitals,
            orbital_rotation=(op.orbital_rotation_a, op.orbital_rotation_b),
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, PrepareSlaterDeterminantSpinlessJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        norb = op.norb
        nelec = len(op.occupied_orbitals)
        vec = states.slater_determinant(
            op.norb, op.occupied_orbitals, orbital_rotation=op.orbital_rotation
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    raise ValueError(
        "The first instruction of the circuit must be one of the following gates: "
        "PrepareHartreeFockJW, PrepareHartreeFockSpinlessJW, "
        "PrepareSlaterDeterminantJW, PrepareSlaterDeterminantSpinlessJW."
    )


def _evolve_state_vector_spinless(
    state_vector: states.StateVector,
    instruction: CircuitInstruction,
    circuit: QuantumCircuit,
) -> states.StateVector:
    op = instruction.operation
    qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
    consecutive_sorted = qubit_indices == list(
        range(min(qubit_indices), max(qubit_indices) + 1)
    )
    vec = state_vector.vec
    norb = state_vector.norb
    nelec = state_vector.nelec

    if isinstance(op, DiagCoulombEvolutionSpinlessJW):
        vec = gates.apply_diag_coulomb_evolution(
            vec, op.mat, op.time, norb=norb, nelec=nelec, copy=False
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, GivensAnsatzOperatorSpinlessJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = protocols.apply_unitary(
            vec, op.givens_ansatz_operator, norb=norb, nelec=nelec, copy=False
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, OrbitalRotationSpinlessJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = gates.apply_orbital_rotation(
            vec, op.orbital_rotation, norb=norb, nelec=nelec, copy=False
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, UCJOpSpinlessJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = protocols.apply_unitary(
            vec, op.ucj_op, norb=norb, nelec=nelec, copy=False
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, Barrier):
        return state_vector

    if isinstance(op, Measure):
        raise ValueError(
            "Encountered a measurement gate, but only unitary operations are allowed."
        )

    raise ValueError(f"Unsupported gate for spinless circuit: {op}.")


def _evolve_state_vector_spinful(
    state_vector: states.StateVector,
    instruction: CircuitInstruction,
    circuit: QuantumCircuit,
) -> states.StateVector:
    op = instruction.operation
    qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
    consecutive_sorted = qubit_indices == list(
        range(min(qubit_indices), max(qubit_indices) + 1)
    )
    vec = state_vector.vec
    norb = state_vector.norb
    nelec = cast(tuple[int, int], state_vector.nelec)

    if isinstance(op, DiagCoulombEvolutionJW):
        vec = gates.apply_diag_coulomb_evolution(
            vec,
            op.mat,
            op.time,
            norb=norb,
            nelec=nelec,
            z_representation=op.z_representation,
            copy=False,
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, GivensAnsatzOperatorJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = protocols.apply_unitary(
            vec, op.givens_ansatz_operator, norb=norb, nelec=nelec, copy=False
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, OrbitalRotationJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = gates.apply_orbital_rotation(
            vec,
            (op.orbital_rotation_a, None),
            norb=norb,
            nelec=nelec,
            copy=False,
        )
        vec = gates.apply_orbital_rotation(
            vec,
            (None, op.orbital_rotation_b),
            norb=norb,
            nelec=nelec,
            copy=False,
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, SimulateTrotterDoubleFactorizedJW):
        vec = trotter.simulate_trotter_double_factorized(
            vec,
            op.hamiltonian,
            op.time,
            norb=norb,
            nelec=nelec,
            n_steps=op.n_steps,
            order=op.order,
            copy=False,
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, (UCJOpSpinBalancedJW, UCJOpSpinUnbalancedJW)):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = protocols.apply_unitary(
            vec, op.ucj_op, norb=norb, nelec=nelec, copy=False
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, UCJOperatorJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = protocols.apply_unitary(
            vec, op.ucj_operator, norb=norb, nelec=nelec, copy=False
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, Barrier):
        return state_vector

    if isinstance(op, Measure):
        raise ValueError(
            "Encountered a measurement gate, but only unitary operations are allowed."
        )

    raise ValueError(f"Unsupported gate for spinful circuit: {op}.")
