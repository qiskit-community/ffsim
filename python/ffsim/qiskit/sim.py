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

import cmath
import math
from typing import cast

from qiskit.circuit import CircuitInstruction, QuantumCircuit
from qiskit.circuit.library import (
    Barrier,
    CPhaseGate,
    Measure,
    PhaseGate,
    RZGate,
    RZZGate,
    XGate,
    XXPlusYYGate,
)

from ffsim import gates, protocols, states, trotter
from ffsim.qiskit.gates import (
    DiagCoulombEvolutionJW,
    DiagCoulombEvolutionSpinlessJW,
    GivensAnsatzOpJW,
    GivensAnsatzOpSpinlessJW,
    OrbitalRotationJW,
    OrbitalRotationSpinlessJW,
    PrepareHartreeFockJW,
    PrepareHartreeFockSpinlessJW,
    PrepareSlaterDeterminantJW,
    PrepareSlaterDeterminantSpinlessJW,
    SimulateTrotterDoubleFactorizedJW,
    UCJOpSpinBalancedJW,
    UCJOpSpinlessJW,
    UCJOpSpinUnbalancedJW,
)
from ffsim.spin import Spin


def final_state_vector(
    circuit: QuantumCircuit,
    norb: int | None = None,
    nelec: tuple[int, int] | None = None,
) -> states.StateVector:
    """Return the final state vector of a fermionic quantum circuit.

    Args:
        circuit: The circuit composed of fermionic gates.

    Returns:
        The final state vector that results from applying the circuit to the vacuum
        state.
    """
    if not circuit.data:
        raise ValueError("Circuit must contain at least one instruction.")
    state_vector, index = _prepare_state_vector(circuit, norb, nelec)
    if norb is not None and norb != state_vector.norb:
        raise ValueError(
            "norb did not match the circuit's state preparation. "
            f"Got {norb}, but the circuit's state preparation gave "
            f"{state_vector.norb}. Try omitting the norb argument."
        )
    if nelec is not None and nelec != state_vector.nelec:
        raise ValueError(
            "nelec did not match the circuit's state preparation. "
            f"Got {nelec}, but the circuit's state preparation gave "
            f"{state_vector.nelec}. Try omitting the nelec argument."
        )
    evolve_func = (
        _evolve_state_vector_spinless
        if isinstance(state_vector.nelec, int)
        else _evolve_state_vector_spinful
    )
    for instruction in circuit.data[index:]:
        state_vector = evolve_func(state_vector, instruction, circuit)
    return state_vector


def _prepare_state_vector(
    circuit: QuantumCircuit, norb: int | None, nelec: tuple[int, int] | None
) -> tuple[states.StateVector, int]:
    instruction = circuit.data[0]
    op = instruction.operation
    if isinstance(
        op,
        (
            PrepareHartreeFockJW,
            PrepareHartreeFockSpinlessJW,
            PrepareSlaterDeterminantJW,
            PrepareSlaterDeterminantSpinlessJW,
        ),
    ):
        # First instruction is an ffsim gate
        qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if not qubit_indices == list(range(min(qubit_indices), max(qubit_indices) + 1)):
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )

        if isinstance(op, (PrepareHartreeFockJW, PrepareHartreeFockSpinlessJW)):
            norb = op.norb
            nelec = op.nelec
            vec = states.hartree_fock_state(norb, nelec)
            return states.StateVector(vec=vec, norb=norb, nelec=nelec), 1

        if isinstance(op, PrepareSlaterDeterminantJW):
            norb = op.norb
            occ_a, occ_b = op.occupied_orbitals
            nelec = (len(occ_a), len(occ_b))
            vec = states.slater_determinant(
                norb,
                occupied_orbitals=op.occupied_orbitals,
                orbital_rotation=(op.orbital_rotation_a, op.orbital_rotation_b),
            )
            return states.StateVector(vec=vec, norb=norb, nelec=nelec), 1

        if isinstance(op, PrepareSlaterDeterminantSpinlessJW):
            norb = op.norb
            nelec = len(op.occupied_orbitals)
            vec = states.slater_determinant(
                op.norb, op.occupied_orbitals, orbital_rotation=op.orbital_rotation
            )
            return states.StateVector(vec=vec, norb=norb, nelec=nelec), 1

    elif isinstance(op, XGate):
        qubit_indices = []
        for index, instruction in enumerate(circuit.data):
            op = instruction.operation
            if not isinstance(op, XGate):
                break
            (qubit,) = instruction.qubits
            qubit_indices.append(circuit.find_bit(qubit).index)
        # TODO handle norb and nelec being None
        if nelec is None or isinstance(nelec, int):
            # Spinless case
            norb = circuit.num_qubits
            nelec = len(qubit_indices)
            vec = states.slater_determinant(norb, qubit_indices)
            return states.StateVector(vec=vec, norb=norb, nelec=nelec), index + 1
        else:
            # Spinful case
            assert circuit.num_qubits % 2 == 0
            norb = circuit.num_qubits // 2
            occ_a = [i for i in qubit_indices if i < norb]
            occ_b = [i - norb for i in qubit_indices if i >= norb]
            nelec = (len(occ_a), len(occ_b))
            vec = states.slater_determinant(norb, (occ_a, occ_b))
            return states.StateVector(vec=vec, norb=norb, nelec=nelec), index + 1

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

    if isinstance(op, GivensAnsatzOpSpinlessJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = protocols.apply_unitary(
            vec, op.givens_ansatz_op, norb=norb, nelec=nelec, copy=False
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

    if isinstance(op, CPhaseGate):
        i, j = qubit_indices
        (theta,) = op.params
        vec = gates.apply_num_num_interaction(
            vec,
            theta,
            target_orbs=(i, j),
            norb=norb,
            nelec=nelec,
            copy=False,
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, PhaseGate):
        (orb,) = qubit_indices
        (theta,) = op.params
        vec = gates.apply_num_interaction(
            vec,
            theta,
            orb,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, RZGate):
        (orb,) = qubit_indices
        (theta,) = op.params
        vec = gates.apply_num_interaction(
            vec,
            theta,
            orb,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
        vec *= cmath.rect(1, -0.5 * theta)
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, XXPlusYYGate):
        i, j = qubit_indices
        if not abs(i - j) == 1:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits."
            )
        theta, beta = op.params
        vec = gates.apply_givens_rotation(
            vec,
            0.5 * theta,
            (i, j),
            phi=-beta - 0.5 * math.pi,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

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

    if isinstance(op, GivensAnsatzOpJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = protocols.apply_unitary(
            vec, op.givens_ansatz_op, norb=norb, nelec=nelec, copy=False
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

    if isinstance(op, Barrier):
        return state_vector

    if isinstance(op, Measure):
        raise ValueError(
            "Encountered a measurement gate, but only unitary operations are allowed."
        )

    if isinstance(op, CPhaseGate):
        i, j = qubit_indices
        target_orbs = ([], [])
        target_orbs[i >= norb].append(i % norb)
        target_orbs[j >= norb].append(j % norb)
        (theta,) = op.params
        vec = gates.apply_num_op_prod_interaction(
            vec,
            theta,
            target_orbs=target_orbs,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, PhaseGate):
        (orb,) = qubit_indices
        spin = Spin.ALPHA if orb < norb else Spin.BETA
        (theta,) = op.params
        vec = gates.apply_num_interaction(
            vec,
            theta,
            orb % norb,
            norb=norb,
            nelec=nelec,
            spin=spin,
            copy=False,
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, RZGate):
        (orb,) = qubit_indices
        spin = Spin.ALPHA if orb < norb else Spin.BETA
        (theta,) = op.params
        vec = gates.apply_num_interaction(
            vec,
            theta,
            orb % norb,
            norb=norb,
            nelec=nelec,
            spin=spin,
            copy=False,
        )
        vec *= cmath.rect(1, -0.5 * theta)
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, RZZGate):
        i, j = qubit_indices
        target_orbs = ([], [])
        target_orbs[i >= norb].append(i % norb)
        target_orbs[j >= norb].append(j % norb)
        (theta,) = op.params
        vec = gates.apply_num_op_prod_interaction(
            vec,
            -2 * theta,
            target_orbs=target_orbs,
            norb=norb,
            nelec=nelec,
            copy=False,
        )
        vec = gates.apply_num_interaction(
            vec,
            theta,
            i % norb,
            norb=norb,
            nelec=nelec,
            spin=Spin.ALPHA if i < norb else Spin.BETA,
            copy=False,
        )
        vec = gates.apply_num_interaction(
            vec,
            theta,
            j % norb,
            norb=norb,
            nelec=nelec,
            spin=Spin.ALPHA if j < norb else Spin.BETA,
            copy=False,
        )
        vec *= cmath.rect(1, -theta)
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, XXPlusYYGate):
        i, j = qubit_indices
        if not abs(i - j) == 1:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits."
            )
        if (i < norb) != (j < norb):
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied on orbitals "
                "of the same spin."
            )
        spin = Spin.ALPHA if i < norb else Spin.BETA
        theta, beta = op.params
        vec = gates.apply_givens_rotation(
            vec,
            0.5 * theta,
            (i % norb, j % norb),
            phi=-beta - 0.5 * math.pi,
            norb=norb,
            nelec=nelec,
            spin=spin,
            copy=False,
        )
        return states.StateVector(vec=vec, norb=norb, nelec=nelec)

    raise ValueError(f"Unsupported gate for spinful circuit: {op}.")
