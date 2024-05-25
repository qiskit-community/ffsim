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

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import CircuitInstruction, QuantumCircuit
from qiskit.circuit.library import Barrier, Measure

from ffsim import gates, protocols, states
from ffsim.qiskit.gates import (
    DiagCoulombEvolutionJW,
    GivensAnsatzOperatorJW,
    GivensAnsatzOperatorSpinlessJW,
    OrbitalRotationJW,
    OrbitalRotationSpinlessJW,
    PrepareHartreeFockJW,
    PrepareHartreeFockSpinlessJW,
    PrepareSlaterDeterminantJW,
    PrepareSlaterDeterminantSpinlessJW,
    UCJOperatorJW,
)


@dataclass
class Statevector:
    """A state vector in the FCI representation.

    Attributes:
        vec: Array of state vector coefficients.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
    """

    vec: np.ndarray
    norb: int
    nelec: tuple[int, int]


def final_statevector(circuit: QuantumCircuit) -> Statevector:
    """Return the final state vector of a fermionic quantum circuit.

    Args:
        circuit: The circuit composed of fermionic gates.

    Returns:
        The final state vector that results from applying the circuit to the vacuum
        state.
    """
    if not circuit.data:
        raise ValueError("Circuit must contain at least one instruction.")
    statevector = _prepare_statevector(circuit.data[0], circuit)
    for instruction in circuit.data[1:]:
        statevector = _evolve_statevector(statevector, instruction, circuit)
    return statevector


def _prepare_statevector(
    instruction: CircuitInstruction, circuit: QuantumCircuit
) -> Statevector:
    op = instruction.operation
    qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
    consecutive_sorted = qubit_indices == list(
        range(min(qubit_indices), max(qubit_indices) + 1)
    )

    if isinstance(op, PrepareHartreeFockJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        norb = op.norb
        nelec = op.nelec
        vec = states.hartree_fock_state(norb, nelec)
        return Statevector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, PrepareHartreeFockSpinlessJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        norb = op.norb
        nelec = (op.nocc, 0)
        vec = states.hartree_fock_state(op.norb, nelec)
        return Statevector(vec=vec, norb=norb, nelec=nelec)

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
        return Statevector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, PrepareSlaterDeterminantSpinlessJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        norb = op.norb
        nelec = (len(op.occupied_orbitals), 0)
        vec = states.slater_determinant(
            op.norb, (op.occupied_orbitals, []), orbital_rotation=op.orbital_rotation
        )
        return Statevector(vec=vec, norb=norb, nelec=nelec)

    raise ValueError(
        "The first instruction of the circuit must be one of the following gates: "
        "PrepareHartreeFockJW, PrepareHartreeFockSpinlessJW, "
        "PrepareSlaterDeterminantJW, PrepareSlaterDeterminantSpinlessJW."
    )


def _evolve_statevector(
    statevector: Statevector, instruction: CircuitInstruction, circuit: QuantumCircuit
) -> Statevector:
    op = instruction.operation
    qubit_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
    consecutive_sorted = qubit_indices == list(
        range(min(qubit_indices), max(qubit_indices) + 1)
    )
    vec = statevector.vec
    norb = statevector.norb
    nelec = statevector.nelec

    if isinstance(op, DiagCoulombEvolutionJW):
        vec = gates.apply_diag_coulomb_evolution(
            vec,
            op.mat,
            op.time,
            norb=norb,
            nelec=nelec,
            mat_alpha_beta=op.mat_alpha_beta,
            z_representation=op.z_representation,
            copy=False,
        )
        return Statevector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, (GivensAnsatzOperatorJW, GivensAnsatzOperatorSpinlessJW)):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = protocols.apply_unitary(
            vec, op.givens_ansatz_operator, norb=norb, nelec=nelec, copy=False
        )
        return Statevector(vec=vec, norb=norb, nelec=nelec)

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
        return Statevector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, OrbitalRotationSpinlessJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = gates.apply_orbital_rotation(
            vec, op.orbital_rotation, norb=norb, nelec=nelec, copy=False
        )
        return Statevector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, UCJOperatorJW):
        if not consecutive_sorted:
            raise ValueError(
                f"Gate of type '{op.__class__.__name__}' must be applied to "
                "consecutive qubits, in ascending order."
            )
        vec = protocols.apply_unitary(
            vec, op.ucj_operator, norb=norb, nelec=nelec, copy=False
        )
        return Statevector(vec=vec, norb=norb, nelec=nelec)

    if isinstance(op, Barrier):
        return statevector

    if isinstance(op, Measure):
        raise ValueError(
            "Encountered a measurement gate, but only unitary operations are allowed."
        )

    raise ValueError(f"Unsupported gate: {op}.")


def sample_statevector(
    statevector: Statevector,
    *,
    indices: list[int],
    shots: int,
    seed: np.random.Generator | int | None = None,
) -> list[str]:
    """Sample bitstrings from a state vector.

    Args:
        statevector: The state vector to sample from.
        indices: The indices of the orbitals to sample from. The indices range from
            ``0`` to ``2 * norb - 1``, with the first half of the range indexing the
            spin alpha orbitals, and the second half indexing the spin beta orbitals.
        shots: The number of bitstrings to sample.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled bitstrings, as a list of strings of length `shots`.
    """
    rng = np.random.default_rng(seed)
    probabilities = np.abs(statevector.vec) ** 2
    samples = rng.choice(len(statevector.vec), size=shots, p=probabilities)
    bitstrings = states.indices_to_strings(samples, statevector.norb, statevector.nelec)
    if indices == list(range(2 * statevector.norb)):
        return bitstrings
    return [
        "".join(bitstring[-1 - i] for i in indices[::-1]) for bitstring in bitstrings
    ]
