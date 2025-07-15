import numpy as np
from qiskit.circuit import (
    ParameterVector,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library import (
    CPhaseGate,
    PhaseGate,
    XGate,
    XXPlusYYGate,
)

from ffsim import variational
from ffsim.qiskit import PrepareHartreeFockJW, UCJOpSpinBalancedJW


def ucj_spin_balanced_ansatz(
    norb: int,
    nelec: tuple[int, int],
    n_reps: int,
    interaction_pairs: tuple[
        list[tuple[int, int]] | None, list[tuple[int, int]] | None
    ],
) -> QuantumCircuit:
    r"""Gate that prepares the circuit for the Hartree-Fock state (under JWT) from
        the all zeros state followed by the LUCJ ansatz.

    This gate assumes the Jordan-Wigner transformation (JWT).

    This gate is meant to be applied to the all zeros state. It decomposes simply as
    a sequence of X gates that prepares the Hartree-Fock electronic configuration.

    This gate assumes that qubits are ordered such that the first `norb` qubits
    correspond to the alpha orbitals and the last `norb` qubits correspond to the
    beta orbitals.
    """
    n_alpha, n_beta = nelec
    assert n_alpha == n_beta

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    pairs_aa, pairs_ab = interaction_pairs

    num_param_per_brickwork = 2 * (norb // 2 + (norb - 1) // 2) * (norb // 2) + (
        norb // 2
    ) * (norb % 2)
    num_param_per_block_orbital_rotation = 2 * num_param_per_brickwork + 2 * norb
    num_param_per_spin_balance = 2 * len(pairs_aa) + len(pairs_ab)
    num_param_simplified_orbital_rotation = 2 * norb + 2 * (
        (norb - n_alpha) * n_alpha + (norb - n_beta) * n_beta
    )

    num_param = (
        num_param_simplified_orbital_rotation
        + (num_param_per_block_orbital_rotation + num_param_per_spin_balance) * n_reps
    )
    pv = ParameterVector("theta", num_param)

    def _add_orbital_rotation(local_pv: ParameterVector):
        for (i, j), id in zip(
            _brickwork(norb, norb), range(0, num_param_per_brickwork, 2)
        ):
            circuit.append(
                XXPlusYYGate(local_pv[id], local_pv[id + 1]), [qubits[i], qubits[j]]
            )
            circuit.append(
                XXPlusYYGate(
                    local_pv[id + num_param_per_brickwork],
                    local_pv[id + num_param_per_brickwork + 1],
                ),
                [qubits[norb + i], qubits[norb + j]],
            )
        for i, id in zip(
            range(2 * norb),
            range(2 * num_param_per_brickwork, 2 * num_param_per_brickwork + 2 * norb),
        ):
            circuit.append(PhaseGate(local_pv[id]), (qubits[i],))

    def _add_spin_balance(local_pv: ParameterVector):
        # gates that involve a single spin sector
        for (i, j), id in zip(pairs_aa, range(0, len(pairs_aa))):
            if i == j:
                circuit.append(PhaseGate(local_pv[id]), (qubits[i],))
                circuit.append(
                    PhaseGate(local_pv[id + len(pairs_aa)]), (qubits[i + norb],)
                )
            else:
                circuit.append(
                    CPhaseGate(local_pv[id]),
                    (qubits[i], qubits[j]),
                )
                circuit.append(
                    CPhaseGate(local_pv[id + len(pairs_aa)]),
                    (qubits[i + norb], qubits[j + norb]),
                )

        # gates that involve both spin sectors
        for (i, j), id in zip(
            pairs_ab, range(2 * len(pairs_aa), 2 * len(pairs_aa) + len(pairs_ab))
        ):
            circuit.append(CPhaseGate(local_pv[id]), (qubits[i], qubits[j + norb]))

    # add PrepareHartreeFockJW
    alpha_qubits = qubits[:norb]
    beta_qubits = qubits[norb:]
    occ_a = range(n_alpha)
    occ_b = range(n_beta)
    for orb in occ_a:
        circuit.append(XGate(), (alpha_qubits[orb],))
    for orb in occ_b:
        circuit.append(XGate(), (beta_qubits[orb],))

    # add the simplified orbital rotation
    pv_id = 0
    for n, local_qubit in zip(nelec, [alpha_qubits, beta_qubits]):
        max_gate_layer = min(n, norb - n)
        start_qubit = n - 1
        num_active_qubit = 2
        factor = 1
        if norb - n < n:
            factor = -1
        for l in range(max(norb - n, n)):
            for i in range(start_qubit, start_qubit + num_active_qubit, 2):
                circuit.append(
                    XXPlusYYGate(pv[pv_id], pv[pv_id + 1]),
                    [local_qubit[i], local_qubit[i + 1]],
                )
                pv_id += 2
            if num_active_qubit // 2 < max_gate_layer:
                start_qubit -= 1
                num_active_qubit += 2
            else:
                start_qubit += factor
        if factor == -1:
            start_qubit += 2
        for l in range(max_gate_layer - 1):
            num_active_qubit -= 2
            for i in range(start_qubit, start_qubit + num_active_qubit, 2):
                circuit.append(
                    XXPlusYYGate(pv[pv_id], pv[pv_id + 1]),
                    [local_qubit[i], local_qubit[i + 1]],
                )
                pv_id += 2
            start_qubit += 1

    for i in range(2 * norb):
        circuit.append(PhaseGate(pv[pv_id + i]), (qubits[i],))

    pv_id += 2 * norb

    for _ in range(n_reps):
        _add_spin_balance(pv[pv_id:])
        pv_id += num_param_per_spin_balance
        _add_orbital_rotation(pv[pv_id:])
        pv_id += num_param_per_block_orbital_rotation

    print(len(pv))
    return circuit


def _brickwork(norb: int, n_layers: int):
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)


def ucj_spin_balanced_parameters_from_t_amplitudes(
    t2: np.ndarray,
    *,
    norb: int,
    nelec: tuple[int, int],
    t1: np.ndarray | None = None,
    n_reps: int | None = None,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    tol: float = 1e-8,
) -> np.ndarray:
    ucj_op = variational.UCJOpSpinBalanced.from_t_amplitudes(
        t2, t1=t1, n_reps=n_reps, interaction_pairs=interaction_pairs, tol=tol
    )
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(PrepareHartreeFockJW(norb, nelec))
    circuit.append(UCJOpSpinBalancedJW(ucj_op))
    parameters = np.zeros(n_params)
