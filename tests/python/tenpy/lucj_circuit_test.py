import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit, QuantumRegister

import ffsim
from ffsim.tenpy.circuits.lucj_circuit import lucj_circuit_as_mps


def _interaction_pairs_spin_balanced_(
    connectivity: str, norb: int
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]:
    """Returns alpha-alpha and alpha-beta diagonal Coulomb interaction pairs."""
    if connectivity == "square":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb)]
    elif connectivity == "hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 2 == 0]
    elif connectivity == "heavy-hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 4 == 0]
    else:
        raise ValueError(f"Invalid connectivity: {connectivity}")
    return pairs_aa, pairs_ab


@pytest.mark.parametrize(
    "norb, nelec, connectivity",
    [
        (4, (2, 2), "square"),
        (4, (1, 2), "square"),
        (4, (0, 2), "square"),
        (4, (0, 0), "square"),
        (4, (2, 2), "hex"),
        (4, (1, 2), "hex"),
        (4, (0, 2), "hex"),
        (4, (0, 0), "hex"),
        (4, (2, 2), "heavy-hex"),
        (4, (1, 2), "heavy-hex"),
        (4, (0, 2), "heavy-hex"),
        (4, (0, 0), "heavy-hex"),
    ],
)
def test_lucj_circuit_as_mps(norb: int, nelec: tuple[int, int], connectivity: str):
    """Test LUCJ circuit MPS construction."""
    rng = np.random.default_rng()

    # generate a random molecular Hamiltonian
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=rng)
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    # convert molecular Hamiltonian to MPO
    mol_hamiltonian_mpo = mol_hamiltonian.to_mpo()

    # generate a random LUCJ ansatz
    n_params = ffsim.UCJOpSpinBalanced.n_params(
        norb=norb,
        n_reps=1,
        interaction_pairs=_interaction_pairs_spin_balanced_(
            connectivity=connectivity, norb=norb
        ),
        with_final_orbital_rotation=True,
    )
    params = rng.uniform(-10, 10, size=n_params)
    lucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
        params,
        norb=norb,
        n_reps=1,
        interaction_pairs=_interaction_pairs_spin_balanced_(
            connectivity=connectivity, norb=norb
        ),
        with_final_orbital_rotation=True,
    )

    # generate the corresponding LUCJ circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(lucj_op), qubits)
    lucj_state = ffsim.qiskit.final_state_vector(circuit).vec

    # convert LUCJ ansatz to MPS
    options = {"trunc_params": {"chi_max": 16, "svd_min": 1e-6}}
    wavefunction_mps, _ = lucj_circuit_as_mps(norb, nelec, lucj_op, options)

    # test expectation is preserved
    original_expectation = np.real(np.vdot(lucj_state, hamiltonian @ lucj_state))
    mpo_expectation = mol_hamiltonian_mpo.expectation_value_finite(wavefunction_mps)
    np.testing.assert_allclose(original_expectation, mpo_expectation)
