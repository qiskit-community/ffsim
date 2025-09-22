import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

import ffsim
from ffsim.qiskit import generate_pm_and_interactions_lucj_heavy_hex

backend = FakeMarrakesh()
num_orbitals = 36


def test_raise_warning1():
    with pytest.warns(UserWarning, match="Argument `initial_layout` is ignored."):
        _, _ = generate_pm_and_interactions_lucj_heavy_hex(
            backend=backend,
            num_orbitals=num_orbitals,
            requested_alpha_beta_indices=None,
            optimization_level=3,
            initial_layout=[1],
        )


def test_raise_warning2():
    with pytest.warns(UserWarning, match="Argument `layout_method` is ignored."):
        _, _ = generate_pm_and_interactions_lucj_heavy_hex(
            backend=backend,
            num_orbitals=num_orbitals,
            requested_alpha_beta_indices=None,
            optimization_level=3,
            layout_method="placeholder",
        )


test_data1 = [[(num_orbitals + 1, num_orbitals + 1)], [(num_orbitals, num_orbitals)]]


@pytest.mark.parametrize("requested_alpha_beta_indices", test_data1)
def test_raise_value_error(requested_alpha_beta_indices):
    with pytest.raises(ValueError):
        _, _ = generate_pm_and_interactions_lucj_heavy_hex(
            backend=backend,
            num_orbitals=num_orbitals,
            requested_alpha_beta_indices=requested_alpha_beta_indices,
            optimization_level=3,
        )


test_data2 = [None, [(32, 32), (4, 4), (8, 8), (24, 24), (16, 16), (28, 28)]]


@pytest.mark.parametrize("requested_alpha_beta_indices", test_data2)
def test_generate_pm_and_interactions_lucj_heavy_hex(requested_alpha_beta_indices):
    """Tests whether the LUCJ ansatz transpiled by the custom pass manager
    retains the expected zig-zag qubit pattern. To be a zig-zag pattern,
        1. A qubit on the alpha (beta) chain have to 1 distance (edge) apart
            from the next qubit on the chain.
        2. Distance between an alpha and a beta qubit connected by an ancilla
            must be 2 (2 edges apart).
    """
    n_reps = 2
    alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
    pm, alpha_beta_indices = generate_pm_and_interactions_lucj_heavy_hex(
        backend=backend,
        num_orbitals=num_orbitals,
        requested_alpha_beta_indices=requested_alpha_beta_indices,
        optimization_level=3,
    )

    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb=num_orbitals,
        n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
        seed=0,
    )

    nelec = (5, 5)

    qubits = QuantumRegister(2 * num_orbitals, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    circuit.measure_all()

    isa_circuit = pm.run(circuit)
    initial_layout = isa_circuit.layout.initial_index_layout(filter_ancillas=False)
    alpha_qubits = initial_layout[:num_orbitals]
    beta_qubits = initial_layout[num_orbitals : 2 * num_orbitals]

    coupling_map = backend.target.build_coupling_map()

    # alpha and beta qubits connected by an ancilla must be 2 edges apart
    # alpha - ancilla - beta
    for idx, _ in alpha_beta_indices:
        dist = coupling_map.distance(alpha_qubits[idx], beta_qubits[idx])
        assert dist == 2

    # Adjacent qubits on alpha/beta chains are 1 edge apart
    for idx1 in range(num_orbitals - 1):
        idx2 = idx1 + 1

        dist = coupling_map.distance(alpha_qubits[idx1], alpha_qubits[idx2])
        assert dist == 1

        dist = coupling_map.distance(beta_qubits[idx1], beta_qubits[idx2])
        assert dist == 1
