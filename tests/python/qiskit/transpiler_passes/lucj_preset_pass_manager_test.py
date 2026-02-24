import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh, FakeNighthawk, FakeTorino

import ffsim
from ffsim.qiskit import generate_lucj_pass_manager

backend_no_ab = FakeTorino()
backend_heavy_hex = FakeMarrakesh()
backend_grid = FakeNighthawk()
num_orbitals = 36


@pytest.mark.parametrize(
    "topology_and_backend", [("heavy-hex", backend_heavy_hex), ("grid", backend_grid)]
)
def test_raise_warning1(topology_and_backend):
    """Tests UserWarning raised when `initial_layout` is specified and ignored."""
    topology, backend = topology_and_backend
    with pytest.warns(UserWarning, match="Argument `initial_layout` is ignored."):
        _, _ = generate_lucj_pass_manager(
            backend=backend,
            num_orbitals=num_orbitals,
            backend_topology=topology,
            requested_alpha_beta_indices=None,
            optimization_level=3,
            initial_layout=[1],
        )


@pytest.mark.parametrize(
    "topology_and_backend", [("heavy-hex", backend_heavy_hex), ("grid", backend_grid)]
)
def test_raise_warning2(topology_and_backend):
    """Tests UserWarning raised when `layout_method` is specified and ignored."""
    topology, backend = topology_and_backend
    with pytest.warns(UserWarning, match="Argument `layout_method` is ignored."):
        _, _ = generate_lucj_pass_manager(
            backend=backend,
            num_orbitals=num_orbitals,
            backend_topology=topology,
            requested_alpha_beta_indices=None,
            optimization_level=3,
            layout_method="placeholder",
        )


@pytest.mark.parametrize(
    "topology_and_backend", [("heavy-hex", backend_heavy_hex), ("grid", backend_grid)]
)
@pytest.mark.parametrize(
    "requested_alpha_beta_indices",
    [[(num_orbitals + 1, num_orbitals + 1)], [(num_orbitals, num_orbitals)]],
)
def test_raise_value_error1(requested_alpha_beta_indices, topology_and_backend):
    """Tests ValueError raised when requested alpha-beta > num orbitals."""
    topology, backend = topology_and_backend
    with pytest.raises(ValueError):
        _, _ = generate_lucj_pass_manager(
            backend=backend,
            num_orbitals=num_orbitals,
            backend_topology=topology,
            requested_alpha_beta_indices=requested_alpha_beta_indices,
            optimization_level=3,
        )


def test_raise_value_error2():
    """Tests ValueError raised when topology is neither 'heavy-hex' nor 'grid'."""
    with pytest.raises(ValueError):
        _, _ = generate_lucj_pass_manager(
            backend=backend_heavy_hex,
            num_orbitals=num_orbitals,
            backend_topology="line",
            requested_alpha_beta_indices=None,
            optimization_level=3,
        )


def test_raise_runtime_error():
    """Tests RuntimeError raised when there are zero alpha-beta interactions."""
    with pytest.raises(RuntimeError):
        _, _ = generate_lucj_pass_manager(
            backend=backend_no_ab,
            num_orbitals=num_orbitals,
            backend_topology="heavy-hex",
            requested_alpha_beta_indices=None,
            optimization_level=3,
        )


@pytest.mark.parametrize(
    "topology_and_backend", [("heavy-hex", backend_heavy_hex), ("grid", backend_grid)]
)
@pytest.mark.parametrize(
    "requested_alpha_beta_indices",
    [None, [(32, 32), (4, 4), (8, 8), (24, 24), (16, 16), (28, 28)]],
)
def test_generate_lucj_pass_manager(requested_alpha_beta_indices, topology_and_backend):
    """Tests whether the transpiled LUCJ ansatz retains correct a-b interactions.

    On heavy-hex, it should alpha - ancilla - beta, and on grid, it must
    be alpha - beta, i.e.,
    1. A qubit on the alpha (beta) chain must be 1 distance (edge) apart
        from the next qubit on the chain, on both heavy-hex and grid.
    2. Distance between an alpha and a beta qubit on a heavy-hex must be 2,
        and on a grid must be 1.
    """
    topology, backend = topology_and_backend
    n_reps = 2
    alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
    pm, alpha_beta_indices = generate_lucj_pass_manager(
        backend=backend,
        num_orbitals=num_orbitals,
        backend_topology=topology,
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

    for idx, _ in alpha_beta_indices:
        dist = coupling_map.distance(alpha_qubits[idx], beta_qubits[idx])
        if topology == "heavy-hex":
            # alpha and beta qubits
            # connected by an ancilla
            # must be 2 edges apart
            # on heavy-hex
            # alpha - ancilla - beta
            assert dist == 2
        elif topology == "grid":
            # alpha and beta qubits
            # must be 2 edges apart
            # on grid
            # alpha - beta
            assert dist == 1

    # Adjacent qubits on alpha/beta chains are 1 edge apart
    for idx1 in range(num_orbitals - 1):
        idx2 = idx1 + 1

        dist = coupling_map.distance(alpha_qubits[idx1], alpha_qubits[idx2])
        assert dist == 1

        dist = coupling_map.distance(beta_qubits[idx1], beta_qubits[idx2])
        assert dist == 1
