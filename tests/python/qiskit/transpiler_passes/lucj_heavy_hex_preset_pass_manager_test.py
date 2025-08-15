from qiskit_ibm_runtime.fake_provider import FakeMarrakesh, FakeSherbrooke

from ffsim.qiskit import generate_preset_pass_manager_lucj_heavy_hex_with_alpha_betas

if __name__ == "__main__":
    import warnings

    import ffsim

    warnings.filterwarnings("ignore")

    from qiskit import QuantumCircuit, QuantumRegister

    # common args
    num_orbitals = 36
    requested_alpha_beta_indices = [
        (32, 32),
        (4, 4),
        (8, 8),
        (24, 24),
        (16, 16),
        (28, 28),
    ]
    n_reps = 2
    alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]

    # heavy hex
    print("\nHeavy Hex ...")
    backend = FakeSherbrooke()
    backend = FakeMarrakesh()
    pm, alpha_beta_indices = (
        generate_preset_pass_manager_lucj_heavy_hex_with_alpha_betas(
            backend=backend,
            num_orbitals=num_orbitals,
            requested_alpha_beta_indices=requested_alpha_beta_indices,
            optimization_level=3,
        )
    )

    num_alpha_beta_indices = len(alpha_beta_indices)
    print(f"Final alpha-beta-interactions {alpha_beta_indices}")
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb=num_orbitals,
        n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
        seed=0,
    )

    qubits = QuantumRegister(2 * num_orbitals, name="q")
    circuit = QuantumCircuit(qubits)
    nelec = (5, 5)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)

    # apply the UCJ operator to the reference state
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    circuit.measure_all()

    isa_circuit_vf2 = pm.run(circuit)
    num_2q_1 = isa_circuit_vf2.count_ops()["cz"]
    print(f"Num 2Q gates: {num_2q_1}")
    print("Initial layout using the automated method: ")
    print(isa_circuit_vf2.layout.initial_index_layout()[: 2 * num_orbitals])

    # no requested alpha-beta indices
    print("No requested alpha-beta indices.\n")
    pm, alpha_beta_indices = (
        generate_preset_pass_manager_lucj_heavy_hex_with_alpha_betas(
            backend=backend, num_orbitals=num_orbitals, optimization_level=3
        )
    )

    num_alpha_beta_indices = len(alpha_beta_indices)
    print(f"Final alpha-beta-interactions {alpha_beta_indices}")
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb=num_orbitals,
        n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
        seed=0,
    )

    qubits = QuantumRegister(2 * num_orbitals, name="q")
    circuit = QuantumCircuit(qubits)
    nelec = (5, 5)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)

    # apply the UCJ operator to the reference state
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    circuit.measure_all()

    isa_circuit_vf2 = pm.run(circuit)
    num_2q_1 = isa_circuit_vf2.count_ops()["cz"]
    print(f"Num 2Q gates: {num_2q_1}")
    print("Initial layout using the automated method: ")
    print(isa_circuit_vf2.layout.initial_index_layout()[: 2 * num_orbitals])
