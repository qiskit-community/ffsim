import warnings
from qiskit.providers import BackendV2
from typing import Literal, Sequence, Any
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager, StagedPassManager
from qiskit.transpiler.passes import VF2PostLayout, ApplyLayout
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.transpiler import PassManager

import copy
from typing import Sequence

from ffsim.qiskit import PRE_INIT

import rustworkx
from qiskit.providers import BackendV2
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino
from rustworkx import NoEdgeBetweenNodes, PyGraph


"""Funtionality

1. Throw warning if qiskit transpiler's intial_layout and layout_method are specified.
    These two directly conflicts with LUCJ's need for custom layout
2. 
"""

def _create_two_linear_chains(num_orbitals: int) -> PyGraph:
    """In zig-zag layout, there are two linear chains (with connecting qubits between
    the chains). This function creates those two linear chains: a rustworkx PyGraph
    with two disconnected linear chains. Each chain contains `num_orbitals` number
    of nodes, i.e., in the final graph there are `2 * num_orbitals` number of nodes.

    Args:
        num_orbitals (int): Number orbitals or nodes in each linear chain. They are
            also known as alpha-alpha interaction qubits.

    Returns:
        A rustworkx.PyGraph with two disconnected linear chains each with `num_orbitals`
            number of nodes.
    """
    G = rustworkx.PyGraph()

    for n in range(num_orbitals):
        G.add_node(n)

    for n in range(num_orbitals - 1):
        G.add_edge(n, n + 1, None)

    for n in range(num_orbitals, 2 * num_orbitals):
        G.add_node(n)

    for n in range(num_orbitals, 2 * num_orbitals - 1):
        G.add_edge(n, n + 1, None)

    return G

def get_layout_graph_and_allowed_alpha_beta_indices(
    num_orbitals: int,
    backend_coupling_graph: PyGraph,
    topology: str,
    alpha_beta_indices: list[tuple[int, int]],
) -> tuple[PyGraph, list[tuple[int, int]]]:
    """This function creates the complete zigzag graph that 'can be mapped' to a IBM QPU with
    heavy-hex connectivity (the zigzag must be an isomorphic sub-graph to the QPU/backend
    coupling graph for it to be mapped).
    The zigzag pattern includes both linear chains (alpha-alpha interactions) and connecting
    qubits between the linear chains (alpha-beta interactions).

    Args:
        num_orbitals (int): Number of orbitals, i.e., number of nodes in each alpha-alpha linear chain.
        backend_coupling_graph (PyGraph): The coupling graph of the backend on which the LUCJ ansatz
            will be mapped and run. This function takes the coupling graph as a undirected
            `rustworkx.PyGraph` where there is only one 'undirected' edge between two nodes,
            i.e., qubits. Usually, the coupling graph of a IBM backend is directed (e.g., Eagle devices
            such as ibm_sherbrooke) or may have two edges between two nodes (e.g., Heron `ibm_torino`).
            A user needs to be make such graphs undirected and/or remove duplicate edges to make them
            compatible with this function. One way to do this is as follows:
            ```
            graph = backend.coupling_map.graph
            if not graph.is_symmetric():
                graph.make_symmetric()
            backend_coupling_graph = graph.to_undirected()

            edge_list = backend_coupling_graph.edge_list()
            removed_edge = []
            for edge in edge_list:
                if set(edge) in removed_edge:
                    continue
                try:
                    backend_coupling_graph.remove_edge(edge[0], edge[1])
                    removed_edge.append(set(edge))
                except NoEdgeBetweenNodes:
                    pass
            ```

    Returns:
        G_new (PyGraph): The graph with IBM backend compliant zigzag pattern.
        num_alpha_beta_qubits (int): Number of connecting qubits between the linear chains
            in the zigzag pattern. While we want as many connecting (alpha-beta) qubits between
            the linear (alpha-alpha) chains, we cannot accomodate all due to qubit and connectivity
            constraints of backends. This is the maximum number of connecting qubits the zigzag pattern
            can have while being backend compliant (i.e., isomorphic to backend coupling graph).
    """
    isomorphic = False
    G = _create_two_linear_chains(num_orbitals=num_orbitals)
    
    G_new = copy.deepcopy(G) # to avoid not bound warning
    while not isomorphic:
        print("Inside while loop")
        G_new = copy.deepcopy(G)
        
        if not alpha_beta_indices:
            break

        # add new nodes and edges
        for i, (a, b) in enumerate(sorted(alpha_beta_indices, key=lambda x: x[0])):
            # print(f'i={i} | (alpha, beta)={(a, b)}')
            if topology == "heavy-hex":
                new_node = 2 * num_orbitals + i
                G_new.add_node(new_node)
                G_new.add_edge(a, new_node, None)
                G_new.add_edge(new_node, b + num_orbitals, None)
            elif topology == "grid":
                G_new.add_edge(a, b + num_orbitals, None)
                # pass
            else:
                raise ValueError(f"topology={topology} not allowed.")
            # num_alpha_beta_qubits = num_alpha_beta_qubits + 1
        isomorphic = rustworkx.is_subgraph_isomorphic(backend_coupling_graph, G_new, call_limit=1_000_000)

        if not isomorphic:
            print(
                f"Backend cannot accomodate alpha_beta_incides {alpha_beta_indices}.\n "
                f"Removing interaction {alpha_beta_indices[-1]} from the end."
            )
            del alpha_beta_indices[-1]

    return G_new, alpha_beta_indices

def _make_backend_cmap_pygraph(backend: BackendV2) -> PyGraph:
    graph = backend.coupling_map.graph
    if not graph.is_symmetric():
        graph.make_symmetric()
    backend_coupling_graph = graph.to_undirected()

    edge_list = backend_coupling_graph.edge_list()
    removed_edge = []
    for edge in edge_list:
        if set(edge) in removed_edge:
            continue
        try:
            backend_coupling_graph.remove_edge(edge[0], edge[1])
            removed_edge.append(set(edge))
        except NoEdgeBetweenNodes:
            pass

    return backend_coupling_graph


def get_placeholder_initial_layout_and_allowed_alpha_beta_indices(
    backend: BackendV2,
    num_orbitals: int,
    topology: str,
    requested_alpha_beta_indices: Sequence[tuple[int, int]],
) -> tuple[list[int], list[tuple[int, int]]]:
    """The main function that generates the zigzag pattern with physical qubits that can be used
    as an `intial_layout` in a preset passmanager/transpiler.

    Args:
        num_orbitals (int): Number of orbitals.
        backend (BackendV2): A backend.
        expected_alpha_beta_indices (list): User-defined arbitrary alpha-beta interactions.
            Due to HW limitations, the full `expected` list of interactions may not be
            accomodated. In that case, interaction pair from the end of the list is removed
            one-by-one. Thus, a user must order the list in a descending order of priority.
        score_layouts (bool): Optional. If `True`, it uses the `lightweight_layout_error_scoring`
            function to score the isomorphic layouts and returns the layout with less errorneous qubits.
            If `False`, returns the first isomorphic subgraph.

    Returns:
        A tuple of device compliant layout (list[int]) with zigzag pattern and an int representing
            number of alpha-beta-interactions.
    """
    backend_coupling_graph = _make_backend_cmap_pygraph(backend=backend)

    G, allowed_alpha_beta_indices = get_layout_graph_and_allowed_alpha_beta_indices(
        num_orbitals=num_orbitals,
        backend_coupling_graph=backend_coupling_graph,
        topology=topology,
        alpha_beta_indices=list(requested_alpha_beta_indices),
    )
    num_allowed_alpha_beta_indices = len(allowed_alpha_beta_indices)
    isomorphic_mappings = rustworkx.vf2_mapping(
        backend_coupling_graph, G, subgraph=True
    )

    mapping = next(isomorphic_mappings)

    if topology == "heavy-hex":
        initial_layout= [-1] * (2 * num_orbitals + num_allowed_alpha_beta_indices)
    elif topology == "grid":
        initial_layout= [-1] * (2 * num_orbitals)
    else:
        raise ValueError("topology")
    
    for key, value in mapping.items():
        initial_layout[value] = key
    
    if -1 in initial_layout:
        raise ValueError(
            f"Negative qubit index in `initial_layout`. "
            f"intial_layout={initial_layout}"
        )
    
    if topology == "grid":
        return initial_layout, allowed_alpha_beta_indices

    return initial_layout[:-num_allowed_alpha_beta_indices], allowed_alpha_beta_indices


def get_pass_manager_and_allowed_alpha_beta_indices_for_lucj(
    backend: BackendV2,
    num_orbitals: int,
    topology: Literal["grid", "heavy-hex"],
    requested_alpha_beta_indices: Sequence[tuple[int, int]] | None,
    **qiskit_pm_kwargs
) -> tuple[StagedPassManager, list[tuple[int, int]]]:
    if "initial_layout" in qiskit_pm_kwargs:
        warnings.warn("Argument `initial_layout` is ignored.")
        del qiskit_pm_kwargs["initial_layout"]

    if "layout_method" in qiskit_pm_kwargs:
        warnings.warn("Argument `layout_method` is ignored.")
        del qiskit_pm_kwargs["layout_method"]
    
    if requested_alpha_beta_indices is None:
        if topology == "heavy-hex":
            requested_alpha_beta_indices = [
                (p, p) for p in range(num_orbitals) if p % 4 == 0
            ]
        elif topology == "grid":
            requested_alpha_beta_indices = [(p, p) for p in range(num_orbitals)]
        else:
            raise ValueError(
                f"topology={topology} not recognized. "
                f"Only 'heavy-hex' or 'grid' is allowed"
            )
    
    (
        placeholder_initial_layout,
        allowed_alpha_beta_indices
    ) = get_placeholder_initial_layout_and_allowed_alpha_beta_indices(
        backend=backend,
        num_orbitals=num_orbitals,
        topology=topology,
        requested_alpha_beta_indices=requested_alpha_beta_indices,
    )

    pm = generate_preset_pass_manager(
        backend=backend,
        initial_layout=placeholder_initial_layout,
        **qiskit_pm_kwargs
    )
    pm.pre_init = PRE_INIT

    def _custom_apply_post_layout_condition(property_set: dict[str, Any]) -> bool:
        return property_set["post_layout"] is not None
    
    pm.routing.append(VF2PostLayout(target=backend.target, strict_direction=False))
    pm.routing.append(
        ConditionalController(
            ApplyLayout(),
            condition=_custom_apply_post_layout_condition
        )
    )

    return pm, allowed_alpha_beta_indices

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap

if __name__ == "__main__":
    import ffsim
    import numpy as np
    import warnings

    warnings.filterwarnings("ignore")

    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService

    # common args
    num_orbitals = 36
    requested_alpha_beta_indices = [
        (32, 32), (4, 4), (8, 8), (24, 24), (16, 16), (28, 28)
    ]
    n_reps = 2
    alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]

    # heavy hex
    print("\nHeavy Hex ...")
    backend = FakeSherbrooke()
    pm, alpha_beta_indices = get_pass_manager_and_allowed_alpha_beta_indices_for_lucj(
        backend=backend,
        num_orbitals=num_orbitals,
        topology="heavy-hex",
        requested_alpha_beta_indices=requested_alpha_beta_indices,
        optimization_level=3
    )
    
    num_alpha_beta_indices = len(alpha_beta_indices)
    print(f'Final alpha-beta-interactions {alpha_beta_indices}')
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb=num_orbitals,
        n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
        seed=0
    )

    qubits = QuantumRegister(2 * num_orbitals, name="q")
    circuit = QuantumCircuit(qubits)
    nelec = (5, 5)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), qubits)

    # apply the UCJ operator to the reference state
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    circuit.measure_all()

    isa_circuit_vf2 = pm.run(circuit)
    num_2q_1 = isa_circuit_vf2.count_ops()["ecr"]
    print(f"Num 2Q gates: {num_2q_1}")
    print("Initial layout using the automated method: ")
    print(isa_circuit_vf2.layout.initial_index_layout()[:2 * num_orbitals])

    
    # grid
    print("\nGrid ...")
    requested_alpha_beta_indices = (
        [(p, p) for p in range(11)] + [(p, p) for p in range(25, 36)]
    )
    grid_cmap = CouplingMap.from_grid(num_rows=12, num_columns=10, bidirectional=True)
    backend_grid = GenericBackendV2(
        num_qubits=grid_cmap.size(),
        basis_gates=["id", "rz", "sx", "x", "cz"],
        coupling_map=grid_cmap,
    )

    pm, alpha_beta_indices = get_pass_manager_and_allowed_alpha_beta_indices_for_lucj(
        backend=backend_grid,
        num_orbitals=num_orbitals,
        topology="grid",
        requested_alpha_beta_indices=requested_alpha_beta_indices,
        optimization_level=3
    )
    
    num_alpha_beta_indices = len(alpha_beta_indices)
    print(f'Final alpha-beta-interactions {alpha_beta_indices}')
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb=num_orbitals,
        n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
        seed=0
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
    print(isa_circuit_vf2.layout.initial_index_layout()[:2 * num_orbitals])