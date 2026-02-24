from __future__ import annotations

import copy
import warnings
from typing import Any, Literal, Sequence

import rustworkx
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.providers import BackendV2
from qiskit.transpiler import (
    StagedPassManager,
    generate_preset_pass_manager,
)
from qiskit.transpiler.passes import ApplyLayout, VF2PostLayout
from qiskit_ibm_runtime.models.exceptions import BackendPropertyError
from rustworkx import NoEdgeBetweenNodes, PyGraph

import ffsim

IBM_TWO_Q_GATES = {"cx", "ecr", "cz", "rzz"}
VF2_CALL_LIMIT = 30_000_000


def _create_two_linear_chains(num_orbitals: int) -> PyGraph:
    """Creates two disconnected linear chains.

    Each chain has ``num_orbitals`` number of nodes (qubits). One represents,
    the alpha chain, and the other represents the beta chain. Later alpha-beta
    qubit connections will be added between these disconneted chains to create
    a complete LUCJ layout.

    Args:
        num_orbitals: Number orbitals or nodes in each linear chain. They are
            also known as alpha-alpha (beta-beta) interaction qubits.

    Returns:
        A rustworkx.PyGraph with two disconnected linear chains each with
            ``num_orbitals`` number of nodes.
    """
    graph = rustworkx.PyGraph()

    for n in range(num_orbitals):
        graph.add_node(n)

    for n in range(num_orbitals - 1):
        graph.add_edge(n, n + 1, None)

    for n in range(num_orbitals, 2 * num_orbitals):
        graph.add_node(n)

    for n in range(num_orbitals, 2 * num_orbitals - 1):
        graph.add_edge(n, n + 1, None)

    return graph


def _get_layout_graph_and_allowed_alpha_beta_indices(
    num_orbitals: int,
    backend_coupling_graph: PyGraph,
    backend_topology: Literal["heavy-hex", "grid"],
    alpha_beta_indices: list[tuple[int, int]],
) -> tuple[PyGraph, list[tuple[int, int]]]:
    """Creates a complete LUCJ layout graph with alpha-beta connections.

    The complete layout _can be mapped_ to an IBM QPU with heavy-hex or grid topology.
    The logic follows slightly different strategies for heavy-hex and grid.

        1. For both topologies, it starts with two disconnected linear chains, one for
            alpha qubits and one for beta qubits.
        2. For heavy-hex, the code adds ancillae qubits (nodes). The number of
            ancillae nodes is equal to the lenght of ```alpha_beta_indices`.
            After adding ancillae nodes, edges are established between corresponding
            alpha, ancilla, and beta nodes to create a candidate layout graph.
        3. For grid, no ancilla qubit is added. Edges are established directly between
            specified (in ``alpha_beta_indices``) alpha and beta nodes.
        4. Then, an isomorphism check is performed, whether the candidate layout graph
            (``graph_new``) is an isomorphic subgraph to the backend coupling graph.
                - If the candidate is not an isomorphic subgraph, one alpha-beta
                    interaction is removed _from the end_ of the ``alpha_beta_indices``
                    list, and steps 2 (or 3) and 4 are repeated.
                - If the candidate is an isomorphic subgraph, the routine ends and
                    returns the ``graph_new`` and ``alpha_beta_indices``.

    Args:
        num_orbitals: Number of orbitals, i.e., number of nodes in each alpha-alpha
            (beta-beta) linear chain.
        backend_coupling_graph: The coupling graph of the backend on which the LUCJ
            ansatz will be mapped and run. This function takes the coupling graph as
            a undirected `rustworkx.PyGraph` where there is only one 'undirected' edge
            between two nodes, i.e., qubits. Usually, the coupling graph of an IBM
            backend is directed (e.g., Eagle devices such as `ibm_brisbane`) or may
            have two edges between same two nodes (e.g., Heron `ibm_torino`). This
            function is only compatible with undirected graphs where there is only
            a single undirected edge between same two nodes.

    Returns:
        - A layout graph with alpha-alpha (beta-beta) and alpha-beta connections
            that is an isomorphic subgraph to an IBM backend.
        - Number of connecting qubits (alpha-beta interactions) between the linear
            chains. While we want as many connecting (alpha-beta) qubits
            between the linear (alpha-alpha/beta-beta) chains, we cannot accomodate
            all due to connectivity constraints of backends.
    """
    isomorphic = False
    graph = _create_two_linear_chains(num_orbitals=num_orbitals)

    graph_new = copy.deepcopy(graph)  # to avoid not bound warning
    while (not isomorphic) and alpha_beta_indices:
        graph_new = copy.deepcopy(graph)

        # add new nodes and edges
        if backend_topology == "heavy-hex":
            for i, (a, b) in enumerate(sorted(alpha_beta_indices, key=lambda x: x[0])):
                new_node = 2 * num_orbitals + i
                graph_new.add_node(new_node)
                graph_new.add_edge(a, new_node, None)
                graph_new.add_edge(new_node, b + num_orbitals, None)
        elif backend_topology == "grid":
            for i, (a, b) in enumerate(sorted(alpha_beta_indices, key=lambda x: x[0])):
                graph_new.add_edge(a, b + num_orbitals, None)
        else:
            ValueError(f"backend_topology={backend_topology} not supported.")

        isomorphic = rustworkx.is_subgraph_isomorphic(
            backend_coupling_graph,
            graph_new,
            call_limit=VF2_CALL_LIMIT,
            id_order=False,
            induced=False,
        )

        if not isomorphic:
            warnings.warn(
                f"Backend cannot accomodate alpha_beta_incides {alpha_beta_indices}.\n "
                f"Removing interaction {alpha_beta_indices[-1]} from the end."
            )
            del alpha_beta_indices[-1]

    return graph_new, alpha_beta_indices


def _make_backend_cmap_pygraph(
    backend: BackendV2,
    thresh_two_q: float,
    thresh_meas: float,
) -> PyGraph:
    """Converts a backend coupling map to a rustworkx.PyGraph.

    The PyGraph has only a single edge between any two nodes.

    Args:
        backend: An IBM backend.

    Returns:
        A rustworkx.PyGraph with a single undirected edge between same two nodes.
    """
    props = backend.properties()
    two_q_gate_name = IBM_TWO_Q_GATES.intersection(
        backend.configuration().basis_gates
    ).pop()
    graph = copy.deepcopy(backend.coupling_map.graph)

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

    # remove bad nodes
    node_indices = backend_coupling_graph.node_indices()
    for node_id in node_indices:
        re = props.readout_error(node_id)
        if re >= thresh_meas:
            backend_coupling_graph.remove_node(node_id)

    # remove bad edges
    edge_list = backend_coupling_graph.edge_list()
    for edge in edge_list:
        try:
            ge = props.gate_error(two_q_gate_name, edge)
        except BackendPropertyError:
            ge = props.gate_error(two_q_gate_name, edge[::-1])

        if ge >= thresh_two_q:
            backend_coupling_graph.remove_edge(edge[0], edge[1])

    edge_list = backend_coupling_graph.edge_list()

    return backend_coupling_graph


def _get_placeholder_layout_and_allowed_interactions(
    backend: BackendV2,
    num_orbitals: int,
    backend_topology: Literal["heavy-hex", "grid"],
    requested_alpha_beta_indices: Sequence[tuple[int, int]],
    thresh_two_q: float,
    thresh_meas: float,
) -> tuple[list[int], list[tuple[int, int]]]:
    """The main function that generates the zigzag pattern with physical qubits
    that can be used as an `intial_layout` in a preset passmanager/transpiler.

    Args:
        num_orbitals: Number of orbitals.
        backend: A backend.
        requested_alpha_beta_indices: A list of requested alpha-beta interactions.
            Due to HW limitations, the full requested list of interactions may not be
            accomodated. In that case, interaction pair from the end of the list is
            removed one-by-one. Thus, if user specified, order the list in a descending
            priority.

    Returns:
        A tuple of device compliant layout (`list[int]`) with zigzag pattern and an
        `int` representing number of alpha-beta-interactions.
    """
    backend_coupling_graph = _make_backend_cmap_pygraph(
        backend=backend, thresh_two_q=thresh_two_q, thresh_meas=thresh_meas
    )

    (layout_graph, allowed_alpha_beta_indices) = (
        _get_layout_graph_and_allowed_alpha_beta_indices(
            num_orbitals=num_orbitals,
            backend_coupling_graph=backend_coupling_graph,
            backend_topology=backend_topology,
            alpha_beta_indices=list(requested_alpha_beta_indices),
        )
    )
    num_allowed_alpha_beta_indices = len(allowed_alpha_beta_indices)
    if num_allowed_alpha_beta_indices == 0:
        raise RuntimeError("No alpha-beta interaction can be accomodated. Terminating.")

    isomorphic_mappings = rustworkx.vf2_mapping(
        backend_coupling_graph,
        layout_graph,
        subgraph=True,
        id_order=False,
        induced=False,
        call_limit=VF2_CALL_LIMIT,
    )

    mapping = next(isomorphic_mappings)
    initial_layout = [-1] * (2 * num_orbitals + num_allowed_alpha_beta_indices)

    for key, value in mapping.items():
        initial_layout[value] = key

    # as grids do not have ancillae qubits, the trailing items
    # in the ``initial_layout`` will remain as ``-1``. Therefore,
    # we are ignoring last several qubits in this check.
    if -1 in initial_layout[:-num_allowed_alpha_beta_indices]:
        raise ValueError(
            f"Not all qubits in the placeholder `initial_layout` is properly set."
            f"Negative qubit index in `initial_layout`. "
            f"intial_layout={initial_layout[:-num_allowed_alpha_beta_indices]}"
        )

    return initial_layout[:-num_allowed_alpha_beta_indices], allowed_alpha_beta_indices


def generate_lucj_pass_manager(
    backend: BackendV2,
    num_orbitals: int,
    backend_topology: Literal["heavy-hex", "grid"],
    requested_alpha_beta_indices: Sequence[tuple[int, int]] | None = None,
    thresh_two_q: float = 1.0,
    thresh_meas: float = 0.10,
    **qiskit_pm_kwargs,
) -> tuple[StagedPassManager, list[tuple[int, int]]]:
    """Generates a Qiskit preset pass manager that adheres to local
    unitary-coupled Jastrow (LUCJ) anstaz's _zigzag_ layout on heavy-hex
    backend topologies (Mario Motta et al.,
    https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02516k).
    In addition to the pass manager, this function also returns a list of
    hardware compatible alpha-beta interactions.

    Args:
        backend: A Qiskit BackendV2.
        num_orbitals: The number of _spatial_ orbitals of the molecule to be
            mapped using the LUCJ ansatz. The number of qubits in the LUCJ
            ansatz will be 2 * num_orbitals + ancilae qubits.
        requested_alpha_beta_indices: A user may optionally request a list of
            alpha-beta interactions. The code will try to find a layout that satisfies
            the user requested alpha-beta pairs. However, due to limited hardware
            connectivity, the request may not be entirely entertained. In that case,
            the code removes pairs from the end of the requested list one-by-one from
            the end of the list until a layout is found. Therefore, a user should list
            the pairs in desceding order of priority. If `None`, the code uses
            sequential alpha-beta interactions, i.e., [(0, 0), (4, 4), ... up to
            allowed by backend connectivity].
            Default: `None`.
        thresh_two_q (float): Removes edges from the backend coupling graph that has
            ``2Q gate error >= thresh_two_q``. Default: `1.0` (faulty edge).
        thresh_meas (float): Removes nodes from the backend coupling graph that has
            ``measurement error >= thresh_meas``. Default: `0.10`.
        **qiskit_pm_kwargs: The function accepts full list of arguments from
            `qiskit.transpiler.generate_preset_pass_manager <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.generate_preset_pass_manager>`_
            except `initial_layout` and `layout_method` as they are conflicting with this
            routine's functionality. If specified, they will be deleted with a warning.

    Returns:
        - A preset pass manager.
        - A list of alpha-beta pairs that can be accomodated on the backend.
    """  # noqa: E501
    if "initial_layout" in qiskit_pm_kwargs:
        warnings.warn("Argument `initial_layout` is ignored.")
        del qiskit_pm_kwargs["initial_layout"]

    if "layout_method" in qiskit_pm_kwargs:
        warnings.warn("Argument `layout_method` is ignored.")
        del qiskit_pm_kwargs["layout_method"]

    if requested_alpha_beta_indices:
        for alpha, beta in requested_alpha_beta_indices:
            if alpha >= num_orbitals or beta >= num_orbitals:
                raise ValueError(
                    f"Requested alpha-beta interaction {(alpha, beta)} is out of "
                    f"range for maximum spatial orbital index of {num_orbitals - 1}."
                )

    if backend_topology.lower() not in ["heavy-hex", "grid"]:
        raise ValueError(
            f"backend_topology={backend_topology} is not supported. "
            f"Only supported topologies are either 'heavy-hex' or 'grid'"
        )

    if requested_alpha_beta_indices is None:
        if backend_topology.lower() == "heavy-hex":
            requested_alpha_beta_indices = [
                (p, p) for p in range(num_orbitals) if p % 4 == 0
            ]
        elif backend_topology.lower() == "grid":
            requested_alpha_beta_indices = [(p, p) for p in range(num_orbitals)]

    (placeholder_initial_layout, allowed_alpha_beta_indices) = (
        _get_placeholder_layout_and_allowed_interactions(
            backend=backend,
            num_orbitals=num_orbitals,
            backend_topology=backend_topology,
            requested_alpha_beta_indices=requested_alpha_beta_indices,
            thresh_two_q=thresh_two_q,
            thresh_meas=thresh_meas,
        )
    )

    pm = generate_preset_pass_manager(
        backend=backend, initial_layout=placeholder_initial_layout, **qiskit_pm_kwargs
    )
    pm.pre_init = ffsim.qiskit.PRE_INIT

    # generating a preset pass manager with `initial_layout`
    # (`=placeholder_initial_layout`) disables the `VF2PostLayout` pass.
    # Therefore, we manually turn on the pass here so that it can search
    # (better) isomorphic subgraph layouts to the initial layout and apply it
    # to the circuit.
    def _custom_apply_post_layout_condition(property_set: dict[str, Any]) -> bool:
        return property_set["post_layout"] is not None

    pm.routing.append(
        VF2PostLayout(
            target=backend.target, strict_direction=False, call_limit=VF2_CALL_LIMIT
        )
    )
    pm.routing.append(
        ConditionalController(
            ApplyLayout(), condition=_custom_apply_post_layout_condition
        )
    )

    return pm, allowed_alpha_beta_indices
