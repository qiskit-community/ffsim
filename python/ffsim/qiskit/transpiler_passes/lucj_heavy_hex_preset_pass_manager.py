import copy
import warnings
from typing import Any, Sequence

import rustworkx
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.providers import BackendV2
from qiskit.transpiler import (
    StagedPassManager,
    generate_preset_pass_manager,
)
from qiskit.transpiler.passes import ApplyLayout, VF2PostLayout
from rustworkx import NoEdgeBetweenNodes, PyGraph

import ffsim


def _create_two_linear_chains(num_orbitals: int) -> PyGraph:
    """In zig-zag layout, there are two linear chains (with connecting qubits between
    the chains). This function creates those two linear chains which is a rustworkx
    PyGraph with two disconnected linear chains. Each chain contains `num_orbitals`
    number of nodes, i.e., in the final graph there are `2 * num_orbitals` number of
    nodes.

    Args:
        num_orbitals: Number orbitals or nodes in each linear chain. They are
            also known as alpha-alpha interaction qubits.

    Returns:
        A rustworkx.PyGraph with two disconnected linear chains each with `num_orbitals`
            number of nodes.
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
    alpha_beta_indices: list[tuple[int, int]],
) -> tuple[PyGraph, list[tuple[int, int]]]:
    """This function creates the complete zigzag graph that _can be mapped_ to a IBM
        QPU with heavy-hex connectivity (i.e., the zigzag pattern is an isomorphic
        sub-graph to the QPU/backend coupling graph). The zigzag pattern includes
        both linear chains (alpha-alpha/beta-beta interactions) and connecting qubits
        between the linear chains (alpha-beta interactions).

        The algorithm works as follows: It starts with an interm graph (`graph_new`)
        that has two linear chains with connecting nodes between two nodes (qubits)
        specified by `alpha_beta_indices` list. The algorithm checks if the starting
        graph is an isomorphic subgraph to the larger `backend_cpupling_graph`. If yes,
        the routine ends and returns the `graph_new`. If not, it removes an alpha-beta
        interaction pair from the end of list `alpha_beta_indices` and checks for
        subgraph isomorphism again. It cycle continues, until a isomorhic subgraph is
        found.

    Args:
        num_orbitals: Number of orbitals, i.e., number of nodes in each alpha-alpha
            linear chain.
        backend_coupling_graph: The coupling graph of the backend on which the LUCJ
            ansatz will be mapped and run. This function takes the coupling graph as
            a undirected `rustworkx.PyGraph` where there is only one 'undirected' edge
            between two nodes, i.e., qubits. Usually, the coupling graph of a IBM
            backend is directed (e.g., Eagle devices such as `ibm_brisbane`) or may
            have two edges between same two nodes (e.g., Heron `ibm_torino`). This
            function is only compatible with undirected graphs where there is only
            a single undirected edge between same two nodes.

    Returns:
        graph_new: A graph that has the _zigzag_ pattern and is an isomorphic subgraph
            to an a heavy-hex IBM backend.
        num_alpha_beta_qubits: Number of connecting qubits between the linear chains
            in the zigzag pattern. While we want as many connecting (alpha-beta) qubits
            between the linear (alpha-alpha) chains, we cannot accomodate all due to
            connectivity constraints of backends. This is the maximum number of
            connecting qubits the zigzag pattern can have while being backend compliant
            (i.e., isomorphic subgraph to the backend coupling graph).
    """
    isomorphic = False
    graph = _create_two_linear_chains(num_orbitals=num_orbitals)

    graph_new = copy.deepcopy(graph)  # to avoid not bound warning
    while not isomorphic:
        graph_new = copy.deepcopy(graph)

        if not alpha_beta_indices:
            break

        # add new nodes and edges
        for i, (a, b) in enumerate(sorted(alpha_beta_indices, key=lambda x: x[0])):
            new_node = 2 * num_orbitals + i
            graph_new.add_node(new_node)
            graph_new.add_edge(a, new_node, None)
            graph_new.add_edge(new_node, b + num_orbitals, None)

        isomorphic = rustworkx.is_subgraph_isomorphic(
            backend_coupling_graph,
            graph_new,
            call_limit=1_000_000,
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


def _make_backend_cmap_pygraph(backend: BackendV2) -> PyGraph:
    """Converts an IBM backend coupling map to an undirected rustworkx.PyGraph where
    there is only a single edge between same two nodes.

    Args:
        backend: An IBM backend.

    Returns:
        A rustworkx.PyGraph with a single undirected edge between same two nodes.
    """
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


def _get_placeholder_initial_layout_and_allowed_alpha_beta_indices(
    backend: BackendV2,
    num_orbitals: int,
    requested_alpha_beta_indices: Sequence[tuple[int, int]],
) -> tuple[list[int], list[tuple[int, int]]]:
    """The main function that generates the zigzag pattern with physical qubits that
        can be used as an `intial_layout` in a preset passmanager/transpiler.

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
    backend_coupling_graph = _make_backend_cmap_pygraph(backend=backend)

    (graph, allowed_alpha_beta_indices) = (
        _get_layout_graph_and_allowed_alpha_beta_indices(
            num_orbitals=num_orbitals,
            backend_coupling_graph=backend_coupling_graph,
            alpha_beta_indices=list(requested_alpha_beta_indices),
        )
    )
    num_allowed_alpha_beta_indices = len(allowed_alpha_beta_indices)
    isomorphic_mappings = rustworkx.vf2_mapping(
        backend_coupling_graph, graph, subgraph=True, id_order=False, induced=False
    )

    mapping = next(isomorphic_mappings)
    initial_layout = [-1] * (2 * num_orbitals + num_allowed_alpha_beta_indices)

    for key, value in mapping.items():
        initial_layout[value] = key

    if -1 in initial_layout:
        raise ValueError(
            f"Not all qubits in the placeholder `initial_layout` is properly set."
            f"Negative qubit index in `initial_layout`. "
            f"intial_layout={initial_layout}"
        )

    return initial_layout[:-num_allowed_alpha_beta_indices], allowed_alpha_beta_indices


def generate_preset_pass_manager_lucj_heavy_hex_with_alpha_betas(
    backend: BackendV2,
    num_orbitals: int,
    requested_alpha_beta_indices: Sequence[tuple[int, int]] | None = None,
    **qiskit_pm_kwargs,
) -> tuple[StagedPassManager, list[tuple[int, int]]]:
    """Generates a Qiskit preset pass manager that adheres to local
    unitary-coupled Jastrow (LUCJ) anstaz's _zigzag_ layout on heavy-hex
    backend topologies (Mario Motta et al.,
    https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02516k).
    In addition to the pass manager, this function also returns a list of
    hardware compatible alpha-beta interactions.

    Args:
        backend: An IBM backend.
        num_orbitals: The number of _spatial_ orbitals of the molecule to be
            mapped using the LUCJ ansatz. The number of qubits in the LUCJ
            ansatz will be 2 * num_orbitals + ancilae qubits.
        requested_alpha_beta_indices: A user may optionally request a list of
            alpha-beta interactions. The code will try to find a layout that satisfies
            the user requested alpha-beta pairs. However, due to limited hardware
            connectivity, the request may not be entirely entertained. It that case,
            the code removes pairs from the end of the requested list one-by-one from
            the end of the list until a layout is found. Therefore, a user should list
            the pairs in desceding order of priority. If `None`, the code uses
            sequential alpha-beta interactions, i.e., [(0, 0), (4, 4), ... up to
            allowed by backend connectivity].
            Default: `None`.
        **qiskit_pm_kwargs: The function accepts full list of arguments from
            [`qiskit.transpiler.generate_preset_pass_manager`](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.generate_preset_pass_manager)
            except `initial_layout` and `layout_method` as they are conflicting with this
            routine's functionality.
            If specified, they will be deleted with a warning.

    Returns:
        pm: A preset pass manager.
        allowed_alpha_beta_indices: A list of alpha-beta pairs that can be accomodated
            on the backend.
    """  # noqa: E501
    if "initial_layout" in qiskit_pm_kwargs:
        warnings.warn("Argument `initial_layout` is ignored.")
        del qiskit_pm_kwargs["initial_layout"]

    if "layout_method" in qiskit_pm_kwargs:
        warnings.warn("Argument `layout_method` is ignored.")
        del qiskit_pm_kwargs["layout_method"]

    if requested_alpha_beta_indices is None:
        requested_alpha_beta_indices = [
            (p, p) for p in range(num_orbitals) if p % 4 == 0
        ]

    (placeholder_initial_layout, allowed_alpha_beta_indices) = (
        _get_placeholder_initial_layout_and_allowed_alpha_beta_indices(
            backend=backend,
            num_orbitals=num_orbitals,
            requested_alpha_beta_indices=requested_alpha_beta_indices,
        )
    )

    pm = generate_preset_pass_manager(
        backend=backend, initial_layout=placeholder_initial_layout, **qiskit_pm_kwargs
    )
    pm.pre_init = ffsim.qiskit.PRE_INIT

    def _custom_apply_post_layout_condition(property_set: dict[str, Any]) -> bool:
        return property_set["post_layout"] is not None

    pm.routing.append(VF2PostLayout(target=backend.target, strict_direction=False))
    pm.routing.append(
        ConditionalController(
            ApplyLayout(), condition=_custom_apply_post_layout_condition
        )
    )

    return pm, allowed_alpha_beta_indices
