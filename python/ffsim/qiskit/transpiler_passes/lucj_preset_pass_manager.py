# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass manager for LUCJ ansatz with backend compliant layout."""

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
from rustworkx import NoEdgeBetweenNodes, PyGraph

import ffsim

IBM_TWO_Q_GATES = {"cx", "ecr", "cz"}
VF2_CALL_LIMIT = 30_000_000


def _create_two_linear_chains(num_orbitals: int) -> PyGraph:
    """Create two disconnected linear qubit chains for the LUCJ ansatz.

    Construct a ``rustworkx.PyGraph`` with two disconnected linear chains
    representing the alpha (nodes ``0`` to ``num_orbitals - 1``) and beta
    (nodes ``num_orbitals`` to ``2 * num_orbitals - 1``) qubits. This graph
    serves as the base structure for the LUCJ layout; alpha-beta connections
    between chains are added separately by
    ``_get_layout_graph_and_allowed_alpha_beta_indices``.

    Args:
        num_orbitals: Number of nodes (qubits) in each linear chain.

    Returns:
        A ``rustworkx.PyGraph`` containing two disconnected linear
        chains, each with ``num_orbitals`` nodes and ``num_orbitals - 1``
        edges.
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
    """Build a LUCJ layout graph and determine accommodated alpha-beta interactions.

    This routine constructs a candidate layout graph representing the LUCJ ansatz
    - two linear qubit chains (alpha and beta) connected by alpha-beta interactions
    - and verifies that it can be embedded as an isomorphic subgraph on the target
    backend. If the full set of requested alpha-beta pairs cannot be accommodated,
    pairs are dropped one-by-one from the end of ``alpha_beta_indices`` until a
    valid embedding is found.

    The construction strategy differs by topology:

    - **heavy-hex**: Ancilla qubits are inserted as intermediate nodes between
      each alpha-beta pair, reflecting the limited connectivity of heavy-hex
      devices. The number of ancilla nodes equals ``len(alpha_beta_indices)``.
    - **grid**: Alpha and beta nodes are connected directly with no ancilla qubits,
      exploiting the denser connectivity of grid devices.

    Args:
        num_orbitals: Number of spatial orbitals, i.e., the number of nodes
            in each linear chain (alpha-alpha and beta-beta).
        backend_coupling_graph: Undirected ``rustworkx.PyGraph`` of the
            target backend with at most one edge between any two nodes. Directed
            backends (e.g., Eagle devices such as ``ibm_brisbane``) and backends
            with duplicate edges (e.g., Heron devices such as ``ibm_torino``) must
            be preprocessed into this form before being passed in. See
            ``_make_backend_cmap_pygraph``.
        backend_topology: Connectivity topology of the
            backend. Determines whether ancilla nodes are inserted between
            alpha-beta pairs.
        alpha_beta_indices: Requested alpha-beta orbital
            interaction pairs. Modified in-place by dropping pairs from the end
            if they cannot be accommodated. List pairs in descending order of
            priority to control which are dropped first.

    Returns:
        - The LUCJ layout graph - two linear chains with
          alpha-beta connections (and ancilla nodes for heavy-hex topologies)
          - that is a valid isomorphic subgraph of ``backend_coupling_graph``.
        - The subset of ``alpha_beta_indices`` that can
          be accommodated given the backend's connectivity constraints.
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
    """Convert a backend coupling map to a filtered ``rustworkx.PyGraph``.

    This function constructs an undirected graph from the backend's coupling map,
    removes duplicate edges so that at most one edge exists between any two nodes,
    then prunes nodes and edges that exceed the specified error thresholds. Nodes
    and edges with missing instruction properties (``None``) are skipped from
    pruning.

    Args:
        backend: The target Qiskit backend.
        thresh_two_q: Two-qubit gate error threshold. Edges with error
            rate ``>= thresh_two_q`` are removed.
        thresh_meas: Measurement error threshold. Nodes with readout
            error ``>= thresh_meas`` are removed.

    Returns:
        An undirected ``rustworkx.PyGraph`` with at most one edge
        between any two nodes, with high-error nodes and edges removed.
    """
    two_q_gate_name = IBM_TWO_Q_GATES.intersection(backend.operation_names).pop()
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

    target = backend.target
    # remove bad nodes
    node_indices = backend_coupling_graph.node_indices()
    for node_id in node_indices:
        meas_prop = target["measure"][(node_id,)]
        if meas_prop is None:
            continue

        if meas_prop.error >= thresh_meas:
            backend_coupling_graph.remove_node(node_id)

    # remove bad edges
    edge_list = backend_coupling_graph.edge_list()
    for edge in edge_list:
        if edge in target[two_q_gate_name]:
            found_edge = edge
        else:
            found_edge = edge[::-1]

        gate_prop = target[two_q_gate_name][found_edge]
        if gate_prop is None:
            continue

        if gate_prop.error >= thresh_two_q:
            backend_coupling_graph.remove_edge(edge[0], edge[1])

    return backend_coupling_graph


def _get_placeholder_layout_and_allowed_interactions(
    backend: BackendV2,
    num_orbitals: int,
    backend_topology: Literal["heavy-hex", "grid"],
    requested_alpha_beta_indices: list[tuple[int, int]],
    thresh_two_q: float,
    thresh_meas: float,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Generate a placeholder layout and the accommodated alpha-beta interactions.

    Build a filtered backend coupling graph, find a subgraph isomorphic to the
    LUCJ layout graph, and return the corresponding physical qubit indices
    as an ``initial_layout`` for use in a Qiskit preset pass manager.
    If the backend cannot accommodate all requested alpha-beta pairs, pairs are
    dropped from the end of the list until a valid isomorphic subgraph is found.

    Args:
        backend: The target Qiskit backend.
        num_orbitals: Number of spatial orbitals. The ansatz uses
            ``2 * num_orbitals`` qubits plus any ancilla qubits.
        backend_topology: Connectivity topology of the
            backend. Must be one of ``"heavy-hex"`` or ``"grid"``.
        requested_alpha_beta_indices: Alpha-beta orbital
            interaction pairs to include in the ansatz. Pairs are dropped from the
            end of the list if they cannot be accommodated by the backend. List
            pairs in descending order of priority to control which are dropped first.
        thresh_two_q: Two-qubit gate error threshold. Edges in the backend
            coupling graph with error rate ``>= thresh_two_q`` are removed.
        thresh_meas: Measurement error threshold. Nodes in the backend
            coupling graph with readout error ``>= thresh_meas`` are removed.

    Returns:
        - Physical qubit indices forming a heavy-hex or grid topology
          compliant layout, suitable for use as ``initial_layout`` in
          a Qiskit preset pass manager.
        - The subset of requested alpha-beta pairs that
          could be accommodated given the backend's connectivity and error
          constraints.

    Raises:
        RuntimeError: If no alpha-beta interactions can be accommodated on the
            backend after pruning.
        RuntimeError: If no layout is found.
        ValueError: If the resulting ``initial_layout`` contains unset (negative)
            qubit indices, indicating an incomplete subgraph mapping.
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

    mapping = next(isomorphic_mappings, None)
    if mapping is None:
        raise RuntimeError("No layout is found.")

    initial_layout = [-1] * (2 * num_orbitals + num_allowed_alpha_beta_indices)

    for key, value in mapping.items():
        initial_layout[value] = key

    # as grids do not have ancillae qubits, the trailing items
    # in the ``initial_layout`` will remain as ``-1``. Therefore,
    # we are ignoring last several qubits in this check.
    if -1 in initial_layout[:-num_allowed_alpha_beta_indices]:
        raise ValueError(
            f"Not all qubits in the placeholder ``initial_layout`` is properly set."
            f"Negative qubit index in ``initial_layout``. "
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
    """Generate a Qiskit preset pass manager for the LUCJ ansatz.

    Construct a pass manager that maps a local unitary cluster Jastrow (LUCJ)
    ansatz circuit onto a target backend using layout strategies, as described
    in Motta et al. (2023) (https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02516k).
    The layout is optimized for heavy-hex or grid backend topologies, subject to
    hardware connectivity and error constraints.

    Args:
        backend: The target Qiskit backend.
        num_orbitals: Number of spatial orbitals. The ansatz uses
            ``2 * num_orbitals`` qubits plus any ancilla qubits.
        backend_topology: Connectivity topology of the
            backend. Must be one of ``"heavy-hex"`` or ``"grid"``.
        requested_alpha_beta_indices:
            Alpha-beta orbital interaction pairs to include in the ansatz. Each
            entry ``(alpha, beta)`` must satisfy ``0 <= alpha, beta < num_orbitals``.

            If hardware connectivity cannot accommodate all requested pairs, pairs
            are dropped from the end of the list until a valid layout is found.
            List pairs in descending order of priority to control which are dropped
            first.

            If ``None``, defaults to pairs as follows:

            - **heavy-hex**: ``[(p, p) for p in range(num_orbitals) if p % 4 == 0]``
            - **grid**: ``[(p, p) for p in range(num_orbitals)]``

            Default: ``None``.
        thresh_two_q: Two-qubit gate error threshold. Edges in the backend
            coupling graph with error rate ``>= thresh_two_q`` are removed.
            Default: ``1.0`` (removes only completely faulty edges).
        thresh_meas: Measurement error threshold. Nodes in the backend
            coupling graph with measurement error ``>= thresh_meas`` are removed.
            Default: ``0.10``.
        **qiskit_pm_kwargs: Additional keyword arguments forwarded to
            :func:`qiskit.transpiler.generate_preset_pass_manager`. The arguments
            ``initial_layout`` and ``layout_method`` are not supported and will be
            ignored with a warning if provided.

    Returns:
        - A configured Qiskit preset pass manager with the
          LUCJ layout and a ``VF2PostLayout`` pass enabled.
        - The subset of requested alpha-beta pairs that
          can be accommodated given the backend's connectivity and error
          constraints.

    Raises:
        ValueError: If any entry in ``requested_alpha_beta_indices`` references
            an orbital index ``>= num_orbitals``.
        ValueError: If ``backend_topology`` is not ``"heavy-hex"`` or ``"grid"``.

    Note:
        Providing ``initial_layout`` to
        :func:`~qiskit.transpiler.generate_preset_pass_manager` normally disables
        the ``VF2PostLayout`` pass. This function re-enables it explicitly so that
        the transpiler can search for a better noise-aware isomorphic subgraph mapping
        after routing.
    """
    if "initial_layout" in qiskit_pm_kwargs:
        warnings.warn("Argument ``initial_layout`` is ignored.")
        del qiskit_pm_kwargs["initial_layout"]

    if "layout_method" in qiskit_pm_kwargs:
        warnings.warn("Argument ``layout_method`` is ignored.")
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
            resolved_alpha_beta_indices = [
                (p, p) for p in range(num_orbitals) if p % 4 == 0
            ]
        elif backend_topology.lower() == "grid":
            resolved_alpha_beta_indices = [(p, p) for p in range(num_orbitals)]
    else:
        resolved_alpha_beta_indices = list(requested_alpha_beta_indices)

    (placeholder_initial_layout, allowed_alpha_beta_indices) = (
        _get_placeholder_layout_and_allowed_interactions(
            backend=backend,
            num_orbitals=num_orbitals,
            backend_topology=backend_topology,
            requested_alpha_beta_indices=resolved_alpha_beta_indices,
            thresh_two_q=thresh_two_q,
            thresh_meas=thresh_meas,
        )
    )

    pm = generate_preset_pass_manager(
        backend=backend, initial_layout=placeholder_initial_layout, **qiskit_pm_kwargs
    )
    pm.pre_init = ffsim.qiskit.PRE_INIT

    # generating a preset pass manager with ``initial_layout``
    # (``=placeholder_initial_layout``) disables the ``VF2PostLayout`` pass.
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
