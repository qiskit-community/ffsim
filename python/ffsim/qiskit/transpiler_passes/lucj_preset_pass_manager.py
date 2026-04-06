# (C) Copyright IBM 2026.
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
from collections import OrderedDict
from typing import Any, Literal

import rustworkx
from qiskit.circuit import Instruction
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.providers import BackendV2
from qiskit.transpiler import (
    StagedPassManager,
    generate_preset_pass_manager,
)
from qiskit.transpiler.passes import ApplyLayout, VF2PostLayout
from rustworkx import NoEdgeBetweenNodes, PyGraph

import ffsim

VF2_CALL_LIMIT = 30_000_000


def _create_two_linear_chains(
    pairs_aa: list[tuple[int, int]], pairs_bb: list[tuple[int, int]]
) -> PyGraph:
    """Create two disconnected linear qubit chains for the LUCJ ansatz.

    Construct a ``rustworkx.PyGraph`` with two disconnected linear chains
    representing the alpha (nodes ``0`` to ``num alpha orb - 1``) and beta
    (nodes ``num alpha orb`` to ``num alpha orb + num beta orb - 1``) qubits.
    This graph serves as the base structure for the LUCJ layout; alpha-beta
    connections between chains are added separately by
    ``_get_layout_graph_and_allowed_pairs_ab``.

    Args:
        pairs_aa: Alpha-alpha orbital interaction pairs.
        pairs_bb: Beta-beta orbital interaction pairs.

    Returns:
        A ``rustworkx.PyGraph`` containing two disconnected linear chains.
    """
    graph = rustworkx.PyGraph()

    for pairs in [pairs_aa, pairs_bb]:
        uniq: OrderedDict = OrderedDict()
        for q0, q1 in pairs:
            if q0 not in uniq:
                uniq[q0] = None

            if q1 not in uniq:
                uniq[q1] = None

        # add nodes
        for q in uniq.keys():
            node_idx = graph.add_node(q)
            uniq[q] = node_idx

        # add edges
        for q0, q1 in pairs:
            node_idx0, node_idx1 = uniq[q0], uniq[q1]
            graph.add_edge(node_idx0, node_idx1, None)

    return graph


def _get_layout_graph_and_allowed_pairs_ab(
    norb: int,
    backend_coupling_graph: PyGraph,
    connectivity: Literal["heavy-hex", "square"],
    pairs_aa: list[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
    pairs_bb: list[tuple[int, int]],
) -> tuple[PyGraph, list[tuple[int, int]]]:
    """Build a LUCJ layout graph and determine accommodated alpha-beta interactions.

    This routine constructs a candidate layout graph representing the LUCJ ansatz
    - two linear qubit chains (alpha and beta) connected by alpha-beta interactions
    - and verifies that it can be embedded as an isomorphic subgraph on the target
    backend. If the full set of requested alpha-beta pairs cannot be accommodated,
    pairs are dropped one-by-one from the end of ``pairs_ab`` until a
    valid embedding is found.

    The construction strategy differs by topology:

    - **heavy-hex**: Ancilla qubits are inserted as intermediate nodes between
      each alpha-beta pair, reflecting the limited connectivity of heavy-hex
      devices. The number of ancilla nodes equals ``len(pairs_ab)``.
    - **square**: Alpha and beta nodes are connected directly with no ancilla qubits,
      exploiting the denser connectivity of square devices.

    Args:
        norb: Number of spatial orbitals, i.e., the number of nodes
            in each linear chain (alpha-alpha and beta-beta).
        backend_coupling_graph: Undirected ``rustworkx.PyGraph`` of the
            target backend with at most one edge between any two nodes. Directed
            backends (e.g., Eagle devices such as ``ibm_brisbane``) and backends
            with duplicate edges (e.g., Heron devices such as ``ibm_torino``) must
            be preprocessed into this form before being passed in. See
            ``_make_backend_cmap_pygraph``.
        connectivity: Connectivity topology of the
            backend. Determines whether ancilla nodes are inserted between
            alpha-beta pairs.
        pairs_aa: Alpha-alpha orbital interaction pairs.
        pairs_ab: Requested alpha-beta orbital
            interaction pairs. Modified in-place by dropping pairs from the end
            if they cannot be accommodated. List pairs in descending order of
            priority to control which are dropped first.
        pairs_bb: Beta-beta orbital interaction pairs.

    Returns:
        - The LUCJ layout graph - two linear chains with
          alpha-beta connections (and ancilla nodes for heavy-hex topologies)
          - that is a valid isomorphic subgraph of ``backend_coupling_graph``.
        - The subset of ``pairs_ab`` that can
          be accommodated given the backend's connectivity constraints.
    """
    isomorphic = False
    graph = _create_two_linear_chains(pairs_aa=pairs_aa, pairs_bb=pairs_bb)

    num_nodes = graph.num_nodes()
    graph_new = copy.deepcopy(graph)  # to avoid not bound warning
    while (not isomorphic) and pairs_ab:
        graph_new = copy.deepcopy(graph)

        # add new nodes and edges
        if connectivity == "heavy-hex":
            for i, (a, b) in enumerate(sorted(pairs_ab, key=lambda x: x[0])):
                new_node = num_nodes + i
                graph_new.add_node(new_node)
                graph_new.add_edge(a, new_node, None)
                graph_new.add_edge(new_node, b + norb, None)
        elif connectivity == "square":
            for i, (a, b) in enumerate(sorted(pairs_ab, key=lambda x: x[0])):
                graph_new.add_edge(a, b + norb, None)
        else:
            ValueError(f"connectivity={connectivity} not supported.")

        isomorphic = rustworkx.is_subgraph_isomorphic(
            backend_coupling_graph,
            graph_new,
            call_limit=VF2_CALL_LIMIT,
            id_order=False,
            induced=False,
        )

        if not isomorphic:
            warnings.warn(
                f"Backend cannot accommodate pairs_ab={pairs_ab}.\n"
                f"Removing interaction {pairs_ab[-1]} from the end."
            )
            del pairs_ab[-1]

    return graph_new, pairs_ab


def _collect_two_q_quantum_gates(operations_in_target: list[Instruction]) -> list[str]:
    """Collect two-qubit gate names from operations in a backend Target object.

    Args:
        operations_in_target: Operations supported by the target.

    Returns:
        A list of two-qubit operation names as ``str``.
    """
    excluded_insts = ["barrier", "reset", "measurement"]
    two_q_gate_names = []
    for inst in operations_in_target:
        if inst.num_qubits == 2 and inst.name not in excluded_insts:
            two_q_gate_names.append(inst.name)

    return two_q_gate_names


def _make_backend_cmap_pygraph(
    backend: BackendV2,
    two_qubit_error_threshold: float,
    readout_error_threshold: float,
) -> PyGraph:
    """Convert a backend coupling map to a filtered ``rustworkx.PyGraph``.

    This function constructs an undirected graph from the backend's coupling map,
    removes duplicate edges so that at most one edge exists between any two nodes,
    then prunes nodes and edges that exceed the specified error thresholds. Nodes
    and edges with missing instruction properties (``None``) are skipped from
    pruning.

    Args:
        backend: The target Qiskit backend.
        two_qubit_error_threshold: Two-qubit gate error threshold. Edges with error
            rate ``>= two_qubit_error_threshold`` are removed.
        readout_error_threshold: Readout error threshold. Nodes with readout
            error ``>= readout_error_threshold`` are removed.

    Returns:
        An undirected ``rustworkx.PyGraph`` with at most one edge
        between any two nodes, with high-error nodes and edges removed.
    """
    target = backend.target
    two_q_gate_names = _collect_two_q_quantum_gates(target.operations)
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
        meas_prop = target["measure"][(node_id,)]
        if meas_prop is None:
            continue

        if meas_prop.error >= readout_error_threshold:
            backend_coupling_graph.remove_node(node_id)

    # remove bad edges
    edge_list = backend_coupling_graph.edge_list()
    for gate in two_q_gate_names:
        for edge in edge_list:
            if edge in target[gate]:
                found_edge = edge
            else:
                found_edge = edge[::-1]

            gate_prop = target[gate][found_edge]
            if gate_prop is None:
                continue

            if gate_prop.error >= two_qubit_error_threshold:
                backend_coupling_graph.remove_edge(edge[0], edge[1])

    return backend_coupling_graph


def _get_placeholder_layout_and_allowed_interactions(
    backend: BackendV2,
    norb: int,
    connectivity: Literal["heavy-hex", "square"],
    pairs_aa: list[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
    pairs_bb: list[tuple[int, int]],
    two_qubit_error_threshold: float,
    readout_error_threshold: float,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Generate a placeholder layout and the accommodated alpha-beta interactions.

    Build a filtered backend coupling graph, find a subgraph isomorphic to the
    LUCJ layout graph, and return the corresponding physical qubit indices
    as an ``initial_layout`` for use in a Qiskit preset pass manager.
    If the backend cannot accommodate all requested alpha-beta pairs, pairs are
    dropped from the end of the list until a valid isomorphic subgraph is found.

    Args:
        backend: The target Qiskit backend.
        norb: Number of spatial orbitals. The ansatz uses
            ``2 * norb`` qubits, and transpilation may add ancilla qubits.
        connectivity: Connectivity topology of the
            backend. Must be one of ``"heavy-hex"`` or ``"square"``.
        pairs_aa: Alpha-alpha orbital interaction pairs.
        pairs_ab: Alpha-beta orbital
            interaction pairs to include in the ansatz. Pairs are dropped from the
            end of the list if they cannot be accommodated by the backend. List
            pairs in descending order of priority to control which are dropped first.
        pairs_bb: Beta-beta orbital interaction pairs.
        two_qubit_error_threshold: Two-qubit gate error threshold. Edges in the backend
            coupling graph with error rate ``>= two_qubit_error_threshold`` are removed.
        readout_error_threshold: Readout error threshold. Nodes in the backend
            coupling graph with readout error ``>= readout_error_threshold``
            are removed.

    Returns:
        - Physical qubit indices forming a heavy-hex or square topology
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
        backend=backend,
        two_qubit_error_threshold=two_qubit_error_threshold,
        readout_error_threshold=readout_error_threshold,
    )

    (layout_graph, allowed_pairs_ab) = _get_layout_graph_and_allowed_pairs_ab(
        norb=norb,
        backend_coupling_graph=backend_coupling_graph,
        connectivity=connectivity,
        pairs_aa=pairs_aa,
        pairs_ab=pairs_ab,
        pairs_bb=pairs_bb,
    )
    num_allowed_pairs_ab = len(allowed_pairs_ab)
    if num_allowed_pairs_ab == 0:
        raise RuntimeError(
            "No alpha-beta interaction can be accommodated. Terminating."
        )

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

    initial_layout = [-1] * (2 * norb + num_allowed_pairs_ab)

    for key, value in mapping.items():
        initial_layout[value] = key

    # as squares do not have ancilla qubits, the trailing items
    # in the ``initial_layout`` will remain as ``-1``. Therefore,
    # we are ignoring last several qubits in this check.
    if -1 in initial_layout[:-num_allowed_pairs_ab]:
        raise ValueError(
            f"Not all qubits in the placeholder ``initial_layout`` is properly set."
            f"Negative qubit index in ``initial_layout``. "
            f"initial_layout={initial_layout[:-num_allowed_pairs_ab]}"
        )

    return initial_layout[:-num_allowed_pairs_ab], allowed_pairs_ab


def generate_lucj_pass_manager(
    backend: BackendV2,
    norb: int,
    connectivity: Literal["heavy-hex", "square"],
    interaction_pairs: tuple[list[tuple[int, int]], list[tuple[int, int]] | None]
    | tuple[list[tuple[int, int]], list[tuple[int, int]] | None, list[tuple[int, int]]],
    two_qubit_error_threshold: float = 1.0,
    readout_error_threshold: float = 0.10,
    **qiskit_pm_kwargs,
) -> tuple[StagedPassManager, list[tuple[int, int]]]:
    """Generate a Qiskit preset pass manager for the LUCJ ansatz.

    Construct a pass manager that maps a local unitary cluster Jastrow (LUCJ)
    ansatz circuit onto a target backend using layout strategies, as described
    in the reference below. The layout is optimized for heavy-hex or square
    backend topologies, subject to hardware connectivity and error constraints.

    References:
        - `Motta, Sung, Whaley, Head-Gordon, and Shee, "Bridging physical intuition and hardware efficiency for correlated electronic states: the local unitary cluster Jastrow ansatz for electronic structure." (2023)`_

    Args:
        backend: The target Qiskit backend.
        norb: Number of spatial orbitals. The ansatz uses
            ``2 * norb`` qubits, and transpilation may add ancilla qubits.
        connectivity: Connectivity topology of the
            backend. Must be one of ``"heavy-hex"`` or ``"square"``.
        interaction_pairs:
            Either a length-2 or length-3 tuple describing alpha-alpha, alpha-beta,
            and optional beta-beta interaction pairs.

            The first item in the tuple is the alpha-alpha interaction pairs (``pairs_aa``).
            The second item is the alpha-beta interaction pairs (``pairs_ab``).
            And, the optional third item is the beta-beta interaction pairs (``pairs_bb``).

            Each entry ``(orb1, orb2)`` in above three lists of interaction  pairs
            must satisfy ``0 <= orb1, orb2 < norb``.

            ``pairs_ab`` interaction pairs are *requested* by a user to be include
            in the ansatz. If hardware connectivity cannot accommodate all requested
            pairs, pairs are dropped from the end of the list until a valid layout
            is found. List pairs in descending order of priority to control which
            are dropped first.

            If ``None``, defaults to pairs as follows:

            - **heavy-hex**: ``[(p, p) for p in range(norb) if p % 4 == 0]``
            - **square**: ``[(p, p) for p in range(norb)]``

        two_qubit_error_threshold: Two-qubit gate error threshold. Edges in the backend
            coupling graph with error rate ``>= two_qubit_error_threshold`` are removed.
            The default value, ``1.0``, results in only completely faulty
            edges being removed.
        readout_error_threshold: Readout error threshold. Nodes in the backend
            coupling graph with readout error ``>= readout_error_threshold`` are removed.
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
        ValueError: If any entry in ``interaction_pairs`` references
            an orbital index ``>= norb``.
        ValueError: If ``connectivity`` is not ``"heavy-hex"`` or ``"square"``.

    Note:
        Providing ``initial_layout`` to
        :func:`~qiskit.transpiler.generate_preset_pass_manager` normally disables
        the ``VF2PostLayout`` pass. This function re-enables it explicitly so that
        the transpiler can search for a better noise-aware isomorphic subgraph mapping
        after routing.

    .. _`Motta, Sung, Whaley, Head-Gordon, and Shee, "Bridging physical intuition and hardware efficiency for correlated electronic states: the local unitary cluster Jastrow ansatz for electronic structure." (2023)`: https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc02516k
    """  # noqa: E501
    if "initial_layout" in qiskit_pm_kwargs:
        warnings.warn("Argument ``initial_layout`` is ignored.")
        del qiskit_pm_kwargs["initial_layout"]

    if "layout_method" in qiskit_pm_kwargs:
        warnings.warn("Argument ``layout_method`` is ignored.")
        del qiskit_pm_kwargs["layout_method"]

    if len(interaction_pairs) == 2:
        pairs_aa, pairs_ab = interaction_pairs
        pairs_bb = pairs_aa
    elif len(interaction_pairs) == 3:
        pairs_aa, pairs_ab, pairs_bb = interaction_pairs
    else:
        raise ValueError(
            "``interaction_pairs`` must be either length-2 or length-3 tuple."
        )

    for pairs in interaction_pairs:
        if pairs is None:
            continue

        for p0, p1 in pairs:
            if not ((0 <= p0 < norb) and (0 <= p1 < norb)):
                raise ValueError(
                    f"Interaction pair {(p0, p1)} is not in range "
                    f"norb[0, norb) = [0, {norb})."
                )

    if connectivity.lower() not in ["heavy-hex", "square"]:
        raise ValueError(
            f"connectivity={connectivity} is not supported. "
            f"Only supported topologies are either 'heavy-hex' or 'square'"
        )

    if pairs_ab is None:
        if connectivity.lower() == "heavy-hex":
            resolved_pairs_ab = [(p, p) for p in range(norb) if p % 4 == 0]
        elif connectivity.lower() == "square":
            resolved_pairs_ab = [(p, p) for p in range(norb)]
    else:
        resolved_pairs_ab = copy.deepcopy(pairs_ab)

    (placeholder_initial_layout, allowed_pairs_ab) = (
        _get_placeholder_layout_and_allowed_interactions(
            backend=backend,
            norb=norb,
            connectivity=connectivity,
            pairs_ab=resolved_pairs_ab,
            two_qubit_error_threshold=two_qubit_error_threshold,
            readout_error_threshold=readout_error_threshold,
            pairs_aa=pairs_aa,
            pairs_bb=pairs_bb,
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

    return pm, allowed_pairs_ab
