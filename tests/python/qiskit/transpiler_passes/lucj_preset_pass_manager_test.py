# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for LUCJ preset pass manager."""

from typing import Literal, cast

import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap

import ffsim
from ffsim.qiskit import generate_lucj_pass_manager

cmap_square = CouplingMap.from_grid(num_rows=12, num_columns=10)
backend_square = GenericBackendV2(
    num_qubits=cmap_square.size(), coupling_map=cmap_square, noise_info=True
)
backend_noise_info_none = GenericBackendV2(
    num_qubits=cmap_square.size(), coupling_map=cmap_square, noise_info=False
)  # creates a backend with ``None`` as error rates


cmap_heavy_hex = CouplingMap.from_heavy_hex(distance=9, bidirectional=True)
backend_heavy_hex = GenericBackendV2(
    num_qubits=cmap_heavy_hex.size(), coupling_map=cmap_heavy_hex, noise_info=True
)

cmap_heavy_hex_directed = CouplingMap.from_heavy_hex(distance=9, bidirectional=False)
backend_heavy_hex_directed = GenericBackendV2(
    num_qubits=cmap_heavy_hex.size(), coupling_map=cmap_heavy_hex, noise_info=True
)

norb = 36


@pytest.mark.parametrize(
    "connectivity_and_backend",
    [("heavy-hex", backend_heavy_hex), ("square", backend_square)],
)
def test_raise_warning1(connectivity_and_backend):
    """Test UserWarning raised when ``initial_layout`` is specified and ignored."""
    connectivity, backend = connectivity_and_backend
    with pytest.warns(UserWarning, match="Argument ``initial_layout`` is ignored."):
        _, _ = generate_lucj_pass_manager(
            backend=backend,
            norb=norb,
            connectivity=connectivity,
            alpha_beta_interaction_pairs=None,
            optimization_level=3,
            initial_layout=[1],
        )


@pytest.mark.parametrize(
    "connectivity_and_backend",
    [("heavy-hex", backend_heavy_hex), ("square", backend_square)],
)
def test_raise_warning2(connectivity_and_backend):
    """Test UserWarning raised when ``layout_method`` is specified and ignored."""
    connectivity, backend = connectivity_and_backend
    with pytest.warns(UserWarning, match="Argument ``layout_method`` is ignored."):
        _, _ = generate_lucj_pass_manager(
            backend=backend,
            norb=norb,
            connectivity=connectivity,
            alpha_beta_interaction_pairs=None,
            optimization_level=3,
            layout_method="placeholder",
        )


@pytest.mark.parametrize(
    "connectivity_and_backend",
    [("heavy-hex", backend_heavy_hex), ("square", backend_square)],
)
@pytest.mark.parametrize(
    "alpha_beta_interaction_pairs",
    [[(norb + 1, norb + 1)], [(norb, norb)]],
)
def test_raise_value_error1(alpha_beta_interaction_pairs, connectivity_and_backend):
    """Test ValueError raised when requested alpha-beta > num orbitals."""
    connectivity, backend = connectivity_and_backend
    with pytest.raises(ValueError):
        _, _ = generate_lucj_pass_manager(
            backend=backend,
            norb=norb,
            connectivity=connectivity,
            alpha_beta_interaction_pairs=alpha_beta_interaction_pairs,
            optimization_level=3,
        )


def test_raise_value_error2():
    """Test ValueError raised when connectivity is neither 'heavy-hex' nor 'square'."""
    with pytest.raises(ValueError):
        _, _ = generate_lucj_pass_manager(
            backend=backend_heavy_hex,
            norb=norb,
            connectivity=cast(
                Literal["heavy-hex", "square"], "line"
            ),  # to avoid mypy error
            alpha_beta_interaction_pairs=None,
            optimization_level=3,
        )


def test_backend_with_none_noise_info():
    """Test handling of backend with no noise info."""
    _, allowed_alpha_beta_pairs = generate_lucj_pass_manager(
        backend=backend_noise_info_none,
        norb=norb,
        connectivity="square",
        alpha_beta_interaction_pairs=None,
        optimization_level=1,
    )

    assert allowed_alpha_beta_pairs


@pytest.mark.parametrize(
    "connectivity_and_backend",
    [
        ("heavy-hex", backend_heavy_hex),
        ("heavy-hex", backend_heavy_hex_directed),
        ("square", backend_square),
    ],
)
@pytest.mark.parametrize(
    "alpha_beta_interaction_pairs",
    [None, [(32, 32), (4, 4), (8, 8), (24, 24), (16, 16), (28, 28)]],
)
def test_generate_lucj_pass_manager(
    alpha_beta_interaction_pairs, connectivity_and_backend
):
    """Test whether the transpiled LUCJ ansatz retains correct a-b interactions.

    On heavy-hex, it should alpha - ancilla - beta, and on square, it must
    be alpha - beta, i.e.,
    1. A qubit on the alpha (beta) chain must be 1 distance (edge) apart
        from the next qubit on the chain, on both heavy-hex and square.
    2. Distance between an alpha and a beta qubit on a heavy-hex must be 2,
        and on a square must be 1.
    """
    connectivity, backend = connectivity_and_backend
    n_reps = 2
    alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
    pm, alpha_beta_indices = generate_lucj_pass_manager(
        backend=backend,
        norb=norb,
        connectivity=connectivity,
        alpha_beta_interaction_pairs=alpha_beta_interaction_pairs,
        optimization_level=3,
    )

    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
        seed=0,
    )

    nelec = (5, 5)

    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    circuit.measure_all()

    isa_circuit = pm.run(circuit)
    initial_layout = isa_circuit.layout.initial_index_layout(filter_ancillas=False)
    alpha_qubits = initial_layout[:norb]
    beta_qubits = initial_layout[norb : 2 * norb]

    coupling_map = backend.target.build_coupling_map()

    for idx, _ in alpha_beta_indices:
        dist = coupling_map.distance(alpha_qubits[idx], beta_qubits[idx])
        if connectivity == "heavy-hex":
            # alpha and beta qubits
            # connected by an ancilla
            # must be 2 edges apart
            # on heavy-hex
            # alpha - ancilla - beta
            assert dist == 2
        elif connectivity == "square":
            # alpha and beta qubits
            # must be 2 edges apart
            # on square
            # alpha - beta
            assert dist == 1

    # Adjacent qubits on alpha/beta chains are 1 edge apart
    for idx1 in range(norb - 1):
        idx2 = idx1 + 1

        dist = coupling_map.distance(alpha_qubits[idx1], alpha_qubits[idx2])
        assert dist == 1

        dist = coupling_map.distance(beta_qubits[idx1], beta_qubits[idx2])
        assert dist == 1
