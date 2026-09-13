# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Parameterized Qiskit circuits for the LUCJ ansatz."""

from __future__ import annotations

import cmath
import itertools
import math
from collections.abc import Iterator

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CPhaseGate, PhaseGate, XXPlusYYGate

from ffsim import variational
from ffsim.variational.util import validate_interaction_pairs

SpinBalancedInteractionPairs = tuple[
    list[tuple[int, int]] | None, list[tuple[int, int]] | None
]


def lucj_spin_balanced_ansatz(
    norb: int,
    n_reps: int,
    *,
    interaction_pairs: SpinBalancedInteractionPairs | None = None,
    parameter_prefix: str = "theta",
) -> QuantumCircuit:
    """Return a parameterized Qiskit circuit for the spin-balanced LUCJ ansatz.

    The returned circuit implements the merged orbital-rotation form obtained by
    expanding :class:`ffsim.qiskit.UCJOpSpinBalancedJW`, running
    :data:`ffsim.qiskit.PRE_INIT` to merge orbital rotations, decomposing the
    result, and replacing the gate angles with Qiskit parameters. Independent
    commuting gates can appear in a different serial order.

    Note:
        This is a Qiskit gate-angle parameterization. Symmetry-related gates,
        such as alpha and beta orbital-rotation gates, receive distinct Qiskit
        parameters. Initializer functions return repeated values for those
        parameters so the bound circuit recovers the spin-balanced UCJ operator.

    Args:
        norb: The number of spatial orbitals.
        n_reps: The number of ansatz repetitions.
        interaction_pairs: Optional restrictions on allowed orbital interactions
            for the diagonal Coulomb operators. If specified, this should be a pair
            of lists for alpha-alpha and alpha-beta interactions, respectively. A
            list can be substituted with ``None`` to indicate no restrictions. Each
            pair must be upper triangular.
        parameter_prefix: Prefix for the generated Qiskit parameters.

    Returns:
        A parameterized Qiskit circuit over ``2 * norb`` qubits.

    Raises:
        ValueError: ``norb`` or ``n_reps`` was not positive.
        ValueError: Interaction pairs list contained duplicate interactions.
        ValueError: Interaction pairs list contained lower triangular pairs.
    """
    if norb <= 0:
        raise ValueError(f"norb must be at least 1. Got {norb}.")
    if n_reps <= 0:
        raise ValueError(f"n_reps must be at least 1. Got {n_reps}.")

    pairs_aa, pairs_ab = _normalize_interaction_pairs(norb, interaction_pairs)
    n_params = _n_qiskit_parameters(norb, n_reps, pairs_aa, pairs_ab)
    params = ParameterVector(parameter_prefix, n_params)
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)

    index = 0
    for layer in range(n_reps):
        index = _append_spinful_orbital_rotation(circuit, qubits, norb, params, index)
        index = _append_diag_coulomb(
            circuit, qubits, norb, pairs_aa, pairs_ab, params, index
        )
    index = _append_spinful_orbital_rotation(circuit, qubits, norb, params, index)
    assert index == n_params
    return circuit


def lucj_spin_balanced_parameters(
    ucj_op: variational.UCJOpSpinBalanced,
    *,
    interaction_pairs: SpinBalancedInteractionPairs | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """Return Qiskit gate-angle values for a spin-balanced LUCJ ansatz circuit.

    The returned parameter values match the order of the parameters in
    :func:`lucj_spin_balanced_ansatz` with the same ``norb``, ``n_reps``, and
    ``interaction_pairs``. Binding these values to the ansatz circuit reproduces
    the same unitary as the merged Qiskit circuit for ``ucj_op``.

    Args:
        ucj_op: The spin-balanced UCJ operator to convert.
        interaction_pairs: Optional restrictions on allowed orbital interactions
            for the diagonal Coulomb operators. The same value must be used when
            constructing the ansatz circuit.
        tol: Tolerance used when converting orbital rotations into Givens rotations.

    Returns:
        A one-dimensional array of Qiskit gate-angle values.

    Raises:
        ValueError: The operator has no ansatz repetitions.
        ValueError: Interaction pairs list contained duplicate interactions.
        ValueError: Interaction pairs list contained lower triangular pairs.
    """
    norb = ucj_op.norb
    n_reps = ucj_op.n_reps
    if n_reps <= 0:
        raise ValueError("ucj_op must have at least one ansatz repetition.")

    pairs_aa, pairs_ab = _normalize_interaction_pairs(norb, interaction_pairs)
    values: list[float] = []
    orbital_rotations = _merged_orbital_rotations(ucj_op)
    for layer in range(n_reps):
        values.extend(_spinful_orbital_rotation_values(orbital_rotations[layer], tol))
        values.extend(
            _diag_coulomb_values(ucj_op.diag_coulomb_mats[layer], pairs_aa, pairs_ab)
        )
    values.extend(_spinful_orbital_rotation_values(orbital_rotations[-1], tol))

    expected = _n_qiskit_parameters(norb, n_reps, pairs_aa, pairs_ab)
    assert len(values) == expected
    return np.array(values)


def lucj_spin_balanced_parameters_from_t2(
    t2: np.ndarray,
    *,
    t1: np.ndarray | None = None,
    n_reps: int | None = None,
    interaction_pairs: SpinBalancedInteractionPairs | None = None,
    tol: float = 1e-8,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    regularization: float = 0,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    orbital_rotation_tol: float = 1e-12,
) -> np.ndarray:
    """Return LUCJ ansatz parameter values initialized from t2 amplitudes.

    The returned parameter values match the order of the parameters in
    :func:`lucj_spin_balanced_ansatz` with the same ``norb``, ``n_reps``, and
    ``interaction_pairs``. The UCJ tensors are initialized with
    :meth:`ffsim.UCJOpSpinBalanced.from_t_amplitudes`.

    Args:
        t2: The t2 amplitudes.
        t1: The t1 amplitudes.
        n_reps: The number of ansatz repetitions. If not specified, the number of
            repetitions is chosen by :meth:`ffsim.UCJOpSpinBalanced.from_t_amplitudes`.
        interaction_pairs: Optional restrictions on allowed orbital interactions
            for the diagonal Coulomb operators. The same value must be used when
            constructing the ansatz circuit.
        tol: Tolerance for error in the double-factorized decomposition of ``t2``.
        optimize: Whether to optimize the tensors returned by the decomposition.
        method: The optimization method. This argument is ignored unless
            ``optimize`` is set to ``True``.
        callback: Callback function for the optimization. This argument is ignored
            unless ``optimize`` is set to ``True``.
        options: Options for the optimization. This argument is ignored unless
            ``optimize`` is set to ``True``.
        regularization: Regularization to use during tensor optimization.
        multi_stage_start: Initial number of tensor terms for multi-stage
            optimization.
        multi_stage_step: Number of tensor terms to add at each multi-stage
            optimization step.
        orbital_rotation_tol: Tolerance used when converting orbital rotations into
            Givens rotations.

    Returns:
        A one-dimensional array of Qiskit gate-angle values.

    Raises:
        ValueError: ``n_reps`` was not positive.
        ValueError: Interaction pairs list contained duplicate interactions.
        ValueError: Interaction pairs list contained lower triangular pairs.
    """
    ucj_op = variational.UCJOpSpinBalanced.from_t_amplitudes(
        t2,
        t1=t1,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
        tol=tol,
        optimize=optimize,
        method=method,
        callback=callback,
        options=options,
        regularization=regularization,
        multi_stage_start=multi_stage_start,
        multi_stage_step=multi_stage_step,
    )
    return lucj_spin_balanced_parameters(
        ucj_op, interaction_pairs=interaction_pairs, tol=orbital_rotation_tol
    )


def _normalize_interaction_pairs(
    norb: int, interaction_pairs: SpinBalancedInteractionPairs | None
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if interaction_pairs is None:
        interaction_pairs = (None, None)
    pairs_aa, pairs_ab = interaction_pairs
    validate_interaction_pairs(pairs_aa, ordered=False)
    validate_interaction_pairs(pairs_ab, ordered=False)
    triu_indices = list(itertools.combinations_with_replacement(range(norb), 2))
    return (
        triu_indices if pairs_aa is None else pairs_aa,
        triu_indices if pairs_ab is None else pairs_ab,
    )


def _n_qiskit_parameters(
    norb: int,
    n_reps: int,
    pairs_aa: list[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
) -> int:
    n_orbital_rotation_params = 2 * norb**2
    n_aa_params = 2 * len(pairs_aa)
    n_ab_params = sum(1 if i == j else 2 for i, j in pairs_ab)
    return (n_reps + 1) * n_orbital_rotation_params + n_reps * (
        n_aa_params + n_ab_params
    )


def _append_spinful_orbital_rotation(
    circuit: QuantumCircuit,
    qubits: QuantumRegister,
    norb: int,
    params: ParameterVector,
    index: int,
) -> int:
    for offset in (0, norb):
        for i, j in _givens_indices(norb):
            circuit.append(
                XXPlusYYGate(params[index], params[index + 1]),
                (qubits[offset + i], qubits[offset + j]),
            )
            index += 2
        for i in range(norb):
            circuit.append(PhaseGate(params[index]), (qubits[offset + i],))
            index += 1
    return index


def _append_diag_coulomb(
    circuit: QuantumCircuit,
    qubits: QuantumRegister,
    norb: int,
    pairs_aa: list[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
    params: ParameterVector,
    index: int,
) -> int:
    for offset in (0, norb):
        for i, j in _ordered_diag_coulomb_pairs(norb, pairs_aa):
            if i == j:
                circuit.append(PhaseGate(params[index]), (qubits[offset + i],))
            else:
                circuit.append(
                    CPhaseGate(params[index]), (qubits[offset + i], qubits[offset + j])
                )
            index += 1

    for i, j in _ordered_diag_coulomb_pairs(norb, pairs_ab):
        circuit.append(CPhaseGate(params[index]), (qubits[i], qubits[j + norb]))
        index += 1
        if i != j:
            circuit.append(CPhaseGate(params[index]), (qubits[j], qubits[i + norb]))
            index += 1
    return index


def _merged_orbital_rotations(
    ucj_op: variational.UCJOpSpinBalanced,
) -> list[np.ndarray]:
    orbital_rotations = ucj_op.orbital_rotations
    merged = [orbital_rotations[0].T.conj()]
    for previous, current in itertools.pairwise(orbital_rotations):
        merged.append(current.T.conj() @ previous)

    final_rotation = orbital_rotations[-1]
    if ucj_op.final_orbital_rotation is not None:
        final_rotation = ucj_op.final_orbital_rotation @ final_rotation
    merged.append(final_rotation)
    return merged


def _spinful_orbital_rotation_values(
    orbital_rotation: np.ndarray, tol: float
) -> Iterator[float]:
    values = list(_orbital_rotation_values(orbital_rotation, tol))
    yield from values
    yield from values


def _orbital_rotation_values(
    orbital_rotation: np.ndarray, tol: float
) -> Iterator[float]:
    givens_rotations, phase_shifts = _full_givens_decomposition(
        orbital_rotation, tol=tol
    )
    for c, s, _, _ in givens_rotations:
        yield 2 * math.acos(c)
        yield cmath.phase(s) - 0.5 * math.pi
    for phase_shift in phase_shifts:
        yield cmath.phase(phase_shift)


def _diag_coulomb_values(
    diag_coulomb_mats: np.ndarray,
    pairs_aa: list[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
) -> Iterator[float]:
    mat_aa, mat_ab = diag_coulomb_mats
    for i, j in _ordered_diag_coulomb_pairs(mat_aa.shape[0], pairs_aa):
        value = 0.5 * mat_aa[i, i] if i == j else mat_aa[i, j]
        yield value
    for i, j in _ordered_diag_coulomb_pairs(mat_aa.shape[0], pairs_aa):
        value = 0.5 * mat_aa[i, i] if i == j else mat_aa[i, j]
        yield value
    for i, j in _ordered_diag_coulomb_pairs(mat_aa.shape[0], pairs_ab):
        yield mat_ab[i, j]
        if i != j:
            yield mat_ab[j, i]


def _ordered_diag_coulomb_pairs(
    norb: int, pairs: list[tuple[int, int]]
) -> Iterator[tuple[int, int]]:
    pair_set = set(pairs)
    for i in range(norb):
        if (i, i) in pair_set:
            yield i, i
    for i, j in _iter_pairs(norb):
        if (i, j) in pair_set:
            yield i, j


def _iter_pairs(n: int) -> Iterator[tuple[int, int]]:
    for distance in range(1, n):
        for offset in range(distance + 1):
            for i in range(offset, n - distance, distance + 1):
                yield i, i + distance


def _givens_indices(norb: int) -> Iterator[tuple[int, int]]:
    right_indices: list[tuple[int, int]] = []
    left_indices: list[tuple[int, int]] = []
    for i in range(norb - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                target_index = i - j
                right_indices.append((target_index + 1, target_index))
        else:
            for j in range(i + 1):
                target_index = norb - i + j - 1
                left_indices.append((target_index - 1, target_index))
    right_indices.extend(reversed(left_indices))
    yield from right_indices


def _full_givens_decomposition(
    mat: np.ndarray, tol: float = 1e-12
) -> tuple[list[tuple[float, complex, int, int]], np.ndarray]:
    shape = mat.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("mat must be a square matrix")

    n = shape[0]
    current_matrix = mat.astype(complex, copy=True)
    left_rotations: list[tuple[float, complex, int, int]] = []
    right_rotations: list[tuple[float, complex, int, int]] = []

    for i in range(n - 1):
        if i % 2 == 0:
            for j in range(i + 1):
                target_index = i - j
                row = n - j - 1
                c, s = _zrotg_safe(
                    current_matrix[row, target_index + 1],
                    current_matrix[row, target_index],
                    tol,
                )
                right_rotations.append((c, s, target_index + 1, target_index))
                _rotate_columns_in_place(
                    current_matrix, target_index + 1, target_index, c, s
                )
        else:
            for j in range(i + 1):
                target_index = n - i + j - 1
                col = j
                c, s = _zrotg_safe(
                    current_matrix[target_index - 1, col],
                    current_matrix[target_index, col],
                    tol,
                )
                left_rotations.append((c, s, target_index - 1, target_index))
                _rotate_rows_in_place(
                    current_matrix, target_index - 1, target_index, c, s
                )

    for c_left, s_left, i, j in reversed(left_rotations):
        c, s = _zrotg_safe(
            c_left * current_matrix[j, j],
            s_left.conjugate() * current_matrix[i, i],
            tol,
        )
        right_rotations.append((c, -s.conjugate(), i, j))

        diag_i = current_matrix[i, i]
        diag_j = current_matrix[j, j]
        g00 = c * diag_i
        g01 = -s * diag_j
        g10 = s.conjugate() * diag_i
        g11 = c * diag_j
        c_new, s_new = _zrotg_safe(g11, g10, tol)

        current_matrix[i, i] = g00 * c_new + g01 * (-s_new.conjugate())
        current_matrix[j, j] = g10 * s_new + g11 * c_new

    return right_rotations, current_matrix.diagonal()


def _zrotg_safe(a: complex, b: complex, tol: float) -> tuple[float, complex]:
    abs_a = abs(a)
    abs_b = abs(b)
    if abs_b <= tol:
        return 1.0, 0.0j
    if abs_a <= tol:
        return 0.0, 1.0 + 0.0j
    r = math.hypot(abs_a, abs_b)
    c = abs_a / r
    s = (a / abs_a) * b.conjugate() / r
    return min(max(c, -1.0), 1.0), s


def _rotate_columns_in_place(
    mat: np.ndarray, col_x: int, col_y: int, c: float, s: complex
) -> None:
    x_old = mat[:, col_x].copy()
    y_old = mat[:, col_y].copy()
    mat[:, col_x] = c * x_old + s * y_old
    mat[:, col_y] = c * y_old - s.conjugate() * x_old


def _rotate_rows_in_place(
    mat: np.ndarray, row_x: int, row_y: int, c: float, s: complex
) -> None:
    x_old = mat[row_x].copy()
    y_old = mat[row_y].copy()
    mat[row_x] = c * x_old + s * y_old
    mat[row_y] = c * y_old - s.conjugate() * x_old
