# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Transpiler pass to merge consecutive orbital rotation gates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from ffsim.qiskit.gates import (
    OrbitalRotationJW,
    OrbitalRotationSpinlessJW,
    PrepareSlaterDeterminantJW,
    PrepareSlaterDeterminantSpinlessJW,
)


def _min_or_none(values: Iterable[int | None]) -> int | None:
    """Minimum of the non-``None`` values, or ``None`` if they are all ``None``.

    Compression budgets (``max_givens``, ``max_layers``) are combined with this
    reduction when merging consecutive orbital rotations into a single orbital rotation:
    the merged gate adopts the tightest (most approximate) budget among the gates being
    merged, mirroring how ``tol`` is combined with ``max``. ``None`` means "no
    constraint" and is ignored. (When an orbital rotation is instead absorbed into a
    Slater determinant preparation, the merged gate keeps the Slater gate's own budget
    rather than combining it, because the two decompositions use different scales.)
    """
    present = [value for value in values if value is not None]
    return min(present) if present else None


class MergeOrbitalRotations(TransformationPass):
    """Merge consecutive orbital rotation gates."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # Merge orbital rotation gates
        for run in dag.collect_runs(["orb_rot_jw"]):
            node = run[0]
            qubits = node.qargs
            norb = node.op.norb
            combined_mat_a = np.eye(norb)
            combined_mat_b = np.eye(norb)
            tol = max(node.op.tol for node in run)
            max_givens = _min_or_none(node.op.max_givens for node in run)
            max_layers = _min_or_none(node.op.max_layers for node in run)
            optimize_kwargs = run[0].op.optimize_kwargs
            for node in run:
                combined_mat_a = node.op.orbital_rotation_a @ combined_mat_a
                combined_mat_b = node.op.orbital_rotation_b @ combined_mat_b
            dag.replace_block_with_op(
                run,
                OrbitalRotationJW(
                    norb,
                    (combined_mat_a, combined_mat_b),
                    tol=tol,
                    max_givens=max_givens,
                    max_layers=max_layers,
                    **optimize_kwargs,
                ),
                {q: i for i, q in enumerate(qubits)},
                cycle_check=False,
            )

        # Merge Slater determinant preparation followed by orbital rotation
        # into a single Slater determinant preparation gate
        for node in dag.named_nodes("slater_jw"):
            successors = list(dag.successors(node))
            if len(successors) == 1 and successors[0].op.name == "orb_rot_jw":
                successor_node = successors[0]
                combined_mat_a = (
                    successor_node.op.orbital_rotation_a @ node.op.orbital_rotation_a
                )
                combined_mat_b = (
                    successor_node.op.orbital_rotation_b @ node.op.orbital_rotation_b
                )
                # The merged gate retains the Slater determinant preparation's own
                # compression budget: max_givens/max_layers count Givens rotations in
                # the diamond-shaped Slater decomposition, a different scale from the
                # absorbed orbital rotation's brickwork decomposition, so the orbital
                # rotation's budget is discarded.
                tol = max(node.op.tol, successor_node.op.tol)
                max_givens = node.op.max_givens
                max_layers = node.op.max_layers
                dag.substitute_node(
                    node,
                    PrepareSlaterDeterminantJW(
                        node.op.norb,
                        node.op.occupied_orbitals,
                        orbital_rotation=(combined_mat_a, combined_mat_b),
                        tol=tol,
                        max_givens=max_givens,
                        max_layers=max_layers,
                        **node.op.optimize_kwargs,
                    ),
                    inplace=True,
                )
                dag.remove_op_node(successor_node)

        # Merge spinless orbital rotation gates
        for run in dag.collect_runs(["orb_rot_spinless_jw"]):
            node = run[0]
            qubits = node.qargs
            norb = node.op.norb
            combined_mat = np.eye(norb)
            tol = max(node.op.tol for node in run)
            max_givens = _min_or_none(node.op.max_givens for node in run)
            max_layers = _min_or_none(node.op.max_layers for node in run)
            optimize_kwargs = run[0].op.optimize_kwargs
            for node in run:
                combined_mat = node.op.orbital_rotation @ combined_mat
            dag.replace_block_with_op(
                run,
                OrbitalRotationSpinlessJW(
                    norb,
                    combined_mat,
                    tol=tol,
                    max_givens=max_givens,
                    max_layers=max_layers,
                    **optimize_kwargs,
                ),
                {q: i for i, q in enumerate(qubits)},
                cycle_check=False,
            )

        # Merge spinless Slater determinant preparation followed by spinless orbital
        # rotation into a single spinless Slater determinant preparation gate
        for node in dag.named_nodes("slater_spinless_jw"):
            successors = list(dag.successors(node))
            if len(successors) == 1 and successors[0].op.name == "orb_rot_spinless_jw":
                successor_node = successors[0]
                combined_mat = (
                    successor_node.op.orbital_rotation @ node.op.orbital_rotation
                )
                # See the spinful case above: the merged gate retains the Slater
                # determinant preparation's own compression budget and discards the
                # absorbed orbital rotation's differently-scaled budget.
                tol = max(node.op.tol, successor_node.op.tol)
                max_givens = node.op.max_givens
                max_layers = node.op.max_layers
                dag.substitute_node(
                    node,
                    PrepareSlaterDeterminantSpinlessJW(
                        node.op.norb,
                        node.op.occupied_orbitals,
                        orbital_rotation=combined_mat,
                        tol=tol,
                        max_givens=max_givens,
                        max_layers=max_layers,
                        **node.op.optimize_kwargs,
                    ),
                    inplace=True,
                )
                dag.remove_op_node(successor_node)

        return dag
