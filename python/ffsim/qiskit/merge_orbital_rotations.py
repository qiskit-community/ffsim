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

import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from ffsim.qiskit.orbital_rotation import OrbitalRotationJW


class MergeOrbitalRotations(TransformationPass):
    """Merge consecutive orbital rotation gates."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for run in dag.collect_runs(["orb_rot_jw"]):
            node = run[0]
            qubits = node.qargs
            norb = node.op.norb
            combined_mat_a = np.eye(norb)
            combined_mat_b = np.eye(norb)
            for node in run:
                combined_mat_a = node.op.orbital_rotation_a @ combined_mat_a
                combined_mat_b = node.op.orbital_rotation_b @ combined_mat_b
            dag.replace_block_with_op(
                run,
                OrbitalRotationJW(norb, (combined_mat_a, combined_mat_b)),
                {q: i for i, q in enumerate(qubits)},
                cycle_check=False,
            )
        return dag
