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
from ffsim.spin import Spin


class MergeOrbitalRotations(TransformationPass):
    """Merge consecutive orbital rotation gates."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for gate_name, spin in [
            ("orb_rot_jw", Spin.ALPHA_AND_BETA),
            ("orb_rot_jw_a", Spin.ALPHA),
            ("orb_rot_jw_b", Spin.BETA),
        ]:
            for run in dag.collect_runs([gate_name]):
                node = run[0]
                qubits = node.qargs
                norb = len(qubits) // 2
                combined_mat = np.eye(norb)
                for node in run:
                    combined_mat = node.op.orbital_rotation @ combined_mat
                dag.replace_block_with_op(
                    run,
                    OrbitalRotationJW(combined_mat, spin=spin),
                    {q: i for i, q in enumerate(qubits)},
                    cycle_check=False,
                )
        return dag
