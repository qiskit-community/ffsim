# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tools for constructing Qiskit transpiler pass managers and stages."""

from __future__ import annotations

from collections.abc import Iterator

from qiskit.transpiler.basepasses import BasePass
from qiskit.transpiler.passes import Decompose

from ffsim.qiskit.transpiler_passes import MergeOrbitalRotations


def pre_init_passes() -> Iterator[BasePass]:
    """Yield transpiler passes recommended for the Qiskit transpiler ``pre_init`` stage.

    The following transpiler passes are yielded:

    - `Decompose`_ pass that decomposes :class:`PrepareHartreeFockJW` and
      :class:`UCJOperatorJW` gates to expose the underlying
      :class:`PrepareSlaterDeterminantJW` and :class:`OrbitalRotationJW` gates.
    - :class:`MergeOrbitalRotations` pass to merge the Slater determinant preparation
      and orbital rotation gates.

    .. _Decompose: https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.Decompose#decompose
    """
    yield Decompose(
        [
            "hartree_fock_jw",
            "hartree_fock_spinless_jw",
            "ucj_jw",
            "ucj_balanced_jw",
            "ucj_unbalanced_jw",
        ]
    )
    yield MergeOrbitalRotations()
