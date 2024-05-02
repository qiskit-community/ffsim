# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Code that uses Qiskit, e.g. for constructing quantum circuits."""

from ffsim.qiskit.gates import (
    DiagCoulombEvolutionJW,
    OrbitalRotationJW,
    PrepareHartreeFockJW,
    PrepareSlaterDeterminantJW,
    UCJOperatorJW,
)
from ffsim.qiskit.transpiler_passes import DropNegligible, MergeOrbitalRotations
from ffsim.qiskit.transpiler_stages import pre_init_passes
from ffsim.qiskit.util import ffsim_vec_to_qiskit_vec, qiskit_vec_to_ffsim_vec

__all__ = [
    "DiagCoulombEvolutionJW",
    "DropNegligible",
    "MergeOrbitalRotations",
    "OrbitalRotationJW",
    "PrepareHartreeFockJW",
    "PrepareSlaterDeterminantJW",
    "UCJOperatorJW",
    "ffsim_vec_to_qiskit_vec",
    "pre_init_passes",
    "qiskit_vec_to_ffsim_vec",
]
