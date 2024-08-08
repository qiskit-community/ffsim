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

from qiskit.transpiler import PassManager

from ffsim.qiskit.gates import (
    DiagCoulombEvolutionJW,
    DiagCoulombEvolutionSpinlessJW,
    GivensAnsatzOperatorJW,
    GivensAnsatzOperatorSpinlessJW,
    GivensAnsatzOpJW,
    GivensAnsatzOpSpinlessJW,
    NumNumAnsatzOpSpinBalancedJW,
    NumOpSumEvolutionJW,
    NumOpSumEvolutionSpinlessJW,
    OrbitalRotationJW,
    OrbitalRotationSpinlessJW,
    PrepareHartreeFockJW,
    PrepareHartreeFockSpinlessJW,
    PrepareSlaterDeterminantJW,
    PrepareSlaterDeterminantSpinlessJW,
    SimulateTrotterDiagCoulombSplitOpJW,
    SimulateTrotterDoubleFactorizedJW,
    UCJOperatorJW,
    UCJOpSpinBalancedJW,
    UCJOpSpinlessJW,
    UCJOpSpinUnbalancedJW,
)
from ffsim.qiskit.jordan_wigner import jordan_wigner
from ffsim.qiskit.sampler import FfsimSampler
from ffsim.qiskit.sim import final_state_vector
from ffsim.qiskit.transpiler_passes import DropNegligible, MergeOrbitalRotations
from ffsim.qiskit.transpiler_stages import pre_init_passes
from ffsim.qiskit.util import ffsim_vec_to_qiskit_vec, qiskit_vec_to_ffsim_vec

PRE_INIT = PassManager(list(pre_init_passes()))
"""Pass manager recommended for the Qiskit transpiler ``pre_init`` stage.

See :func:`pre_init_passes` for a description of the transpiler passes included in this
pass manager.
"""


__all__ = [
    "DiagCoulombEvolutionJW",
    "DiagCoulombEvolutionSpinlessJW",
    "DropNegligible",
    "FfsimSampler",
    "GivensAnsatzOpJW",
    "GivensAnsatzOpSpinlessJW",
    "GivensAnsatzOperatorJW",
    "GivensAnsatzOperatorSpinlessJW",
    "MergeOrbitalRotations",
    "NumNumAnsatzOpSpinBalancedJW",
    "NumOpSumEvolutionJW",
    "NumOpSumEvolutionSpinlessJW",
    "OrbitalRotationJW",
    "OrbitalRotationSpinlessJW",
    "PRE_INIT",
    "PrepareHartreeFockJW",
    "PrepareHartreeFockSpinlessJW",
    "PrepareSlaterDeterminantJW",
    "PrepareSlaterDeterminantSpinlessJW",
    "SimulateTrotterDiagCoulombSplitOpJW",
    "SimulateTrotterDoubleFactorizedJW",
    "UCJOperatorJW",
    "UCJOpSpinBalancedJW",
    "UCJOpSpinUnbalancedJW",
    "UCJOpSpinlessJW",
    "ffsim_vec_to_qiskit_vec",
    "final_state_vector",
    "jordan_wigner",
    "pre_init_passes",
    "qiskit_vec_to_ffsim_vec",
]
