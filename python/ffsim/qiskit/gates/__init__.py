# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit fermionic quantum gates."""

from ffsim.qiskit.gates.diag_coulomb import DiagCoulombEvolutionJW
from ffsim.qiskit.gates.givens_ansatz import (
    GivensAnsatzOperatorJW,
    GivensAnsatzOperatorSpinlessJW,
)
from ffsim.qiskit.gates.orbital_rotation import (
    OrbitalRotationJW,
    OrbitalRotationSpinlessJW,
)
from ffsim.qiskit.gates.slater_determinant import (
    PrepareHartreeFockJW,
    PrepareHartreeFockSpinlessJW,
    PrepareSlaterDeterminantJW,
    PrepareSlaterDeterminantSpinlessJW,
)
from ffsim.qiskit.gates.ucj import UCJOpSpinUnbalancedJW
from ffsim.qiskit.gates.ucj_operator import UCJOperatorJW

__all__ = [
    "DiagCoulombEvolutionJW",
    "GivensAnsatzOperatorJW",
    "GivensAnsatzOperatorSpinlessJW",
    "OrbitalRotationJW",
    "OrbitalRotationSpinlessJW",
    "PrepareHartreeFockJW",
    "PrepareHartreeFockSpinlessJW",
    "PrepareSlaterDeterminantJW",
    "PrepareSlaterDeterminantSpinlessJW",
    "UCJOperatorJW",
    "UCJOpSpinUnbalancedJW",
]
