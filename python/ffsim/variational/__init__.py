# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Variational ansatzes."""

from ffsim.variational.givens import GivensAnsatzOp, GivensAnsatzOperator
from ffsim.variational.hopgate import HopGateAnsatzOperator
from ffsim.variational.multireference import (
    multireference_state,
    multireference_state_prod,
)
from ffsim.variational.num_num import NumNumAnsatzOpSpinBalanced
from ffsim.variational.ucj import RealUCJOperator, UCJOperator
from ffsim.variational.ucj_spin_balanced import UCJOpSpinBalanced
from ffsim.variational.ucj_spin_unbalanced import UCJOpSpinUnbalanced
from ffsim.variational.ucj_spinless import UCJOpSpinless

__all__ = [
    "GivensAnsatzOp",
    "GivensAnsatzOperator",
    "HopGateAnsatzOperator",
    "NumNumAnsatzOpSpinBalanced",
    "RealUCJOperator",
    "UCJOperator",
    "UCJOpSpinBalanced",
    "UCJOpSpinUnbalanced",
    "UCJOpSpinless",
    "multireference_state",
    "multireference_state_prod",
]
