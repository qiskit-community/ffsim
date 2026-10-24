# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit gates that do not conserve fermionic symmetries.

The gates in this module act on ancilla qubits, or on qubits that do not represent
fermionic modes, so they do not preserve the numbers of spin alpha and spin beta
fermions. As a result, they cannot be simulated with
:func:`ffsim.qiskit.final_state_vector`. They are not exported from
``ffsim.qiskit`` and must be accessed from this module.
"""

from ffsim.qiskit.nonconserving.paldus_transform import PaldusTransformJW
from ffsim.qiskit.nonconserving.schur_transform import SchurTransform

__all__ = [
    "PaldusTransformJW",
    "SchurTransform",
]
