# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Protocols."""

from ffsim.protocols.apply_unitary_protocol import SupportsApplyUnitary, apply_unitary
from ffsim.protocols.approximate_equality_protocol import (
    SupportsApproximateEquality,
    approx_eq,
)
from ffsim.protocols.diagonal_protocol import SupportsDiagonal, diag
from ffsim.protocols.fermion_operator_protocol import (
    SupportsFermionOperator,
    fermion_operator,
)
from ffsim.protocols.linear_operator_protocol import (
    SupportsLinearOperator,
    linear_operator,
)
from ffsim.protocols.trace_protocol import SupportsTrace, trace

__all__ = [
    "SupportsApplyUnitary",
    "SupportsApproximateEquality",
    "SupportsDiagonal",
    "SupportsFermionOperator",
    "SupportsLinearOperator",
    "SupportsTrace",
    "apply_unitary",
    "approx_eq",
    "diag",
    "fermion_operator",
    "linear_operator",
    "trace",
]
