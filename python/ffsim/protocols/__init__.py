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

from ffsim.protocols.approximate_equality import SupportsApproximateEquality, approx_eq
from ffsim.protocols.linear_operator import SupportsLinearOperator, linear_operator
from ffsim.protocols.trace import SupportsTrace, trace
