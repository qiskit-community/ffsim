# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Hamiltonian simulation via Trotter-Suzuki formulas."""

from ffsim.trotter.qdrift import simulate_qdrift_double_factorized
from ffsim.trotter.trotter import simulate_trotter_double_factorized
