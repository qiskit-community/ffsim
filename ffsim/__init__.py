# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""ffsim is a software library for fast simulation of fermionic quantum circuits."""

from ffsim.double_factorized import (
    double_factorized_decomposition,
)

from ffsim.gates import (
    apply_diag_coulomb_evolution,
    apply_givens_rotation,
    apply_num_interaction,
    apply_num_op_prod_interaction,
    apply_num_op_sum_evolution,
    apply_orbital_rotation,
    apply_phase_shift,
    apply_tunneling_interaction,
)

from ffsim.states import (
    one_hot,
    slater_determinant,
)

from ffsim.trotter import (
    simulate_trotter_suzuki_double_factorized,
)
