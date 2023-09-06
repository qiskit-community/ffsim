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

from ffsim import (
    contract,
    gates,
    hamiltonians,
    linalg,
    protocols,
    random,
    states,
    testing,
    trotter,
    variational,
)
from ffsim.contract import (
    contract_diag_coulomb,
    contract_num_op_sum,
    diag_coulomb_linop,
    hamiltonian_linop,
    hamiltonian_trace,
    num_op_sum_linop,
)
from ffsim.gates import (
    apply_diag_coulomb_evolution,
    apply_givens_rotation,
    apply_num_interaction,
    apply_num_op_prod_interaction,
    apply_num_op_sum_evolution,
    apply_orbital_rotation,
    apply_tunneling_interaction,
)
from ffsim.hamiltonians import (
    DoubleFactorizedHamiltonian,
    MolecularHamiltonian,
    double_factorized_hamiltonian,
)
from ffsim.molecular_data import MolecularData
from ffsim.protocols import (
    SupportsLinearOperator,
    SupportsTrace,
    linear_operator,
    trace,
)
from ffsim.states import (
    dim,
    dims,
    one_hot,
    slater_determinant,
    slater_determinant_one_rdm,
)
from ffsim.trotter import (
    simulate_qdrift_double_factorized,
    simulate_trotter_double_factorized,
)
from ffsim.variational import UCJOperator, apply_ucj_operator
