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

from ffsim import contract, linalg, random, testing
from ffsim.cistring import init_cache
from ffsim.gates import (
    apply_diag_coulomb_evolution,
    apply_fsim_gate,
    apply_givens_rotation,
    apply_hop_gate,
    apply_num_interaction,
    apply_num_num_interaction,
    apply_num_op_prod_interaction,
    apply_num_op_sum_evolution,
    apply_orbital_rotation,
    apply_tunneling_interaction,
)
from ffsim.hamiltonians import (
    DoubleFactorizedHamiltonian,
    MolecularHamiltonian,
    SingleFactorizedHamiltonian,
)
from ffsim.molecular_data import MolecularData
from ffsim.operators import (
    FermionAction,
    FermionOperator,
    cre,
    cre_a,
    cre_b,
    des,
    des_a,
    des_b,
)
from ffsim.protocols import (
    SupportsApplyUnitary,
    SupportsApproximateEquality,
    SupportsFermionOperator,
    SupportsLinearOperator,
    SupportsTrace,
    apply_unitary,
    approx_eq,
    fermion_operator,
    linear_operator,
    trace,
)
from ffsim.states import (
    dim,
    dims,
    expectation_one_body_power,
    expectation_one_body_product,
    hartree_fock_state,
    indices_to_strings,
    one_hot,
    rdm,
    slater_determinant,
    slater_determinant_rdm,
)
from ffsim.trotter import (
    simulate_qdrift_double_factorized,
    simulate_trotter_double_factorized,
)
from ffsim.variational import (
    HopGateAnsatzOperator,
    RealUCJOperator,
    UCJOperator,
    multireference_state,
)

__all__ = [
    "DoubleFactorizedHamiltonian",
    "FermionAction",
    "FermionOperator",
    "HopGateAnsatzOperator",
    "MolecularData",
    "MolecularHamiltonian",
    "RealUCJOperator",
    "SingleFactorizedHamiltonian",
    "SupportsApplyUnitary",
    "SupportsApproximateEquality",
    "SupportsFermionOperator",
    "SupportsLinearOperator",
    "SupportsTrace",
    "UCJOperator",
    "apply_diag_coulomb_evolution",
    "apply_fsim_gate",
    "apply_givens_rotation",
    "apply_hop_gate",
    "apply_num_interaction",
    "apply_num_num_interaction",
    "apply_num_op_prod_interaction",
    "apply_num_op_sum_evolution",
    "apply_orbital_rotation",
    "apply_tunneling_interaction",
    "apply_unitary",
    "approx_eq",
    "contract",
    "cre",
    "cre_a",
    "cre_b",
    "des",
    "des_a",
    "des_b",
    "dim",
    "dims",
    "expectation_one_body_power",
    "expectation_one_body_product",
    "fermion_operator",
    "hartree_fock_state",
    "indices_to_strings",
    "init_cache",
    "linalg",
    "linear_operator",
    "multireference_state",
    "one_hot",
    "random",
    "rdm",
    "simulate_qdrift_double_factorized",
    "simulate_trotter_double_factorized",
    "slater_determinant",
    "slater_determinant_rdm",
    "testing",
    "trace",
]
