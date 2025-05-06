# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""States."""

from ffsim.states.bitstring import (
    BitstringType,
    addresses_to_strings,
    strings_to_addresses,
)
from ffsim.states.product_state_sum import ProductStateSum
from ffsim.states.rdm import rdms
from ffsim.states.sample_slater import sample_slater_determinant
from ffsim.states.slater import (
    hartree_fock_state,
    slater_determinant,
    slater_determinant_amplitudes,
    slater_determinant_rdms,
)
from ffsim.states.states import (
    StateVector,
    dim,
    dims,
    sample_state_vector,
    spin_square,
    spinful_to_spinless_rdm1,
    spinful_to_spinless_rdm2,
    spinful_to_spinless_vec,
)
from ffsim.states.wick import expectation_one_body_power, expectation_one_body_product

__all__ = [
    "BitstringType",
    "ProductStateSum",
    "StateVector",
    "addresses_to_strings",
    "dim",
    "dims",
    "expectation_one_body_power",
    "expectation_one_body_product",
    "hartree_fock_state",
    "rdms",
    "sample_slater_determinant",
    "sample_state_vector",
    "slater_determinant",
    "slater_determinant_amplitudes",
    "slater_determinant_rdms",
    "spin_square",
    "spinful_to_spinless_rdm1",
    "spinful_to_spinless_rdm2",
    "spinful_to_spinless_vec",
    "strings_to_addresses",
]
