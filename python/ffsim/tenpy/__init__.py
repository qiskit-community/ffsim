# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Code that uses TeNPy, e.g. for emulating quantum circuits."""

from ffsim.tenpy.gates.abstract_gates import (
    apply_single_site,
    apply_two_site,
)
from ffsim.tenpy.gates.basic_gates import (
    givens_rotation,
    num_interaction,
    num_num_interaction,
    on_site_interaction,
)
from ffsim.tenpy.gates.diag_coulomb import apply_diag_coulomb_evolution
from ffsim.tenpy.gates.orbital_rotation import apply_orbital_rotation
from ffsim.tenpy.gates.ucj import apply_ucj_op_spin_balanced
from ffsim.tenpy.hamiltonians.molecular_hamiltonian import MolecularHamiltonianMPOModel
from ffsim.tenpy.util import bitstring_to_mps

__all__ = [
    "apply_ucj_op_spin_balanced",
    "apply_diag_coulomb_evolution",
    "apply_orbital_rotation",
    "apply_single_site",
    "apply_two_site",
    "bitstring_to_mps",
    "givens_rotation",
    "MolecularHamiltonianMPOModel",
    "num_interaction",
    "num_num_interaction",
    "on_site_interaction",
]
