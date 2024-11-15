# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import itertools

import numpy as np
from tenpy.algorithms.tebd import TEBDEngine

from ffsim.spin import Spin
from ffsim.tenpy.gates.abstract_gates import apply_gate1, apply_gate2
from ffsim.tenpy.gates.basic_gates import num_num_interaction, on_site_interaction


def apply_diag_coulomb_evolution(
    eng: TEBDEngine,
    mat: np.ndarray,
    *,
    norm_tol: float = 1e-5,
) -> None:
    r"""Apply a diagonal Coulomb evolution gate to an MPS.

    The diagonal Coulomb evolution gate is defined in
    `apply_diag_coulomb_evolution <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_diag_coulomb_evolution>`__.

    Args:
        eng: The TEBD Engine.
        mat: The diagonal Coulomb matrices of dimension `(2, norb, norb)`.
        norm_tol: The norm error above which we recanonicalize the MPS.

    Returns:
        None
    """

    # extract norb
    norb = eng.get_resume_data()["psi"].L

    # unpack alpha-alpha and alpha-beta matrices
    mat_aa, mat_ab = mat

    # apply alpha-alpha gates
    for i, j in itertools.product(range(norb), repeat=2):
        if j > i and mat_aa[i, j]:
            apply_gate2(
                eng,
                num_num_interaction(-mat_aa[i, j], Spin.ALPHA_AND_BETA),
                (i, j),
                norm_tol=norm_tol,
            )

    # apply alpha-beta gates
    for i in range(norb):
        apply_gate1(eng, on_site_interaction(-mat_ab[i, i]), i)
