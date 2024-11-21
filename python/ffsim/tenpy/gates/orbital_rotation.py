# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import cmath
import math

import numpy as np
from tenpy.algorithms.tebd import TEBDEngine

from ffsim.linalg import givens_decomposition
from ffsim.tenpy.gates.abstract_gates import apply_single_site, apply_two_site
from ffsim.tenpy.gates.basic_gates import givens_rotation, num_interaction


def apply_orbital_rotation(
    eng: TEBDEngine,
    mat: np.ndarray,
    *,
    norm_tol: float = 1e-5,
) -> None:
    r"""Apply an orbital rotation gate to an MPS.

    The orbital rotation gate is defined in
    `apply_orbital_rotation <https://qiskit-community.github.io/ffsim/api/ffsim.html#ffsim.apply_orbital_rotation>`__.

    Args:
        eng: The TEBD Engine.
        mat: The orbital rotation matrix of dimension `(norb, norb)`.
        norm_tol: The norm error above which we recanonicalize the MPS.

    Returns:
        None
    """

    # Givens decomposition
    givens_list, diag_mat = givens_decomposition(mat)

    # apply the Givens rotation gates
    for gate in givens_list:
        theta = math.acos(gate.c)
        phi = cmath.phase(gate.s) - np.pi
        apply_two_site(
            eng,
            givens_rotation(theta, phi=phi),
            (gate.i, gate.j),
            norm_tol=norm_tol,
        )

    # apply the number interaction gates
    for i, z in enumerate(diag_mat):
        theta = cmath.phase(z)
        apply_single_site(eng, num_interaction(-theta), i)
