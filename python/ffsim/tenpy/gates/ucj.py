# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import numpy as np
from tenpy.algorithms.tebd import TEBDEngine

from ffsim.tenpy.gates.diag_coulomb import apply_diag_coulomb_evolution
from ffsim.tenpy.gates.orbital_rotation import apply_orbital_rotation
from ffsim.variational.ucj_spin_balanced import UCJOpSpinBalanced


def apply_ucj_op_spin_balanced(
    eng: TEBDEngine,
    ucj_op: UCJOpSpinBalanced,
    *,
    norm_tol: float = 1e-5,
) -> None:
    r"""Construct the LUCJ circuit as an MPS.

    Args:
        eng: The TEBD engine.
        ucj_op: The LUCJ operator.
        norm_tol: The norm error above which we recanonicalize the MPS.

    Returns:
        None
    """

    # extract norb
    norb = eng.get_resume_data()["psi"].L

    # construct the LUCJ MPS
    current_basis = np.eye(norb)
    for orb_rot, diag_mats in zip(ucj_op.orbital_rotations, ucj_op.diag_coulomb_mats):
        apply_orbital_rotation(
            eng,
            orb_rot.conjugate().T @ current_basis,
            norm_tol=norm_tol,
        )
        apply_diag_coulomb_evolution(eng, diag_mats, norm_tol=norm_tol)
        current_basis = orb_rot
    if ucj_op.final_orbital_rotation is None:
        apply_orbital_rotation(
            eng,
            current_basis,
            norm_tol=norm_tol,
        )
    else:
        apply_orbital_rotation(
            eng,
            ucj_op.final_orbital_rotation @ current_basis,
            norm_tol=norm_tol,
        )
