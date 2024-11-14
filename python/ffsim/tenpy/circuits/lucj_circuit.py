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
from tenpy.networks.mps import MPS

import ffsim
from ffsim.tenpy.circuits.gates import (
    apply_diag_coulomb_evolution,
    apply_orbital_rotation,
)
from ffsim.tenpy.util import product_state_as_mps
from ffsim.variational.ucj_spin_balanced import UCJOpSpinBalanced


def apply_ucj_op_spin_balanced(
    ucj_op: UCJOpSpinBalanced,
    norb: int,
    nelec: int | tuple[int, int],
    options: dict,
    *,
    norm_tol: float = 1e-5,
) -> tuple[MPS, list[int]]:
    r"""Construct the LUCJ circuit as an MPS.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        ucj_op: The LUCJ operator.
        options: The options parsed by the
            `TeNPy TEBDEngine <https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.tebd.TEBDEngine.html#tenpy.algorithms.tebd.TEBDEngine>`__.
        norm_tol: The norm error above which we recanonicalize the wavefunction, as
            defined in the
            `TeNPy documentation <https://tenpy.readthedocs.io/en/latest/reference/tenpy.algorithms.dmrg.DMRGEngine.html#cfg-option-DMRGEngine.norm_tol>`__.

    Returns:
        `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__
            LUCJ circuit as an MPS.

        list[int]
            Complete list of MPS bond dimensions compiled during circuit evaluation.
    """

    # initialize chi_list
    chi_list: list[int] = []

    # prepare initial Hartree-Fock state
    dim = ffsim.dim(norb, nelec)
    strings_a, strings_b = ffsim.addresses_to_strings(
        range(dim),
        norb=norb,
        nelec=nelec,
        bitstring_type=ffsim.BitstringType.STRING,
        concatenate=False,
    )
    psi = product_state_as_mps((strings_a[0], strings_b[0]))

    # define the TEBD engine
    eng = TEBDEngine(psi, None, options)

    # construct the LUCJ MPS
    current_basis = np.eye(norb)
    for orb_rot, diag_mats in zip(ucj_op.orbital_rotations, ucj_op.diag_coulomb_mats):
        apply_orbital_rotation(
            psi,
            orb_rot.conjugate().T @ current_basis,
            eng=eng,
            chi_list=chi_list,
            norm_tol=norm_tol,
        )
        apply_diag_coulomb_evolution(
            psi, diag_mats, eng=eng, chi_list=chi_list, norm_tol=norm_tol
        )
        current_basis = orb_rot
    if ucj_op.final_orbital_rotation is None:
        apply_orbital_rotation(
            psi,
            current_basis,
            eng=eng,
            chi_list=chi_list,
            norm_tol=norm_tol,
        )
    else:
        apply_orbital_rotation(
            psi,
            ucj_op.final_orbital_rotation @ current_basis,
            eng=eng,
            chi_list=chi_list,
            norm_tol=norm_tol,
        )

    return psi, chi_list
