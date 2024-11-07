from __future__ import annotations

import numpy as np
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.networks.mps import MPS

from ffsim.tenpy.circuits.gates import (
    apply_diag_coulomb_evolution,
    apply_orbital_rotation,
)
from ffsim.tenpy.util import product_state_as_mps
from ffsim.variational.ucj_spin_balanced import UCJOpSpinBalanced


def lucj_circuit_as_mps(
    norb: int,
    nelec: int | tuple[int, int],
    ucj_op: UCJOpSpinBalanced,
    options: dict,
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
    psi = product_state_as_mps(norb, nelec, 0)

    # define the TEBD engine
    eng = TEBDEngine(psi, None, options)

    # construct the LUCJ MPS
    n_reps = np.shape(ucj_op.orbital_rotations)[0]
    for i in range(n_reps):
        apply_orbital_rotation(
            np.conj(ucj_op.orbital_rotations[i]).T, psi, eng, chi_list, norm_tol
        )
        apply_diag_coulomb_evolution(
            ucj_op.diag_coulomb_mats[i], psi, eng, chi_list, norm_tol
        )
        apply_orbital_rotation(
            ucj_op.orbital_rotations[i], psi, eng, chi_list, norm_tol
        )
        if ucj_op.final_orbital_rotation is not None:
            apply_orbital_rotation(
                ucj_op.final_orbital_rotation, psi, eng, chi_list, norm_tol
            )

    return psi, chi_list
