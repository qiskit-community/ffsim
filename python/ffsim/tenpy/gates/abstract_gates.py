# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TeNPy abstract gates."""

import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.linalg.charges import LegPipe
from tenpy.networks.site import SpinHalfFermionSite

# ignore lowercase argument and variable checks to maintain TeNPy naming conventions
# ruff: noqa: N803, N806

# define sites
shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
shfsc = LegPipe([shfs.leg, shfs.leg])


def apply_single_site(eng: TEBDEngine, U1: np.ndarray, site: int) -> None:
    r"""Apply a single-site gate to an MPS.

    Args:
        eng: The TEBD engine.
        U1: The single-site quantum gate.
        site: The gate will be applied to `site` on the MPS.

    Returns:
        None
    """
    U1_npc = npc.Array.from_ndarray(U1, [shfs.leg, shfs.leg.conj()], labels=["p", "p*"])
    psi = eng.get_resume_data()["psi"]
    psi.apply_local_op(site, U1_npc)


def apply_two_site(
    eng: TEBDEngine,
    U2: np.ndarray,
    sites: tuple[int, int],
    *,
    norm_tol: float = 1e-8,
) -> None:
    r"""Apply a two-site gate to an MPS.

    Args:
        eng: The TEBD engine.
        U2: The two-site quantum gate.
        sites: The gate will be applied to adjacent sites `(site1, site2)` on the MPS.
        norm_tol: The norm error above which we recanonicalize the MPS. In general, the
         application of a two-site gate to an MPS with truncation may degrade its
         canonical form. To mitigate this, we explicitly bring the MPS back into
         canonical form, if the Frobenius norm of the `site-resolved norm errors array <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS.norm_test>`_
         is greater than `norm_tol`.

    Returns:
        None
    """

    # check that sites are adjacent
    if abs(sites[0] - sites[1]) != 1:
        raise ValueError("sites must be adjacent")

    # check whether to transpose gate
    if sites[0] > sites[1]:
        U2 = U2.T

    # apply NN gate between (site1, site2)
    U2_npc = npc.Array.from_ndarray(
        U2, [shfsc, shfsc.conj()], labels=["(p0.p1)", "(p0*.p1*)"]
    )
    U2_npc_split = U2_npc.split_legs()
    eng.update_bond(max(sites), U2_npc_split)

    # recanonicalize psi if below error threshold
    psi = eng.get_resume_data()["psi"]
    if np.linalg.norm(psi.norm_test()) > norm_tol:
        psi.canonical_form_finite()
