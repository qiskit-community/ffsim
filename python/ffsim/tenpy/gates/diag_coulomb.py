# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TeNPy diagonal Coulomb evolution gate."""

from __future__ import annotations

import itertools

import numpy as np
from tenpy.algorithms.tebd import TEBDEngine

from ffsim.tenpy.gates.abstract_gates import apply_single_site, apply_two_site
from ffsim.tenpy.gates.basic_gates import num_num_interaction, on_site_interaction


def apply_diag_coulomb_evolution(
    eng: TEBDEngine,
    mat: np.ndarray,
    time: float,
    *,
    norm_tol: float = 1e-8,
) -> None:
    r"""Apply a diagonal Coulomb evolution gate to an MPS.

    The diagonal Coulomb evolution gate is defined in
    :func:`~ffsim.apply_diag_coulomb_evolution`.

    Args:
        eng: The TEBD engine.
        mat: The diagonal Coulomb matrices of dimension `(2, norb, norb)`.
        time: The evolution time.
        norm_tol: The norm error above which we recanonicalize the MPS. In general, the
         application of a two-site gate to an MPS with truncation may degrade its
         canonical form. To mitigate this, we explicitly bring the MPS back into
         canonical form, if the Frobenius norm of the `site-resolved norm errors array <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS.norm_test>`_
         is greater than `norm_tol`.

    Returns:
        None
    """

    # extract norb
    norb = eng.get_resume_data()["psi"].L

    # unpack alpha-alpha and alpha-beta matrices
    mat_aa, mat_ab = mat

    # apply alpha-alpha gates
    for i, j in itertools.combinations(range(norb), 2):
        if mat_aa[i, j]:
            apply_two_site(
                eng,
                num_num_interaction(-time * mat_aa[i, j]),
                (i, j),
                norm_tol=norm_tol,
            )

    # apply alpha-beta gates
    for i in range(norb):
        apply_single_site(eng, on_site_interaction(-time * mat_ab[i, i]), i)
