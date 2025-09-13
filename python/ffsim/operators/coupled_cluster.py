# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Coupled cluster operators."""

import itertools

import numpy as np

from ffsim._lib import FermionOperator
from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b


def coupled_cluster_singles_restricted(t1: np.ndarray) -> FermionOperator:
    """Restricted coupled cluster singles operator."""
    nocc, nvrt = t1.shape
    op = FermionOperator({})
    for i, a in itertools.product(range(nocc), range(nvrt)):
        coeff = t1[i, a]
        op += FermionOperator(
            {
                (cre_a(nocc + a), des_a(i)): coeff,
                (cre_b(nocc + a), des_b(i)): coeff,
            }
        )
    return op


def coupled_cluster_doubles_restricted(t2: np.ndarray) -> FermionOperator:
    """Restricted coupled cluster doubles operator."""
    nocc, _, nvrt, _ = t2.shape
    op = FermionOperator({})
    for i, j, a, b in itertools.product(
        range(nocc), range(nocc), range(nvrt), range(nvrt)
    ):
        coeff = t2[i, j, a, b]
        op += FermionOperator(
            {
                (cre_a(nocc + a), cre_a(nocc + b), des_a(j), des_a(i)): 0.5 * coeff,
                (cre_b(nocc + a), cre_b(nocc + b), des_b(j), des_b(i)): 0.5 * coeff,
                (cre_a(nocc + a), cre_b(nocc + b), des_b(j), des_a(i)): coeff,
            }
        )
    return op


def ccsd_generator_restricted(t1: np.ndarray, t2: np.ndarray) -> FermionOperator:
    """Restricted coupled cluster, singles and doubles generator."""
    return coupled_cluster_singles_restricted(t1) + coupled_cluster_doubles_restricted(
        t2
    )


def uccsd_generator_restricted(t1: np.ndarray, t2: np.ndarray) -> FermionOperator:
    """Restricted unitary coupled cluster, singles and doubles generator."""
    ccsd_gen = ccsd_generator_restricted(t1=t1, t2=t2)
    return ccsd_gen - ccsd_gen.adjoint()
