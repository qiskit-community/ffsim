# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Trace protocol."""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from ffsim.operators import FermionOperator


class SupportsTrace(Protocol):
    """A linear operator whose trace can be computed."""

    def _trace_(self, norb: int, nelec: tuple[int, int]) -> float:
        """Return the trace of the linear operator.

        Args:
            norb: The number of spatial orbitals.
            nelec: The number of alpha and beta electrons.

        Returns:
            The trace of the linear operator.
        """


def trace(obj: Any, norb: int, nelec: tuple[int, int]) -> complex:
    """Return the trace of the linear operator."""
    if isinstance(obj, FermionOperator):
        return _trace_fermion_operator(obj, norb, nelec)

    method = getattr(obj, "_trace_", None)
    if method is not None:
        return method(norb=norb, nelec=nelec)
    method = getattr(obj, "_diag_", None)
    if method is not None:
        return np.sum(method(norb=norb, nelec=nelec))
    raise TypeError(
        f"Could not compute trace of object of type {type(obj)}.\n"
        "The object did not have a _trace_ method that returned the trace "
        "or a _diag_ method that returned its diagonal entries."
    )


def _trace_term(
    op: tuple[tuple[bool, bool, int], ...], norb: int, nelec: tuple[int, int]
) -> complex:
    n_alpha, n_beta = nelec

    spin_orbs = set((spin, orb) for _, spin, orb in op)
    norb_alpha = sum(not spin for spin, _ in spin_orbs)
    norb_beta = len(spin_orbs) - norb_alpha

    nelec_alpha = 0
    nelec_beta = 0

    # loop over the support of the operator
    # assume that each site is either 0 or 1 at the beginning
    # track the state of the site through the application of the operator
    # if the state exceed 1 or goes below 0,
    # the state is not physical and the trace must be 0
    for this_spin, this_orb in spin_orbs:
        initial_zero = 0
        initial_one = 1
        is_zero = True
        is_one = True
        for action, spin, orb in reversed(op):
            if (spin, orb) != (this_spin, this_orb):
                continue

            change = action * 2 - 1
            initial_zero += change
            initial_one += change
            if initial_zero < 0 or initial_zero > 1:
                is_zero = False
            if initial_one < 0 or initial_one > 1:
                is_one = False

        # if the operator has support on this_orb,
        # either the initial state is 0 or 1, but not both
        assert not is_zero or not is_one

        # return 0 if there is no possible initial state
        if not is_zero and not is_one:
            return 0j
        # the state must return to the initial state, otherwise the trace is zero
        if (is_zero and initial_zero != 0) or (is_one and initial_one != 1):
            return 0j
        # count the number of electrons
        if is_one:
            if not this_spin:
                nelec_alpha += 1
            else:
                nelec_beta += 1
        if nelec_alpha > n_alpha or nelec_beta > n_beta:
            # the number of electrons exceeds the number of allowed electrons
            return 0j

    # the trace is nontrival and is a product of
    # binom(#orbs not in the support of op, #elec on these orbs)
    # for each spin species

    return math.comb(norb - norb_alpha, n_alpha - nelec_alpha) * math.comb(
        norb - norb_beta, n_beta - nelec_beta
    )


def _trace_fermion_operator(
    ferm: FermionOperator, norb: int, nelec: tuple[int, int]
) -> complex:
    return sum(coeff * _trace_term(op, norb, nelec) for op, coeff in ferm.items())
