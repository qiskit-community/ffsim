# (C) Copyright IBM 2023.
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

from typing import Any, Protocol

from ffsim.operators import FermionOperator

import numpy as np

import math


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


def trace(obj: Any, norb: int, nelec: tuple[int, int]) -> float:
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

def _trace_fermion_operator_single(
    A: FermionOperator, norb: int, nelec: tuple[int, int]
):
    n_elec_alpha, n_elec_beta = nelec
    op = list(A.keys())[0]
    coeff = A[op]
    alpha_indices = []
    beta_indices = []
    for _, spin, orb in op:
        if not spin:
            alpha_indices.append(orb)
        else:
            beta_indices.append(orb)
    alpha_indices = list(set(alpha_indices))
    beta_indices = list(set(beta_indices))
    combined_indices = [(i,False) for i in alpha_indices]  +[(i,True) for i in beta_indices]
    
    
    n_alpha = len(alpha_indices)
    n_beta = len(beta_indices)

    no_configuration = False
    eta_alpha = 0
    eta_beta = 0

    # loop over the support of the operator
    # assume that each site is either 0 or 1 at the beginning
    # track the state of the site through the application of the operator
    # if the state exceed 1 or goes below 0, the state is not physical and the trace must be 0
    for i, spin in combined_indices:
        initial_zero = 0
        initial_one = 1
        is_zero = True
        is_one = True
        for action, aspin, orb in reversed(op):  
            if aspin == spin and orb == i:
                change = a[0]*2-1
                initial_zero += change
                initial_one += change
                if initial_zero < 0 or initial_zero > 1:
                    is_zero = False
                if initial_one < 0 or initial_one > 1:
                    is_one = False
            if not is_zero and not is_one:
                no_configuration = True
                break
        if (is_zero and initial_zero != 0) or (is_one and initial_one != 1):
            no_configuration = True
        assert not is_zero or not is_one # only one case is possible if the operator has support on site i
        # count the number of electrons
        if is_one:
            if spin == False:
                eta_alpha += 1
            else:
                eta_beta += 1
        # the number of electrons exceed the number of allowed electrons
        if eta_alpha > n_elec_alpha or eta_beta > n_elec_beta: 
            no_configuration = True
        if no_configuration:
            trace = 0
            break
            
    if not no_configuration:
        # the trace is nontrival and is a product of
        # coeff, and
        # binom(number of orbitals not in the support of A, number of electrons allowed on these orbitals) for both spin species
        trace = coeff*math.comb(norb - len(alpha_indices),n_elec_alpha - eta_alpha)*math.comb(norb - len(beta_indices),n_elec_beta - eta_beta)
        
    return trace

def _trace_fermion_operator(
    A: FermionOperator, norb: int, nelec: tuple[int, int]
):
    n_elec_alpha, n_elec_beta = nelec
    for op in A:
        B = FermionOperator({op:A[op]})
        trace+= _trace_fermion_operator_single(B, norb, nelec)
    
            
    return trace
