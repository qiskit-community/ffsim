# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear operator protocol."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pyscf.fci
from scipy.sparse.linalg import LinearOperator

from ffsim import states
from ffsim.operators import FermionOperator


class SupportsLinearOperator(Protocol):
    """An object that can be converted to a SciPy LinearOperator."""

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object.

        Args:
            norb: The number of spatial orbitals.
            nelec: The number of alpha and beta electrons.

        Returns:
            A Scipy LinearOperator representing the object.
        """


def linear_operator(obj: Any, norb: int, nelec: tuple[int, int]) -> LinearOperator:
    """Return a SciPy LinearOperator representing the object.

    Args:
        obj: The object to convert to a LinearOperator.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        A Scipy LinearOperator representing the object.
    """
    if isinstance(obj, FermionOperator):
        return _fermion_operator_to_linear_operator(obj, norb=norb, nelec=nelec)

    method = getattr(obj, "_linear_operator_", None)
    if method is not None:
        return method(norb=norb, nelec=nelec)

    raise TypeError(f"Object of type {type(obj)} has no _linear_operator_ method.")


def _fermion_operator_to_linear_operator(
    operator: FermionOperator, norb: int, nelec: tuple[int, int]
):
    if not (operator.conserves_particle_number() and operator.conserves_spin_z()):
        raise ValueError(
            "The given FermionOperator could not be converted to a LinearOperator "
            "because it does not conserve particle number and the Z component of spin. "
            f"Conserves particle number: {operator.conserves_particle_number()} "
            f"Conserves spin Z: {operator.conserves_spin_z()}"
        )

    dim = states.dim(norb, nelec)

    def matvec(vec: np.ndarray):
        result = np.zeros(dim, dtype=complex)
        for term, coeff in operator.items():
            result += coeff * _apply_fermion_term(vec, term, norb, nelec)
        return result

    return LinearOperator(
        shape=(dim, dim), matvec=matvec, rmatvec=matvec, dtype=complex
    )


def _apply_fermion_term(
    vec: np.ndarray,
    term: tuple[tuple[bool, bool, int], ...],
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    """Apply a product of ladder operators to a state vector.

    Given a state vector and a string of ladder operators that conserves particle number
    and total spin Z, return the state vector that results from applying the ladder
    operators to the given state vector. The string of ladder operators is represented
    as a sequence of (`action`, `spin`, `orbital`) tuples, where:

    - `action` is a bool. False indicates a destruction operator and True indicates
      a creation operator.
    - `spin` is a bool. False indicates spin alpha and True indicates spin beta.
    - `orbital` is an integer giving the index of the spatial orbital to act on.

    The string of ladder operators acts on a state vector by left multiplication,
    so the resulting state vector is obtained by applying the ladder operators to the
    initial vector in the reverse order in which they are given.

    Args:
        vec: The state vector.
        term: The product of ladder operators to apply to the state vector.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.

    Returns:
        The result of applying the ladder operators to the state vector.
    """
    result = _apply_fermion_term_real(vec.real, term, norb, nelec)
    result += 1j * _apply_fermion_term_real(vec.imag, term, norb, nelec)
    return result


def _apply_fermion_term_real(
    vec: np.ndarray,
    term: tuple[tuple[bool, bool, int], ...],
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    action_funcs = {
        # key: (action, spin)
        (False, False): pyscf.fci.addons.des_a,
        (False, True): pyscf.fci.addons.des_b,
        (True, False): pyscf.fci.addons.cre_a,
        (True, True): pyscf.fci.addons.cre_b,
    }
    (dim_a, dim_b) = states.dims(norb, nelec)
    transformed = vec.reshape((dim_a, dim_b))
    this_nelec = list(nelec)
    for action, spin, orb in reversed(term):
        action_func = action_funcs[(action, spin)]
        transformed = action_func(transformed, norb, this_nelec, orb)
        this_nelec[spin] += 1 if action else -1
        if this_nelec[spin] < 0 or this_nelec[spin] > norb:
            return np.zeros(dim_a * dim_b, dtype=complex)
    return transformed.reshape(-1).astype(complex, copy=False)
