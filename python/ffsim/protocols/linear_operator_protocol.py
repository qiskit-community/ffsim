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

from ffsim._lib import FermionOperator
from ffsim.states import dim, dims


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

    dim_ = dim(norb, nelec)
    dims_ = dims(norb, nelec)

    action_funcs = {
        # key: (action, spin)
        (False, False): pyscf.fci.addons.des_a,
        (False, True): pyscf.fci.addons.des_b,
        (True, False): pyscf.fci.addons.cre_a,
        (True, True): pyscf.fci.addons.cre_b,
    }

    def matvec(vec: np.ndarray):
        result = np.zeros(dim_, dtype=complex)
        vec_real = np.real(vec)
        vec_imag = np.imag(vec)
        for term in operator:
            coeff = operator[term]
            transformed_real = vec_real.reshape(dims_)
            transformed_imag = vec_imag.reshape(dims_)
            this_nelec = list(nelec)
            zero = False
            for action, spin, orb in reversed(term):
                action_func = action_funcs[(action, spin)]
                transformed_real = action_func(transformed_real, norb, this_nelec, orb)
                transformed_imag = action_func(transformed_imag, norb, this_nelec, orb)
                this_nelec[spin] += 1 if action else -1
                if this_nelec[spin] < 0 or this_nelec[spin] > norb:
                    zero = True
                    break
            if zero:
                continue
            result += coeff * transformed_real.reshape(-1)
            result += coeff * 1j * transformed_imag.reshape(-1)
        return result

    return LinearOperator(
        shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
    )
