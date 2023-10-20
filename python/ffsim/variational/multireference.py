# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tools for multireference calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

from ffsim.linalg.linalg import reduced_matrix
from ffsim.protocols.apply_unitary import SupportsApplyUnitary, apply_unitary
from ffsim.protocols.linear_operator import SupportsLinearOperator, linear_operator
from ffsim.states import slater_determinant


def multireference_state(
    hamiltonian: LinearOperator | SupportsLinearOperator,
    ansatz_operator: SupportsApplyUnitary,
    reference_occupations: Sequence[tuple[Sequence[int], Sequence[int]]],
    norb: int,
    nelec: tuple[int, int],
    root: int = 0,  # use lowest eigenvector by default
) -> np.ndarray:
    """Compute multireference state.

    Args:
        hamiltonian: The Hamiltonian.
        ansatz_operator: The ansatz operator.
        reference_occupations: The orbital occupations of the reference states.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        root: The index of the desired eigenvector. Defaults to 0, which yields the
            lowest-energy state.
    """
    if not isinstance(hamiltonian, LinearOperator):
        hamiltonian = linear_operator(hamiltonian, norb=norb, nelec=nelec)
    basis_states = [
        apply_unitary(
            slater_determinant(norb=norb, occupied_orbitals=occ),
            ansatz_operator,
            norb=norb,
            nelec=nelec,
        )
        for occ in reference_occupations
    ]
    mat = reduced_matrix(hamiltonian, basis_states)
    _, vecs = scipy.linalg.eigh(mat)
    return np.tensordot(vecs[:, root], basis_states, axes=1)
