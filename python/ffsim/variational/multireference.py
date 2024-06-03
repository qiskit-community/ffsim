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

from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from ffsim.hamiltonians.single_factorized_hamiltonian import SingleFactorizedHamiltonian
from ffsim.linalg.linalg import reduced_matrix
from ffsim.protocols.apply_unitary_protocol import SupportsApplyUnitary, apply_unitary
from ffsim.protocols.linear_operator_protocol import (
    SupportsLinearOperator,
    linear_operator,
)
from ffsim.states import ProductStateSum, slater_determinant


def multireference_state(
    hamiltonian: LinearOperator | SupportsLinearOperator,
    ansatz_operator: SupportsApplyUnitary,
    reference_occupations: Sequence[tuple[Sequence[int], Sequence[int]]],
    norb: int,
    nelec: tuple[int, int],
    root: int = 0,  # use lowest eigenvector by default
) -> tuple[float, np.ndarray]:
    """Compute multireference energy and state.

    Args:
        hamiltonian: The Hamiltonian.
        ansatz_operator: The ansatz operator.
        reference_occupations: The orbital occupations of the reference states.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        root: The index of the desired eigenvector. Defaults to 0, which yields the
            lowest-energy state.

    Returns:
        The energy of the multireference state, and the state itself.
    """
    if not isinstance(hamiltonian, LinearOperator):
        hamiltonian = linear_operator(hamiltonian, norb=norb, nelec=nelec)
    reference_configurations = [
        slater_determinant(norb, occ) for occ in reference_occupations
    ]
    basis_states = [
        apply_unitary(vec, ansatz_operator, norb=norb, nelec=nelec, copy=False)
        for vec in reference_configurations
    ]
    mat = reduced_matrix(hamiltonian, basis_states)
    _, vecs = scipy.linalg.eigh(mat)
    coeffs = vecs[:, root]
    energy = np.real(np.sum(np.outer(coeffs, coeffs) * mat))
    return energy, np.tensordot(coeffs, basis_states, axes=1)


def multireference_state_prod(
    hamiltonian: MolecularHamiltonian | SingleFactorizedHamiltonian,
    ansatz_operator: tuple[SupportsApplyUnitary, SupportsApplyUnitary],
    reference_occupations: Sequence[tuple[Sequence[int], Sequence[int]]],
    norb: int,
    nelec: tuple[int, int],
    root: int = 0,  # use lowest eigenvector by default
    tol: float = 1e-8,
) -> tuple[float, ProductStateSum]:
    """Compute multireference state for a product ansatz operator.

    Args:
        hamiltonian: The Hamiltonian.
        ansatz_operator: The alpha and beta parts of the ansatz operator.
        reference_occupations: The orbital occupations of the reference states.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        root: The index of the desired eigenvector. Defaults to 0, which yields the
            lowest-energy state.
        tol: Numerical tolerance to use for the single factorization of the molecular
            Hamiltonian. If the input is already a SingleFactorizedHamiltonian,
            this argument is ignored.

    Returns:
        The energy of the multireference state, and the state itself.
    """
    if isinstance(hamiltonian, MolecularHamiltonian):
        sf_hamiltonian = SingleFactorizedHamiltonian.from_molecular_hamiltonian(
            hamiltonian, tol=tol
        )
    else:
        sf_hamiltonian = hamiltonian

    n_alpha, n_beta = nelec
    ansatz_operator_a, ansatz_operator_b = ansatz_operator
    reference_configurations = [
        (slater_determinant(norb, occ_a), slater_determinant(norb, occ_b))
        for occ_a, occ_b in reference_occupations
    ]
    basis_states = [
        (
            apply_unitary(
                vec_a, ansatz_operator_a, norb=norb, nelec=n_alpha, copy=False
            ),
            apply_unitary(
                vec_b, ansatz_operator_b, norb=norb, nelec=n_beta, copy=False
            ),
        )
        for vec_a, vec_b in reference_configurations
    ]
    mat = sf_hamiltonian.reduced_matrix_product_states(basis_states, norb, nelec)
    _, vecs = scipy.linalg.eigh(mat)
    coeffs = vecs[:, root]
    energy = np.real(np.sum(np.outer(coeffs, coeffs) * mat))
    return energy, ProductStateSum(coeffs=coeffs, states=basis_states)
