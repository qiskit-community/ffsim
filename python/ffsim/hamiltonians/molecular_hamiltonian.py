# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
from opt_einsum import contract
from pyscf.fci.direct_nosym import absorb_h1e, contract_1e, contract_2e, make_hdiag
from scipy.sparse.linalg import LinearOperator

from ffsim._lib import FermionOperator
from ffsim.cistring import gen_linkstr_index
from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b
from ffsim.states import dim


@dataclasses.dataclass(frozen=True)
class MolecularHamiltonian:
    r"""A molecular Hamiltonian.

    A Hamiltonian of the form

    .. math::

        H = \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q}
            + \frac12 \sum_{\sigma \tau, pqrs} h_{pqrs}
            a^\dagger_{\sigma, p} a^\dagger_{\tau, r} a_{\tau, s} a_{\sigma, q}
            + \text{constant}.

    Here :math:`h_{pq}` is called the one-body tensor and :math:`h_{pqrs}` is called
    the two-body tensor.

    Attributes:
        one_body_tensor (np.ndarray): The one-body tensor.
        two_body_tensor (np.ndarray): The two-body tensor.
        constant (float): The constant.
    """

    one_body_tensor: np.ndarray
    two_body_tensor: np.ndarray
    constant: float = 0.0

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    def rotated(self, orbital_rotation: np.ndarray) -> MolecularHamiltonian:
        """Return the Hamiltonian in a rotated orbital basis.

        Given an orbital rotation :math:`\mathcal{U}`, returns the operator

        .. math::

            \mathcal{U} H \mathcal{U}^\dagger

        where :math:`H` is the original Hamiltonian.

        Args:
            orbital_rotation: The orbital rotation.

        Returns:
            The rotated Hamiltonian.
        """
        one_body_tensor_rotated = contract(
            "ab,Aa,Bb->AB",
            self.one_body_tensor,
            orbital_rotation,
            orbital_rotation.conj(),
            optimize="greedy",
        )
        two_body_tensor_rotated = contract(
            "abcd,Aa,Bb,Cc,Dd->ABCD",
            self.two_body_tensor,
            orbital_rotation,
            orbital_rotation.conj(),
            orbital_rotation,
            orbital_rotation.conj(),
            optimize="greedy",
        )
        return MolecularHamiltonian(
            one_body_tensor=one_body_tensor_rotated,
            two_body_tensor=two_body_tensor_rotated,
            constant=self.constant,
        )

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        n_alpha, n_beta = nelec
        linkstr_index_a = gen_linkstr_index(range(norb), n_alpha)
        linkstr_index_b = gen_linkstr_index(range(norb), n_beta)
        link_index = (linkstr_index_a, linkstr_index_b)
        two_body = absorb_h1e(
            self.one_body_tensor.real, self.two_body_tensor, norb, nelec, 0.5
        )
        dim_ = dim(norb, nelec)

        def matvec(vec: np.ndarray):
            result = self.constant * vec.astype(complex, copy=False)
            result += 1j * contract_1e(
                self.one_body_tensor.imag, vec.real, norb, nelec, link_index=link_index
            )
            result -= contract_1e(
                self.one_body_tensor.imag, vec.imag, norb, nelec, link_index=link_index
            )
            result += contract_2e(
                two_body, vec.real, norb, nelec, link_index=link_index
            )
            result += 1j * contract_2e(
                two_body, vec.imag, norb, nelec, link_index=link_index
            )
            return result

        return LinearOperator(
            shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
        )

    def _trace_(self, norb: int, nelec: tuple[int, int]) -> float:
        """Return the trace of the object."""
        return self.constant * dim(norb, nelec) + np.sum(
            make_hdiag(self.one_body_tensor, self.two_body_tensor, norb, nelec)
        )

    def _fermion_operator_(self) -> FermionOperator:
        """Return a FermionOperator representing the object."""
        op = FermionOperator({(): self.constant})
        for p, q in itertools.product(range(self.norb), repeat=2):
            coeff = self.one_body_tensor[p, q]
            op += FermionOperator(
                {
                    (cre_a(p), des_a(q)): coeff,
                    (cre_b(p), des_b(q)): coeff,
                }
            )
        for p, q, r, s in itertools.product(range(self.norb), repeat=4):
            coeff = 0.5 * self.two_body_tensor[p, q, r, s]
            op += FermionOperator(
                {
                    (cre_a(p), cre_a(r), des_a(s), des_a(q)): coeff,
                    (cre_a(p), cre_b(r), des_b(s), des_a(q)): coeff,
                    (cre_b(p), cre_a(r), des_a(s), des_b(q)): coeff,
                    (cre_b(p), cre_b(r), des_b(s), des_b(q)): coeff,
                }
            )
        return op
