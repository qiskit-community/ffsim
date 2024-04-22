# (C) Copyright IBM 2024.
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
class DiagonalCoulombHamiltonian:
    r"""A diagonal Coulomb Hamiltonian.

    A Hamiltonian of the form

    .. math::

        H = \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q}
            + \frac12 \sum_{\sigma \tau, pq} V_{pq} n_{\sigma, p} n_{\tau, q}
            + \text{constant}.

    where :math:`n_{\sigma, p} = a_{\sigma, p}^\dagger a_{\sigma, p}` is the number
    operator on orbital :math:`p` with spin :math:`\sigma`.

    Here :math:`h_{pq}` is called the one-body tensor and :math:`V_{pq}` is called the
    diagonal Coulomb matrix.

    Attributes:
        one_body_tensor (np.ndarray): The one-body tensor.
        diag_coulomb_mat (np.ndarray): The diagonal Coulomb matrix.
        constant (float): The constant.
    """

    one_body_tensor: np.ndarray
    diag_coulomb_mat: np.ndarray
    constant: float = 0.0

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    def rotated(self, orbital_rotation: np.ndarray) -> DiagonalCoulombHamiltonian:
        r"""Return the Hamiltonian in a rotated orbital basis.

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
        diag_coulomb_mat_rotated = contract(
            "ab,Aa,Bb->AB",
            self.diag_coulomb_mat,
            orbital_rotation,
            orbital_rotation.conj(),
            optimize="greedy",
        )
        return DiagonalCoulombHamiltonian(
            one_body_tensor=one_body_tensor_rotated,
            diag_coulomb_mat=diag_coulomb_mat_rotated,
            constant=self.constant,
        )

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        n_alpha, n_beta = nelec
        linkstr_index_a = gen_linkstr_index(range(norb), n_alpha)
        linkstr_index_b = gen_linkstr_index(range(norb), n_beta)
        link_index = (linkstr_index_a, linkstr_index_b)
        two_body = absorb_h1e(
            self.one_body_tensor.real, self.diag_coulomb_mat, norb, nelec, 0.5
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
            make_hdiag(self.one_body_tensor, self.diag_coulomb_mat, norb, nelec)
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
        for p, q in itertools.product(range(self.norb), repeat=2):
            coeff = 0.5 * self.diag_coulomb_mat[p, q]
            op += FermionOperator(
                {
                    (cre_a(p), cre_a(q), des_a(q), des_a(p)): coeff,
                    (cre_a(p), cre_b(q), des_b(q), des_a(p)): coeff,
                    (cre_b(p), cre_a(q), des_a(q), des_b(p)): coeff,
                    (cre_b(p), cre_b(q), des_b(q), des_b(p)): coeff,
                }
            )
        return op

    @staticmethod
    def from_fermion_operator(op: FermionOperator) -> DiagonalCoulombHamiltonian:
        """Convert a FermionOperator to a DiagonalCoulombHamiltonian."""

        print("op = ", op)
        print("op (norm) = ", op.normal_ordered())

        # normal ordering
        op_norm = op.normal_ordered()
        dict_op = dict(op_norm)

        # extract norb
        orb_list = []
        for key in dict_op:
            for operator in key:
                orb = operator[2]
                orb_list.append(orb)
        norb = max(orb_list) + 1

        one_body_tensor = np.zeros((norb, norb), dtype=complex)
        diag_coulomb_mat = np.zeros((norb, norb), dtype=complex)

        # populate tensors
        for p, q in itertools.product(range(norb), repeat=2):
            # normal-ordered one-body terms
            one_body_list = [
                (cre_a(p), des_a(q)),
                (cre_b(p), des_b(q)),
            ]
            # normal-ordered two-body terms
            diag_coulomb_list = [
                (cre_a(p), cre_a(q), des_a(q), des_a(p)),
                (cre_a(q), cre_a(p), des_a(q), des_a(p)),  # rearrangement
                (cre_b(p), cre_a(q), des_b(q), des_a(p)),
                (cre_b(q), cre_a(p), des_b(q), des_a(p)),  # rearrangement
                (cre_b(p), cre_b(q), des_b(q), des_b(p)),
                (cre_b(q), cre_b(p), des_b(q), des_b(p)),  # rearrangement
            ]
            # excluded terms
            excluded_list = [
                (cre_b(p), des_a(q)),
                (cre_a(p), des_b(q)),
            ]
            for key in dict(op_norm):
                if key in one_body_list:
                    one_body_tensor[p, q] += 0.5 * dict_op.pop(key)
                if key in diag_coulomb_list:
                    diag_coulomb_mat[p, q] += 0.5 * dict_op.pop(key)
                if key in excluded_list:
                    del dict_op[key]

        if dict_op:
            print(f"Incompatible terms = {dict_op}")
            raise ValueError(
                "FermionOperator cannot be converted to DiagonalCoulombHamiltonian"
            )

        return DiagonalCoulombHamiltonian(
            one_body_tensor=one_body_tensor,
            diag_coulomb_mat=diag_coulomb_mat,
            constant=0,
        )
