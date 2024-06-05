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
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

from ffsim._lib import FermionOperator
from ffsim.contract.diag_coulomb import diag_coulomb_linop
from ffsim.contract.num_op_sum import num_op_sum_linop
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

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        dim_ = dim(norb, nelec)
        eigs, vecs = scipy.linalg.eigh(self.one_body_tensor)
        num_linop = num_op_sum_linop(eigs, norb, nelec, orbital_rotation=vecs)
        dc_linop = diag_coulomb_linop(self.diag_coulomb_mat, norb, nelec)

        def matvec(vec: np.ndarray):
            result = self.constant * vec
            result += num_linop @ vec
            result += dc_linop @ vec
            return result

        return LinearOperator(
            shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
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
                    (cre_a(p), des_a(p), cre_a(q), des_a(q)): coeff,
                    (cre_a(p), des_a(p), cre_b(q), des_b(q)): coeff,
                    (cre_b(p), des_b(p), cre_a(q), des_a(q)): coeff,
                    (cre_b(p), des_b(p), cre_b(q), des_b(q)): coeff,
                }
            )
        return op

    @staticmethod
    def from_fermion_operator(op: FermionOperator) -> DiagonalCoulombHamiltonian:
        """Convert a FermionOperator to a DiagonalCoulombHamiltonian."""

        dict_op = dict(op)

        # extract norb
        orb_list = []
        for key in dict_op:
            for operator in key:
                orb = operator[2]
                orb_list.append(orb)
        norb = max(orb_list) + 1

        # check for incompatible onsite interaction terms
        same_spin_total, diff_spin_total = 0.0, 0.0
        for p in range(norb):
            same_spin_list = [
                (cre_a(p), des_a(p), cre_a(p), des_a(p)),
                (cre_b(p), des_b(p), cre_b(p), des_b(p)),
            ]
            diff_spin_list = [
                (cre_a(p), des_a(p), cre_b(p), des_b(p)),
                (cre_b(p), des_b(p), cre_a(p), des_a(p)),
            ]
            for key in dict(op):
                if key in diff_spin_list:
                    diff_spin_total += np.real(dict_op[key])
                elif key in same_spin_list:
                    same_spin_total += np.real(dict_op[key])
        if same_spin_total != diff_spin_total:
            raise ValueError(
                "FermionOperator cannot be converted to DiagonalCoulombHamiltonian due "
                "to incompatible onsite interaction terms"
            )

        # initialize variables
        constant: float = 0
        one_body_tensor = np.zeros((norb, norb), dtype=complex)
        diag_coulomb_mat = np.zeros((norb, norb), dtype=float)

        # populate tensors
        for p, q in itertools.product(range(norb), repeat=2):
            # one-body terms
            one_body_list = [
                (cre_a(p), des_a(q)),
                (cre_b(p), des_b(q)),
            ]
            # two-body terms
            diag_coulomb_list = [
                (cre_a(p), des_a(p), cre_a(q), des_a(q)),
                (cre_a(p), des_a(p), cre_b(q), des_b(q)),
                (cre_b(p), des_b(p), cre_a(q), des_a(q)),
                (cre_b(p), des_b(p), cre_b(q), des_b(q)),
            ]
            for key in dict(op):
                if key == ():
                    constant = np.real(dict_op[key])
                if key in one_body_list:
                    one_body_tensor[p, q] += 0.5 * dict_op.pop(key)
                if key in diag_coulomb_list:
                    diag_coulomb_mat[p, q] += 0.5 * np.real(dict_op.pop(key))

        # remove constant term
        if () in dict_op:
            del dict_op[()]

        # check for incompatible leftover terms
        if dict_op:
            raise ValueError(
                "FermionOperator cannot be converted to DiagonalCoulombHamiltonian due "
                "to incompatible leftover terms"
            )

        # ensure diag_coulomb_mat symmetry
        diag_coulomb_mat = (diag_coulomb_mat + diag_coulomb_mat.T) / 2

        return DiagonalCoulombHamiltonian(
            one_body_tensor=one_body_tensor,
            diag_coulomb_mat=diag_coulomb_mat,
            constant=constant,
        )
