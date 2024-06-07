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
import fnmatch

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

        H = \sum_{\sigma, pq} h_{\sigma, pq} a^\dagger_{\sigma, p} a_{\sigma, q}
            + \frac12 \sum_{\sigma \tau, pq} V_{\sigma \tau, pq} n_{\sigma, p}
            n_{\tau, q} + \text{constant}.

    where :math:`n_{\sigma, p} = a_{\sigma, p}^\dagger a_{\sigma, p}` is the number
    operator on orbital :math:`p` with spin :math:`\sigma`.

    Here :math:`h_{\sigma, pq}` are called the one-body tensors and
    :math:`V_{\sigma \tau, pq}` are called the diagonal Coulomb matrices, which are
    indexed by the spin indices.

    Attributes:
        one_body_tensors (np.ndarray): The one-body tensors.
        diag_coulomb_mats (np.ndarray): The diagonal Coulomb matrices.
        constant (float): The constant.
    """

    one_body_tensors: np.ndarray
    diag_coulomb_mats: np.ndarray
    constant: float = 0.0

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""

        print("shape = ", self.one_body_tensors.shape)

        return self.one_body_tensors.shape[1]

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        dim_ = dim(norb, nelec)

        num_linops = []
        for one_body_tensor in self.one_body_tensors:
            eigs, vecs = scipy.linalg.eigh(one_body_tensor)
            num_linop = num_op_sum_linop(eigs, norb, nelec, orbital_rotation=vecs)
            num_linops.append(num_linop)

        # dc_linops = []
        # for diag_coulomb_mat in self.diag_coulomb_mats:
        #     dc_linop = diag_coulomb_linop(diag_coulomb_mat, norb, nelec)
        #     dc_linops.append(dc_linop)

        # dc_linop = diag_coulomb_linop((self.diag_coulomb_mats[0], self.diag_coulomb_mats[1], self.diag_coulomb_mats[2]), norb, nelec)
        dc_linop = diag_coulomb_linop(tuple(self.diag_coulomb_mats), norb, nelec)

        # dc_linops[1] = diag_coulomb_linop(2*self.diag_coulomb_mats[1], norb, nelec)
        print("dc_linop = ", np.diag(dc_linop.dot(np.eye(dim_))))

        # dc_linops[1] = np.diag([2.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 2.+0.j, 1.+0.j, 1.+0.j,
        #                         0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 2.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j,
        #                         0.+0.j, 2.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j, 1.+0.j, 2.+0.j, 1.+0.j,
        #                         0.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j, 2.+0.j])

        def matvec(vec: np.ndarray):
            result = self.constant * vec
            for linop in num_linops:
                result += linop @ vec
            # for linop in dc_linops:
            #     result += linop @ vec
            result += dc_linop @ vec
            return result

        return LinearOperator(
            shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
        )

    def _fermion_operator_(self) -> FermionOperator:
        """Return a FermionOperator representing the object."""
        op = FermionOperator({(): self.constant})
        for p, q in itertools.product(range(self.norb), repeat=2):
            # (coeff_a, coeff_b) = self.one_body_tensors[:, p, q]
            coeff_a = 2 * self.one_body_tensors[0, p, q]
            coeff_b = 2 * self.one_body_tensors[1, p, q]
            op += FermionOperator(
                {
                    (cre_a(p), des_a(q)): coeff_a,
                    (cre_b(p), des_b(q)): coeff_b,
                }
            )
        for p, q in itertools.product(range(self.norb), repeat=2):
            # (coeff_aa, coeff_ab, coeff_bb) = 0.5 * self.diag_coulomb_mats[:, p, q]
            coeff_aa = 0.5 * self.diag_coulomb_mats[0, p, q]
            coeff_ab = 0.5 * self.diag_coulomb_mats[1, p, q]
            coeff_bb = 0.5 * self.diag_coulomb_mats[2, p, q]
            op += FermionOperator(
                {
                    (cre_a(p), des_a(p), cre_a(q), des_a(q)): coeff_aa,
                    (cre_a(p), des_a(p), cre_b(q), des_b(q)): coeff_ab,
                    (cre_b(p), des_b(p), cre_a(q), des_a(q)): coeff_ab,
                    (cre_b(p), des_b(p), cre_b(q), des_b(q)): coeff_bb,
                }
            )
        return op

    @staticmethod
    def from_fermion_operator(op: FermionOperator) -> DiagonalCoulombHamiltonian:
        """Convert a FermionOperator to a DiagonalCoulombHamiltonian."""

        # print("op here = ", op)

        # extract norb
        orb_list = []
        for key, _ in op.items():
            for operator in key:
                orb = operator[2]
                orb_list.append(orb)
        norb = max(orb_list) + 1

        # initialize variables
        constant: float = 0
        one_body_tensors = np.zeros((2, norb, norb), dtype=complex)
        diag_coulomb_mats = np.zeros((3, norb, norb), dtype=float)

        # populate the tensors (new)
        for term, coeff in op.items():
            # print(f"(term, coeff) = ({term}, {coeff})")
            if len(term) == 0:  # constant term
                constant = np.real(coeff)
            elif len(term) == 2:  # one-body term candidate
                p = term[0][2]
                q = term[1][2]
                # print(p, q)
                term_a = (cre_a(p), des_a(q))
                term_b = (cre_b(p), des_b(q))
                if term == term_a:
                    # print("one-body a match")
                    one_body_tensors[0][p, q] += 0.5 * coeff  # 0.5
                elif term == term_b:
                    # print("one-body b match")
                    one_body_tensors[1][p, q] += 0.5 * coeff  # 0.5
                else:
                    raise ValueError(
                        "FermionOperator cannot be converted to DiagonalCoulombHamiltonian")
            elif len(term) == 4:  # two-body term candidate
                p = term[0][2]
                q = term[2][2]
                # print(p, q)
                # two-body terms
                term_aa = (cre_a(p), des_a(p), cre_a(q), des_a(q))
                terms_ab = [(cre_a(p), des_a(p), cre_b(q), des_b(q)),
                            (cre_b(p), des_b(p), cre_a(q), des_a(q))]
                term_bb = (cre_b(p), des_b(p), cre_b(q), des_b(q))
                if term == term_aa:
                    # print("two-body aa match")
                    diag_coulomb_mats[0][p, q] += 2 * np.real(coeff)
                elif term in terms_ab:
                    # print("two-body ab match")
                    diag_coulomb_mats[1][p, q] += np.real(coeff)
                elif term == term_bb:
                    # print("two-body bb match")
                    diag_coulomb_mats[2][p, q] += 2 * np.real(coeff)
                else:
                    raise ValueError(
                        "FermionOperator cannot be converted to DiagonalCoulombHamiltonian")
            else:
                raise ValueError("FermionOperator cannot be converted to DiagonalCoulombHamiltonian")

        # ensure diag_coulomb_mats symmetry
        for i in range(3):
            diag_coulomb_mats[i] = (diag_coulomb_mats[i] + diag_coulomb_mats[i].T) / 2

        # print("diag_coulomb_mats[0] = ", diag_coulomb_mats[0])
        # print("diag_coulomb_mats[1] = ", diag_coulomb_mats[1])
        # print("diag_coulomb_mats[2] = ", diag_coulomb_mats[2])

        # print("obt shape = ", one_body_tensors.shape)

        # print("actual one_body_tensors = ", one_body_tensors)
        # print("actual diag_coulomb_mats = ", diag_coulomb_mats)
        # print("actual constant = ", constant)

        return DiagonalCoulombHamiltonian(
            one_body_tensors=one_body_tensors,
            diag_coulomb_mats=diag_coulomb_mats,
            constant=constant,
        )
