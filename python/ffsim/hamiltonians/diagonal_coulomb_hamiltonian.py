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
from pyscf.fci.direct_uhf import make_hdiag
from scipy.sparse.linalg import LinearOperator

from ffsim.contract.diag_coulomb import diag_coulomb_linop
from ffsim.contract.num_op_sum import num_op_sum_linop
from ffsim.operators import FermionOperator, cre_a, cre_b, des_a, des_b
from ffsim.states import dim


@dataclasses.dataclass(frozen=True)
class DiagonalCoulombHamiltonian:
    r"""A diagonal Coulomb Hamiltonian.

    A Hamiltonian of the form

    .. math::

        H = \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q}
            + \frac12 \sum_{\sigma \tau, pq} V_{(\sigma \tau), pq} n_{\sigma, p}
            n_{\tau, q} + \text{constant}.

    where :math:`n_{\sigma, p} = a_{\sigma, p}^\dagger a_{\sigma, p}` is the number
    operator on orbital :math:`p` with spin :math:`\sigma`.

    Here :math:`h_{pq}` is called the one-body tensor and :math:`V_{(\sigma \tau), pq}`
    are called the diagonal Coulomb matrices. The brackets indicate that
    :math:`V_{(\sigma \tau)}` is a circulant matrix, which satisfies
    :math:`V_{\alpha\alpha}=V_{\beta\beta}` and :math:`V_{\alpha\beta}=V_{\beta\alpha}`.

    Attributes:
        one_body_tensor (np.ndarray): The one-body tensor :math:`h`.
        diag_coulomb_mats (np.ndarray): The diagonal Coulomb matrices
            :math:`V_{(\sigma \tau)}`, given as a pair of Numpy arrays specifying
            independent coefficients for alpha-alpha and alpha-beta interactions (in
            that order).
        constant (float): The constant.
    """

    one_body_tensor: np.ndarray
    diag_coulomb_mats: np.ndarray
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
        dc_linop = diag_coulomb_linop(
            (
                self.diag_coulomb_mats[0],
                self.diag_coulomb_mats[1],
                self.diag_coulomb_mats[0],
            ),
            norb,
            nelec,
        )

        def matvec(vec: np.ndarray):
            vec = vec.astype(complex, copy=False)
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
            (coeff_same_spin, coeff_diff_spin) = 0.5 * self.diag_coulomb_mats[:, p, q]
            op += FermionOperator(
                {
                    (cre_a(p), des_a(p), cre_a(q), des_a(q)): coeff_same_spin,
                    (cre_a(p), des_a(p), cre_b(q), des_b(q)): coeff_diff_spin,
                    (cre_b(p), des_b(p), cre_a(q), des_a(q)): coeff_diff_spin,
                    (cre_b(p), des_b(p), cre_b(q), des_b(q)): coeff_same_spin,
                }
            )
        return op

    @staticmethod
    def from_fermion_operator(op: FermionOperator) -> DiagonalCoulombHamiltonian:
        """Convert a FermionOperator to a DiagonalCoulombHamiltonian."""

        # extract norb
        norb = 1 + max(orb for term in op for _, _, orb in term)

        # initialize variables
        constant: float = 0
        one_body_tensor = np.zeros((norb, norb), dtype=complex)
        diag_coulomb_mats = np.zeros((2, norb, norb), dtype=float)

        # populate the tensors
        for term, coeff in op.items():
            if not term:  # constant term
                constant = coeff.real
            elif len(term) == 2:  # one-body term candidate
                (_, _, p), (_, _, q) = term
                valid_terms = [(cre_a(p), des_a(q)), (cre_b(p), des_b(q))]
                if term in valid_terms:
                    one_body_tensor[p, q] += 0.5 * coeff
                else:
                    raise ValueError(
                        "FermionOperator cannot be converted to "
                        f"DiagonalCoulombHamiltonian. The one-body term {term} is not "
                        "of the form a^\\dagger_{\\sigma, p} a_{\\sigma, q}."
                    )
            elif len(term) == 4:  # two-body term candidate
                (_, _, p), _, (_, _, q), _ = term
                valid_terms_same_spin = [
                    (cre_a(p), des_a(p), cre_a(q), des_a(q)),
                    (cre_b(p), des_b(p), cre_b(q), des_b(q)),
                ]
                valid_terms_diff_spin = [
                    (cre_a(p), des_a(p), cre_b(q), des_b(q)),
                    (cre_b(p), des_b(p), cre_a(q), des_a(q)),
                ]
                if term in valid_terms_same_spin:
                    diag_coulomb_mats[0][p, q] += coeff.real
                elif term in valid_terms_diff_spin:
                    diag_coulomb_mats[1][p, q] += coeff.real
                else:
                    raise ValueError(
                        "FermionOperator cannot be converted to "
                        f"DiagonalCoulombHamiltonian. The two-body term {term} is not "
                        "of the form n_{\\sigma, p} n_{\\tau, q}."
                    )
            else:
                raise ValueError(
                    "FermionOperator cannot be converted to DiagonalCoulombHamiltonian."
                    f" The term {term} is neither a constant, one-body, or two-body "
                    "term."
                )

        # ensure diag_coulomb_mats symmetry
        diag_coulomb_mats += diag_coulomb_mats.transpose(0, 2, 1)
        diag_coulomb_mats *= 0.5

        return DiagonalCoulombHamiltonian(
            one_body_tensor=one_body_tensor,
            diag_coulomb_mats=diag_coulomb_mats,
            constant=constant,
        )

    def _diag_(self, norb: int, nelec: tuple[int, int]) -> np.ndarray:
        """Return the diagonal entries of the Hamiltonian."""
        if np.iscomplexobj(self.one_body_tensor):
            raise NotImplementedError(
                "Computing diagonal of complex diagonal Coulomb Hamiltonian is not yet "
                "supported."
            )
        two_body_tensor_aa = np.zeros((self.norb, self.norb, self.norb, self.norb))
        two_body_tensor_ab = np.zeros((self.norb, self.norb, self.norb, self.norb))
        diag_coulomb_mat_aa, diag_coulomb_mat_ab = self.diag_coulomb_mats
        for p, q in itertools.product(range(self.norb), repeat=2):
            two_body_tensor_aa[p, p, q, q] = diag_coulomb_mat_aa[p, q]
            two_body_tensor_ab[p, p, q, q] = diag_coulomb_mat_ab[p, q]
        one_body_tensor = self.one_body_tensor + 0.5 * np.einsum(
            "prqr", two_body_tensor_aa
        )
        h1e = (one_body_tensor, one_body_tensor)
        h2e = (two_body_tensor_aa, two_body_tensor_ab, two_body_tensor_aa)
        return make_hdiag(h1e, h2e, norb=norb, nelec=nelec) + self.constant
