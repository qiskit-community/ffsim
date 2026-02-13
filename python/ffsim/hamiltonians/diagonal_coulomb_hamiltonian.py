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

from ffsim import cistring, protocols
from ffsim.contract.diag_coulomb import diag_coulomb_linop
from ffsim.contract.num_op_sum import num_op_sum_linop
from ffsim.dimensions import dim
from ffsim.operators import FermionOperator, cre_a, cre_b, des_a, des_b


@dataclasses.dataclass(frozen=True)
class DiagonalCoulombHamiltonian(
    protocols.SupportsApproximateEquality,
    protocols.SupportsDiagonal,
    protocols.SupportsFermionOperator,
    protocols.SupportsLinearOperator,
):
    r"""A diagonal Coulomb Hamiltonian.

    A Hamiltonian of the form

    .. math::

        H = \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q}
            + \frac12 \sum_{\sigma \tau, pq} J^{\sigma \tau}_{pq} n_{\sigma, p}
            n_{\tau, q} + \text{constant}.

    where :math:`n_{\sigma, p} = a_{\sigma, p}^\dagger a_{\sigma, p}` is the number
    operator on orbital :math:`p` with spin :math:`\sigma`.

    Here :math:`h_{pq}` is called the one-body tensor and the :math:`J^{\sigma \tau}`
    are called diagonal Coulomb matrices. We require that
    :math:`J^{\alpha\alpha}=J^{\beta\beta}` and
    :math:`J^{\alpha\beta}=J^{\beta\alpha}`, so only two matrices are needed to describe
    the Hamiltonian.

    Attributes:
        one_body_tensor (np.ndarray): The one-body tensor :math:`h`.
        diag_coulomb_mats (np.ndarray): The diagonal Coulomb matrices
            :math:`J^{\alpha\alpha}` and :math:`J^{\alpha\beta}`, given as a pair
            of Numpy arrays specifying independent coefficients for alpha-alpha and
            alpha-beta interactions (in that order).
        constant (float): The constant.
    """

    one_body_tensor: np.ndarray
    diag_coulomb_mats: np.ndarray
    constant: float = 0.0

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    def _linear_operator_(
        self, norb: int, nelec: int | tuple[int, int]
    ) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        assert isinstance(nelec, tuple)
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
        r"""Initialize a DiagonalCoulombHamiltonian from a FermionOperator.

        The input operator must contain only terms of the following form:

        - A real-valued constant
        - :math:`a^\dagger_{\sigma, p} a_{\sigma, q}`
        - :math:`n_{\sigma, p} n_{\tau, q}`

        Any other terms will cause an error to be raised. No attempt will be made to
        normal-order terms.

        Args:
            op: The FermionOperator from which to initialize the
                DiagonalCoulombHamiltonian.

        Returns:
            The DiagonalCoulombHamiltonian represented by the input FermionOperator.
        """

        # extract norb
        norb = 1 + max(orb for term in op for _, _, orb in term)

        # initialize variables
        constant: float = 0
        one_body_tensor = np.zeros((norb, norb), dtype=complex)
        diag_coulomb_mats = np.zeros((2, norb, norb), dtype=float)

        # populate the tensors
        for term, coeff in op.items():
            if not term:
                # constant term
                if coeff.imag:
                    raise ValueError(
                        f"Constant term must be real. Instead, got {coeff}."
                    )
                constant = coeff.real
            elif len(term) == 2:
                # one-body term
                (_, _, p), (_, _, q) = term
                valid_terms = [(cre_a(p), des_a(q)), (cre_b(p), des_b(q))]
                if term in valid_terms:
                    one_body_tensor[p, q] += 0.5 * coeff
                else:
                    raise ValueError(
                        "FermionOperator cannot be converted to "
                        f"DiagonalCoulombHamiltonian. The quadratic term {term} is not "
                        r"of the form a^\dagger_{\sigma, p} a_{\sigma, q}."
                    )
            elif len(term) == 4:
                # two-body term
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
                        f"DiagonalCoulombHamiltonian. The quartic term {term} is not "
                        r"of the form n_{\sigma, p} n_{\tau, q}."
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

    def _diag_(self, norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
        """Return the diagonal entries of the Hamiltonian."""
        assert isinstance(nelec, tuple)
        n_alpha, n_beta = nelec
        mat_aa, mat_ab = self.diag_coulomb_mats

        # Build occupation vectors from occupied orbital lists
        occslst_a = cistring.gen_occslst(range(norb), n_alpha)
        occslst_b = cistring.gen_occslst(range(norb), n_beta)
        occ_a = np.zeros((len(occslst_a), norb))
        occ_b = np.zeros((len(occslst_b), norb))
        occ_a[np.arange(len(occslst_a))[:, None], occslst_a] = 1
        occ_b[np.arange(len(occslst_b))[:, None], occslst_b] = 1

        # One-body
        diag = np.diag(self.one_body_tensor).real
        vals_a = occ_a @ diag
        vals_b = occ_b @ diag

        # Same-spin two-body
        vals_a += 0.5 * np.sum((occ_a @ mat_aa) * occ_a, axis=1)
        vals_b += 0.5 * np.sum((occ_b @ mat_aa) * occ_b, axis=1)

        # Opposit-spin two-body
        result = occ_a @ mat_ab @ occ_b.T

        # Combine
        result += vals_a[:, None]
        result += vals_b[None, :]
        result += self.constant
        return result.reshape(-1)

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, DiagonalCoulombHamiltonian):
            if not np.allclose(self.constant, other.constant, rtol=rtol, atol=atol):
                return False
            if not np.allclose(
                self.one_body_tensor, other.one_body_tensor, rtol=rtol, atol=atol
            ):
                return False
            if not np.allclose(
                self.diag_coulomb_mats, other.diag_coulomb_mats, rtol=rtol, atol=atol
            ):
                return False
            return True
        return NotImplemented
