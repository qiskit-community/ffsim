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
import scipy.sparse.linalg
from pyscf.fci import cistring
from pyscf.fci.direct_nosym import absorb_h1e, make_hdiag
from pyscf.fci.fci_slow import contract_2e
from scipy.sparse.linalg import LinearOperator

from ffsim._lib import FermionOperator
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

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        n_alpha, n_beta = nelec
        linkstr_index_a = cistring.gen_linkstr_index(range(norb), n_alpha)
        linkstr_index_b = cistring.gen_linkstr_index(range(norb), n_beta)
        link_index = (linkstr_index_a, linkstr_index_b)
        two_body = absorb_h1e(
            self.one_body_tensor, self.two_body_tensor, norb, nelec, 0.5
        )
        dim_ = dim(norb, nelec)

        def matvec(vec: np.ndarray):
            return self.constant * vec + contract_2e(
                two_body, vec, norb, nelec, link_index=link_index
            )

        return scipy.sparse.linalg.LinearOperator(
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
