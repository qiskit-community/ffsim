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

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ffsim.contract.hamiltonian import hamiltonian_linop, hamiltonian_trace


@dataclasses.dataclass
class MolecularHamiltonian:
    r"""A molecular Hamiltonian.

    A Hamiltonian of the form

    .. math::

        H = \sum_{pq, \sigma} h_{pq} a^\dagger_{p, \sigma} a_{q, \sigma}
            + \frac12 \sum_{pqrs, \sigma \tau} h_{pqrs}
            a^\dagger_{p, \sigma} a^\dagger_{r, \tau} a_{s, \tau} a_{q, \sigma}
            + \text{constant}.

    Here :math:`h_{pq}` is called the one-body tensor and :math:`h_{pqrs}` is called
    the two-body tensor.

    Attributes:
        one_body_tensor: The one-body tensor.
        two_body_tensor: The two-body tensor.
        constant: The constant.
    """

    one_body_tensor: np.ndarray
    two_body_tensor: np.ndarray
    constant: float = 0.0

    @property
    def norb(self):
        """The number of spatial orbitals."""
        return self.one_body_tensor.shape[0]

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]) -> LinearOperator:
        """Return a SciPy LinearOperator representing the object."""
        return hamiltonian_linop(
            norb=norb,
            nelec=nelec,
            one_body_tensor=self.one_body_tensor,
            two_body_tensor=self.two_body_tensor,
            constant=self.constant,
        )

    def _trace_(self, norb: int, nelec: tuple[int, int]) -> float:
        """Return the trace of the object."""
        return hamiltonian_trace(
            norb=norb,
            nelec=nelec,
            one_body_tensor=self.one_body_tensor,
            two_body_tensor=self.two_body_tensor,
            constant=self.constant,
        )
