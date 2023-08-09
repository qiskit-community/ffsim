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
from collections.abc import Iterable

import numpy as np
import pyscf
from pyscf import mcscf


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

    @staticmethod
    def from_hartree_fock(
        hartree_fock: pyscf.scf.hf.SCF, active_space: Iterable[int] | None = None
    ) -> "MolecularHamiltonian":
        if active_space is None:
            norb = hartree_fock.mo_coeff.shape[0]
            active_space = range(norb)

        active_space = list(active_space)
        norb = len(active_space)
        n_electrons = int(np.sum(hartree_fock.mo_occ[active_space]))
        n_alpha = (n_electrons + hartree_fock.mol.spin) // 2
        n_beta = (n_electrons - hartree_fock.mol.spin) // 2

        mc = mcscf.CASCI(hartree_fock, norb, (n_alpha, n_beta))
        one_body_tensor, core_energy = mc.get_h1cas()
        two_body_tensor = pyscf.ao2mo.restore(1, mc.get_h2cas(), mc.ncas)

        return MolecularHamiltonian(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            constant=core_energy,
        )
