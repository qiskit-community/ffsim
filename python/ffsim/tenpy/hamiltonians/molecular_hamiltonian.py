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

import itertools

import numpy as np
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfFermionSite

from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian

# ignore lowercase variable checks to maintain TeNPy naming conventions
# ruff: noqa: N806


class MolecularHamiltonianMPOModel(CouplingMPOModel):
    """Molecular Hamiltonian."""

    def init_sites(self, params):
        cons_N = params.get("cons_N", "N")
        cons_Sz = params.get("cons_Sz", "Sz")
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_lattice(self, params):
        L = params.get("L", 1)
        norb = params.get("norb", 4)
        site = self.init_sites(params)
        basis = np.array(([norb, 0.0], [0, 1]))
        pos = np.array([[i, 0] for i in range(norb)])
        lat = Lattice(
            [L, 1],
            [site] * norb,
            order="default",
            bc="open",
            bc_MPS="finite",
            basis=basis,
            positions=pos,
        )
        return lat

    def init_terms(self, params):
        dx0 = np.array([0, 0])
        norb = params.get("norb", 4)
        one_body_tensor = params.get("one_body_tensor", np.zeros((norb, norb)))
        two_body_tensor = params.get(
            "two_body_tensor", np.zeros((norb, norb, norb, norb))
        )
        constant = params.get("constant", 0)

        for p, q in itertools.product(range(norb), repeat=2):
            h1 = one_body_tensor[q, p]
            if p == q:
                self.add_onsite(h1, p, "Nu")
                self.add_onsite(h1, p, "Nd")
                self.add_onsite(constant / norb, p, "Id")
            else:
                self.add_coupling(h1, p, "Cdu", q, "Cu", dx0)
                self.add_coupling(h1, p, "Cdd", q, "Cd", dx0)

            for r, s in itertools.product(range(norb), repeat=2):
                h2 = two_body_tensor[q, p, s, r]
                if p == q == r == s:
                    self.add_onsite(0.5 * h2, p, "Nu")
                    self.add_onsite(-0.5 * h2, p, "Nu Nu")
                    self.add_onsite(0.5 * h2, p, "Nu")
                    self.add_onsite(-0.5 * h2, p, "Cdu Cd Cdd Cu")
                    self.add_onsite(0.5 * h2, p, "Nd")
                    self.add_onsite(-0.5 * h2, p, "Cdd Cu Cdu Cd")
                    self.add_onsite(0.5 * h2, p, "Nd")
                    self.add_onsite(-0.5 * h2, p, "Nd Nd")
                else:
                    self.add_multi_coupling(
                        0.5 * h2,
                        [
                            ("Cdu", dx0, p),
                            ("Cdu", dx0, r),
                            ("Cu", dx0, s),
                            ("Cu", dx0, q),
                        ],
                    )
                    self.add_multi_coupling(
                        0.5 * h2,
                        [
                            ("Cdu", dx0, p),
                            ("Cdd", dx0, r),
                            ("Cd", dx0, s),
                            ("Cu", dx0, q),
                        ],
                    )
                    self.add_multi_coupling(
                        0.5 * h2,
                        [
                            ("Cdd", dx0, p),
                            ("Cdu", dx0, r),
                            ("Cu", dx0, s),
                            ("Cd", dx0, q),
                        ],
                    )
                    self.add_multi_coupling(
                        0.5 * h2,
                        [
                            ("Cdd", dx0, p),
                            ("Cdd", dx0, r),
                            ("Cd", dx0, s),
                            ("Cd", dx0, q),
                        ],
                    )

    @staticmethod
    def from_molecular_hamiltonian(
        molecular_hamiltonian: MolecularHamiltonian,
    ) -> MolecularHamiltonianMPOModel:
        r"""Convert MolecularHamiltonian to a MolecularHamiltonianMPOModel.

        Args:
            molecular_hamiltonian: The molecular Hamiltonian.

        Returns:
            The molecular Hamiltonian as a `TeNPy MPOModel <https://tenpy.readthedocs.io/en/stable/reference/tenpy.models.model.MPOModel.html#tenpy.models.model.MPOModel>`__.
        """

        model_params = dict(
            cons_N="N",
            cons_Sz="Sz",
            L=1,
            norb=molecular_hamiltonian.norb,
            one_body_tensor=molecular_hamiltonian.one_body_tensor,
            two_body_tensor=molecular_hamiltonian.two_body_tensor,
            constant=molecular_hamiltonian.constant,
        )

        return MolecularHamiltonianMPOModel(model_params)
