# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""TeNPy molecular Hamiltonian."""

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

    def init_sites(self, params) -> SpinHalfFermionSite:
        """Initialize sites."""
        return SpinHalfFermionSite()

    def init_lattice(self, params) -> Lattice:
        """Initialize lattice."""
        one_body_tensor = params.get("one_body_tensor", None, expect_type="array")
        norb = one_body_tensor.shape[0]
        site = self.init_sites(params)
        basis = np.array(([norb, 0], [0, 1]))
        pos = np.array([[i, 0] for i in range(norb)])
        lat = Lattice(
            [1, 1],
            [site] * norb,
            basis=basis,
            positions=pos,
        )
        return lat

    def init_terms(self, params) -> None:
        """Initialize terms."""
        dx0 = np.array([0, 0])
        one_body_tensor = params.get("one_body_tensor", None, expect_type="array")
        two_body_tensor = params.get("two_body_tensor", None, expect_type="array")
        constant = params.get("constant", 0, expect_type="real")
        norb = one_body_tensor.shape[0]

        for p in range(norb):
            h1 = one_body_tensor[p, p]
            self.add_onsite(h1, p, "Ntot")
            h2 = two_body_tensor[p, p, p, p]
            self.add_onsite(h2, p, "Ntot")
            self.add_onsite(-0.5 * h2, p, "Nu Nu")
            self.add_onsite(-0.5 * h2, p, "Cdu Cd Cdd Cu")
            self.add_onsite(-0.5 * h2, p, "Cdd Cu Cdu Cd")
            self.add_onsite(-0.5 * h2, p, "Nd Nd")
            self.add_onsite(constant / norb, p, "Id")

        for p, q in itertools.combinations(range(norb), 2):
            self.add_coupling(
                one_body_tensor[p, q], p, "Cdu", q, "Cu", dx0, plus_hc=True
            )
            self.add_coupling(
                one_body_tensor[p, q], p, "Cdd", q, "Cd", dx0, plus_hc=True
            )

        for p, s in itertools.combinations_with_replacement(range(norb), 2):
            for q, r in itertools.combinations_with_replacement(range(norb), 2):
                if not p == q == r == s:
                    indices = [(p, q, r, s)]
                    if p < s:
                        indices.append((s, q, r, p))
                    if q < r:
                        indices.append((p, r, q, s))
                    if p < s and q < r:
                        indices.append((s, r, q, p))

                    for i, j, k, l in indices:
                        h2 = two_body_tensor[i, j, k, l]
                        self.add_multi_coupling(
                            0.5 * h2,
                            [
                                ("Cdu", dx0, j),
                                ("Cdu", dx0, l),
                                ("Cu", dx0, k),
                                ("Cu", dx0, i),
                            ],
                        )
                        self.add_multi_coupling(
                            0.5 * h2,
                            [
                                ("Cdu", dx0, j),
                                ("Cdd", dx0, l),
                                ("Cd", dx0, k),
                                ("Cu", dx0, i),
                            ],
                        )
                        self.add_multi_coupling(
                            0.5 * h2,
                            [
                                ("Cdd", dx0, j),
                                ("Cdu", dx0, l),
                                ("Cu", dx0, k),
                                ("Cd", dx0, i),
                            ],
                        )
                        self.add_multi_coupling(
                            0.5 * h2,
                            [
                                ("Cdd", dx0, j),
                                ("Cdd", dx0, l),
                                ("Cd", dx0, k),
                                ("Cd", dx0, i),
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
            The molecular Hamiltonian as an MPO model.
        """

        model_params = dict(
            one_body_tensor=molecular_hamiltonian.one_body_tensor,
            two_body_tensor=molecular_hamiltonian.two_body_tensor,
            constant=molecular_hamiltonian.constant,
        )

        return MolecularHamiltonianMPOModel(model_params)
