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

    def __init__(self, params):
        if "one_body_tensor" in params and isinstance(
            params["one_body_tensor"], np.ndarray
        ):
            self.one_body_tensor = params["one_body_tensor"]
        else:
            raise ValueError(
                "required parameter one_body_tensor is undefined or not an array"
            )
        self.norb = self.one_body_tensor.shape[0]
        CouplingMPOModel.__init__(self, params)

    def init_sites(self, params) -> SpinHalfFermionSite:
        """Initialize sites."""
        return SpinHalfFermionSite()

    def init_lattice(self, params) -> Lattice:
        """Initialize lattice."""
        site = self.init_sites(params)
        basis = np.array(([self.norb, 0], [0, 1]))
        pos = np.array([[i, 0] for i in range(self.norb)])
        lat = Lattice(
            [1, 1],
            [site] * self.norb,
            basis=basis,
            positions=pos,
        )
        return lat

    def init_terms(self, params) -> None:
        """Initialize terms."""
        two_body_tensor = params.get(
            "two_body_tensor",
            np.zeros((self.norb, self.norb, self.norb, self.norb)),
            expect_type="array",
        )
        constant = params.get("constant", 0, expect_type="real")

        # constant
        for p in range(self.norb):
            self.add_onsite(constant / self.norb, p, "Id")

        # one-body terms
        for p, q in itertools.product(range(self.norb), repeat=2):
            self._add_one_body(self.one_body_tensor[p, q], p, q)

        # two-body terms
        for p, q, r, s in itertools.product(range(self.norb), repeat=4):
            self._add_two_body(0.5 * two_body_tensor[p, q, r, s], p, q, r, s)

    def _add_one_body(self, coeff: complex, p: int, q: int) -> None:
        if p == q:
            self.add_onsite(coeff, p, "Ntot")
        else:
            dx0 = np.zeros(2)
            self.add_coupling(coeff, p, "Cdu", q, "Cu", dx0)
            self.add_coupling(coeff, p, "Cdd", q, "Cd", dx0)

    def _add_two_body(self, coeff: complex, p: int, q: int, r: int, s: int) -> None:
        if p == q == r == s:
            self.add_onsite(2 * coeff, p, "Nu Nd")
        else:
            dx0 = np.zeros(2)
            self.add_multi_coupling(
                coeff,
                [("Cdu", dx0, p), ("Cdu", dx0, r), ("Cu", dx0, s), ("Cu", dx0, q)],
            )
            self.add_multi_coupling(
                coeff,
                [("Cdu", dx0, p), ("Cdd", dx0, r), ("Cd", dx0, s), ("Cu", dx0, q)],
            )
            self.add_multi_coupling(
                coeff,
                [("Cdd", dx0, p), ("Cdu", dx0, r), ("Cu", dx0, s), ("Cd", dx0, q)],
            )
            self.add_multi_coupling(
                coeff,
                [("Cdd", dx0, p), ("Cdd", dx0, r), ("Cd", dx0, s), ("Cd", dx0, q)],
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
