from __future__ import annotations

import numpy as np
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfFermionSite

from ffsim.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from ffsim.tenpy.hamiltonians.lattices import MolecularChain

# ignore lowercase variable checks to maintain TeNPy naming conventions
# ruff: noqa: N806


class MolecularHamiltonianMPOModel(CouplingMPOModel):
    """Molecular Hamiltonian."""

    def __init__(self, params):
        if hasattr(self, "flag"):  # only call __init__ once
            return
        self.flag = True
        CouplingMPOModel.__init__(self, params)

    def init_sites(self, params):
        cons_N = params.get("cons_N", "N")
        cons_Sz = params.get("cons_Sz", "Sz")
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_lattice(self, params):
        L = params.get("L", 1)
        norb = params.get("norb", 4)
        site = self.init_sites(params)
        lat = MolecularChain(L, norb, site, basis=[[norb, 0], [0, 1]])
        return lat

    def init_terms(self, params):
        dx0 = np.array([0, 0])
        norb = params.get("norb", 4)
        one_body_tensor = params.get("one_body_tensor", np.zeros((norb, norb)))
        two_body_tensor = params.get(
            "two_body_tensor", np.zeros((norb, norb, norb, norb))
        )
        constant = params.get("constant", 0)

        for p in range(norb):
            for q in range(norb):
                h1 = one_body_tensor[q, p]
                if p == q:
                    self.add_onsite(h1, p, "Nu")
                    self.add_onsite(h1, p, "Nd")
                    self.add_onsite(constant / norb, p, "Id")
                else:
                    self.add_coupling(h1, p, "Cdu", q, "Cu", dx0)
                    self.add_coupling(h1, p, "Cdd", q, "Cd", dx0)

                for r in range(norb):
                    for s in range(norb):
                        h2 = two_body_tensor[q, p, s, r]
                        if p == q == r == s:
                            self.add_onsite(h2 / 2, p, "Nu")
                            self.add_onsite(-h2 / 2, p, "Nu Nu")
                            self.add_onsite(h2 / 2, p, "Nu")
                            self.add_onsite(-h2 / 2, p, "Cdu Cd Cdd Cu")
                            self.add_onsite(h2 / 2, p, "Nd")
                            self.add_onsite(-h2 / 2, p, "Cdd Cu Cdu Cd")
                            self.add_onsite(h2 / 2, p, "Nd")
                            self.add_onsite(-h2 / 2, p, "Nd Nd")
                        else:
                            self.add_multi_coupling(
                                h2 / 2,
                                [
                                    ("Cdu", dx0, p),
                                    ("Cdu", dx0, r),
                                    ("Cu", dx0, s),
                                    ("Cu", dx0, q),
                                ],
                            )
                            self.add_multi_coupling(
                                h2 / 2,
                                [
                                    ("Cdu", dx0, p),
                                    ("Cdd", dx0, r),
                                    ("Cd", dx0, s),
                                    ("Cu", dx0, q),
                                ],
                            )
                            self.add_multi_coupling(
                                h2 / 2,
                                [
                                    ("Cdd", dx0, p),
                                    ("Cdu", dx0, r),
                                    ("Cu", dx0, s),
                                    ("Cd", dx0, q),
                                ],
                            )
                            self.add_multi_coupling(
                                h2 / 2,
                                [
                                    ("Cdd", dx0, p),
                                    ("Cdd", dx0, r),
                                    ("Cd", dx0, s),
                                    ("Cd", dx0, q),
                                ],
                            )

    @staticmethod
    def from_molecular_hamiltonian(
        molecular_hamiltonian: MolecularHamiltonian, decimal_places: int | None = None
    ) -> MolecularHamiltonianMPOModel:
        r"""Convert MolecularHamiltonian to a MolecularHamiltonianMPOModel.

        Args:
            molecular_hamiltonian: The molecular Hamiltonian.
            decimal_places: The number of decimal places to which to round the input
                one-body and two-body tensors.

                .. note::
                    Rounding may reduce the MPO bond dimension.

        Returns:
            The molecular Hamiltonian as a TeNPy MPOModel.
        """

        if decimal_places:
            one_body_tensor = np.round(
                molecular_hamiltonian.one_body_tensor, decimals=decimal_places
            )
            two_body_tensor = np.round(
                molecular_hamiltonian.two_body_tensor, decimals=decimal_places
            )
        else:
            one_body_tensor = molecular_hamiltonian.one_body_tensor
            two_body_tensor = molecular_hamiltonian.two_body_tensor

        model_params = dict(
            cons_N="N",
            cons_Sz="Sz",
            L=1,
            norb=molecular_hamiltonian.norb,
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            constant=molecular_hamiltonian.constant,
        )
        mpo_model = MolecularHamiltonianMPOModel(model_params)

        return mpo_model
