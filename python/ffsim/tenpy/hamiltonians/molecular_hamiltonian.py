import numpy as np
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfFermionSite

from ffsim.tenpy.hamiltonians.lattices import MolecularChain

# ignore lowercase variable checks to maintain TeNPy naming conventions
# ruff: noqa: N806


class MolecularHamiltonianMPOModel(CouplingMPOModel):
    """Molecular Hamiltonian."""

    def __init__(self, params):
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
