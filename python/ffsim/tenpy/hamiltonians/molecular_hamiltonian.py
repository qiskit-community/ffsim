import numpy as np
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfFermionSite

# ignore lowercase argument and variable checks to maintain TeNPy naming conventions
# ruff: noqa: N803, N806


class MolecularChain(Lattice):
    def __init__(self, L, norb, site_a, **kwargs):
        basis = np.array(([norb, 0.0], [0, 1]))
        pos = np.array([[i, 0] for i in range(norb)])

        kwargs.setdefault("order", "default")
        kwargs.setdefault("bc", "open")
        kwargs.setdefault("bc_MPS", "finite")
        kwargs.setdefault("basis", basis)
        kwargs.setdefault("positions", pos)

        super().__init__([L, 1], [site_a] * norb, **kwargs)


class MolecularHamiltonianMPOModel(CouplingMPOModel):
    def __init__(self, params):
        CouplingMPOModel.__init__(self, params)

    def init_sites(self, params):
        cons_N = params.get("cons_N", "N")
        cons_Sz = params.get("cons_Sz", "Sz")
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        print(sorted(site.opnames))
        print(site.state_labels)
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
                h1 = one_body_tensor[p, q]
                if p == q:
                    self.add_onsite(h1, p, "Nu")
                    self.add_onsite(h1, p, "Nd")
                    self.add_onsite(constant / norb, p, "Id")
                else:
                    self.add_coupling(h1, p, "Cdu", q, "Cu", dx0)
                    self.add_coupling(h1, p, "Cdd", q, "Cd", dx0)

                for r in range(norb):
                    for s in range(norb):
                        h2 = two_body_tensor[p, q, r, s]
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
