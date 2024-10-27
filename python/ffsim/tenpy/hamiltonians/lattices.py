import numpy as np
from tenpy.models.lattice import Lattice

# ignore lowercase argument checks to maintain TeNPy naming conventions
# ruff: noqa: N803


class MolecularChain(Lattice):
    """Molecular chain."""

    def __init__(self, L, norb, site_a, **kwargs):
        basis = np.array(([norb, 0.0], [0, 1]))
        pos = np.array([[i, 0] for i in range(norb)])

        kwargs.setdefault("order", "default")
        kwargs.setdefault("bc", "open")
        kwargs.setdefault("bc_MPS", "finite")
        kwargs.setdefault("basis", basis)
        kwargs.setdefault("positions", pos)

        super().__init__([L, 1], [site_a] * norb, **kwargs)
