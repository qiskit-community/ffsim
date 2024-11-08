# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
