# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

from ffsim import FermionOperator


class FermionOperatorBenchmark:
    """Benchmark FermionOperator."""

    def setup(self):
        norb = 50
        n_terms = 100
        rng = np.random.default_rng()

        coeffs = {}
        for _ in range(n_terms):
            term_length = int(rng.integers(1, norb + 1))
            actions = [bool(i) for i in rng.integers(2, size=term_length)]
            spins = [bool(i) for i in rng.integers(2, size=term_length)]
            indices = [int(i) for i in rng.integers(norb, size=term_length)]
            coeff = rng.standard_normal() + 1j * rng.standard_normal()
            fermion_action = tuple(zip(actions, spins, indices))
            if fermion_action in coeffs:
                coeffs[fermion_action] += coeff
            else:
                coeffs[fermion_action] = coeff

        self.op = FermionOperator(coeffs)

    def time_normal_order(self):
        self.op.normal_ordered()
