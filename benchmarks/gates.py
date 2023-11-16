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

import ffsim


class GatesBenchmark:
    """Benchmark gates."""

    param_names = [
        "norb",
        "filling_fraction",
    ]
    params = [
        (4, 8, 12),
        (0.25, 0.5),
    ]

    def setup(self, norb: int, filling_fraction: float):
        self.norb = norb
        nocc = int(norb * filling_fraction)
        self.nelec = (nocc, nocc)

        rng = np.random.default_rng()

        self.vec = ffsim.random.random_statevector(
            ffsim.dim(self.norb, self.nelec), seed=rng
        )
        self.orbital_rotation = ffsim.random.random_unitary(self.norb, seed=rng)
        self.orbital_energies = rng.uniform(-1.0, 1.0, size=self.norb)
        self.diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(
            self.norb, seed=rng
        )
        ffsim.init_cache(self.norb, self.nelec)

    def time_apply_orbital_rotation_givens(self, *_):
        ffsim.apply_orbital_rotation(
            self.vec,
            self.orbital_rotation,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_orbital_rotation_lu(self, *_):
        ffsim.apply_orbital_rotation(
            self.vec,
            self.orbital_rotation,
            norb=self.norb,
            nelec=self.nelec,
            allow_col_permutation=True,
            copy=False,
        )

    def time_apply_num_op_sum_evolution(self, *_):
        ffsim.apply_num_op_sum_evolution(
            self.vec,
            self.orbital_energies,
            time=1.0,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_diag_coulomb_evolution(self, *_):
        ffsim.apply_diag_coulomb_evolution(
            self.vec,
            self.diag_coulomb_mat,
            time=1.0,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_givens_rotation(self, *_):
        ffsim.apply_givens_rotation(
            self.vec,
            theta=1.0,
            target_orbs=(0, 2),
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_num_interaction(self, *_):
        ffsim.apply_num_interaction(
            self.vec,
            theta=1.0,
            target_orb=1,
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_num_num_interaction(self, *_):
        ffsim.apply_num_num_interaction(
            self.vec,
            theta=1.0,
            target_orbs=(0, 1),
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_num_op_prod_interaction(self, *_):
        ffsim.apply_num_op_prod_interaction(
            self.vec,
            theta=1.0,
            target_orbs=([1], [0]),
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )

    def time_apply_tunneling_interaction(self, *_):
        ffsim.apply_tunneling_interaction(
            self.vec,
            theta=1.0,
            target_orbs=(0, 2),
            norb=self.norb,
            nelec=self.nelec,
            copy=False,
        )
