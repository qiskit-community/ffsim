# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

import numpy as np
from pyscf import ao2mo, mcscf, scf, symm

from ffsim.hamiltonians import MolecularHamiltonian

MOLPRO_ID = {
    "D2h": {
        "Ag": 1,
        "B1g": 4,
        "B2g": 6,
        "B3g": 7,
        "Au": 8,
        "B1u": 5,
        "B2u": 3,
        "B3u": 2,
    },
    "C2v": {"A1": 1, "A2": 4, "B1": 2, "B2": 3},
    "C2h": {"Ag": 1, "Bg": 4, "Au": 2, "Bu": 3},
    "D2": {"A ": 1, "B1": 4, "B2": 3, "B3": 2},
    "Cs": {"A'": 1, 'A"': 2},
    "C2": {"A": 1, "B": 2},
    "Ci": {"Ag": 1, "Au": 2},
    "C1": {
        "A": 1,
    },
}


def orbital_symmetries(
    hartree_fock: scf.hf.SCF, orbitals: Sequence[int]
) -> list[int] | None:
    if not hartree_fock.mol.symmetry:
        return None

    coeff = hartree_fock.mo_coeff[:, orbitals]
    idx = symm.label_orb_symm(
        hartree_fock.mol, hartree_fock.mol.irrep_name, hartree_fock.mol.symm_orb, coeff
    )
    return [MOLPRO_ID[hartree_fock.mol.groupname][i] for i in idx]


@dataclasses.dataclass
class MolecularData:
    norb: int
    nelec: tuple[int, int]
    core_energy: float
    one_body_tensor: np.ndarray
    two_body_tensor: np.ndarray
    dipole_integrals: np.ndarray | None = None
    orbital_symmetries: list[int] | None = None

    # TODO remove this after Python 3.10 dataclasses allows keyword-only arguments
    def __init__(
        self,
        *,
        norb: int,
        nelec: tuple[int, int],
        core_energy: float,
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        dipole_integrals: np.ndarray | None = None,
        orbital_symmetries: list[int] | None = None,
    ):
        self.norb = norb
        self.nelec = nelec
        self.core_energy = core_energy
        self.one_body_tensor = one_body_tensor
        self.two_body_tensor = two_body_tensor
        self.dipole_integrals = dipole_integrals
        self.orbital_symmetries = orbital_symmetries
        super().__init__()

    @property
    def hamiltonian(self):
        return MolecularHamiltonian(
            one_body_tensor=self.one_body_tensor,
            two_body_tensor=self.two_body_tensor,
            constant=self.core_energy,
        )

    @staticmethod
    def from_hartree_fock(
        hartree_fock: scf.hf.SCF, active_space: Iterable[int] | None = None
    ) -> "MolecularData":
        if active_space is None:
            norb = hartree_fock.mo_coeff.shape[0]
            active_space = range(norb)

        active_space = list(active_space)
        norb = len(active_space)
        n_electrons = int(np.sum(hartree_fock.mo_occ[active_space]))
        n_alpha = (n_electrons + hartree_fock.mol.spin) // 2
        n_beta = (n_electrons - hartree_fock.mol.spin) // 2

        mc = mcscf.CASCI(hartree_fock, norb, (n_alpha, n_beta))
        one_body_tensor, core_energy = mc.get_h1cas()
        two_body_tensor = ao2mo.restore(1, mc.get_h2cas(), mc.ncas)

        charges = hartree_fock.mol.atom_charges()
        coords = hartree_fock.mol.atom_coords()
        nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
        hartree_fock.mol.set_common_orig_(nuc_charge_center)

        mo_coeffs = hartree_fock.mo_coeff[:, active_space]
        dipole_integrals = hartree_fock.mol.intor("cint1e_r_sph", comp=3)
        dipole_integrals = np.einsum(
            "xij,ip,jq->xpq", dipole_integrals, mo_coeffs, mo_coeffs
        )

        orbsym = orbital_symmetries(hartree_fock, active_space)

        return MolecularData(
            norb=norb,
            nelec=(n_alpha, n_beta),
            core_energy=core_energy,
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dipole_integrals=dipole_integrals,
            orbital_symmetries=orbsym,
        )
