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
from pyscf import ao2mo, cc, gto, mcscf, mp, scf, symm
from pyscf.scf.hf import SCF

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


def orbital_symmetries(hartree_fock: SCF, orbitals: Sequence[int]) -> list[int] | None:
    if not hartree_fock.mol.symmetry:
        return None

    coeff = hartree_fock.mo_coeff[:, orbitals]
    idx = symm.label_orb_symm(
        hartree_fock.mol, hartree_fock.mol.irrep_name, hartree_fock.mol.symm_orb, coeff
    )
    return [MOLPRO_ID[hartree_fock.mol.groupname][i] for i in idx]


@dataclasses.dataclass(frozen=True)
class MolecularData:
    """Class for storing molecular data.

    Attributes:
        atom: The coordinates of the atoms in the molecule.
        basis: The basis set, e.g. "sto-6g".
        spin: The spin of the molecule.
        symmetry: The symmetry of the molecule.
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        mo_coeff: Hartree-Fock canonical orbital coefficients in the AO basis.
        mo_occ: Hartree-Fock canonical orbital occupancies.
        active_space: The orbitals included in the active space.
        core_energy: The core energy.
        one_body_tensor: The one-body tensor.
        two_body_integrals: The two-body integrals in compressed format.
        hf_energy: The Hartree-Fock energy.
        mp2_energy: The MP2 energy.
        mp2_t2: The MP2 t2 amplitudes.
        ccsd_energy: The CCSD energy.
        ccsd_t1: The CCSD t1 amplitudes.
        ccsd_t2: The CCSD t2 amplitudes.
        fci_energy: The FCI energy.
        fci_vec: The FCI state vector.
        dipole_integrals: The dipole integrals.
        orbital_symmetries: The orbital symmetries.
    """

    # molecule information corresponding to attributes of pyscf.gto.Mole
    atom: list[tuple[str, tuple[float, float, float]]]
    basis: str
    spin: int
    symmetry: str | None
    # active space information
    norb: int
    nelec: tuple[int, int]
    active_space: list[int]
    # Hamiltonian coefficients
    core_energy: float
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    # Hartree-Fock data
    hf_energy: float
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    # MP2 data
    mp2_energy: float | None = None
    mp2_t2: np.ndarray | None = None
    # CCSD data
    ccsd_energy: float | None = None
    ccsd_t1: np.ndarray | None = None
    ccsd_t2: np.ndarray | None = None
    # FCI data
    fci_energy: float | None = None
    fci_vec: np.ndarray | None = None
    # other information
    dipole_integrals: np.ndarray | None = None
    orbital_symmetries: list[int] | None = None

    @property
    def hamiltonian(self) -> MolecularHamiltonian:
        """The Hamiltonian defined by the molecular data."""
        return MolecularHamiltonian(
            one_body_tensor=self.one_body_integrals,
            two_body_tensor=ao2mo.restore(1, self.two_body_integrals, self.norb),
            constant=self.core_energy,
        )

    @staticmethod
    def from_scf(
        hartree_fock: SCF,
        active_space: Iterable[int] | None = None,
        *,
        mp2: bool = False,
        ccsd: bool = False,
        fci: bool = False,
    ) -> "MolecularData":
        """Initialize a MolecularData object from a Hartree-Fock calculation.

        Args:
            hartree_fock: The Hartree-Fock object.
            active_space: An optional list of orbitals to use for the active space.
            mp2: Whether to calculate and store the MP2 energy and t2 amplitudes.
            ccsd: Whether to calculate and store the CCSD energy, t1, and t2 amplitudes.
            fci: Whether to calculate and store the FCI energy and state vector.
        """
        if not hartree_fock.e_tot:
            raise ValueError(
                "You must run the Hartree-Fock object before a MolecularData can be "
                "initialized from it."
            )
        hf_energy = hartree_fock.e_tot

        # get core energy and one- and two-body integrals
        if active_space is None:
            norb = hartree_fock.mol.nao_nr()
            active_space = range(norb)
        active_space = list(active_space)
        norb = len(active_space)
        n_electrons = int(sum(hartree_fock.mo_occ[active_space]))
        n_alpha = (n_electrons + hartree_fock.mol.spin) // 2
        n_beta = (n_electrons - hartree_fock.mol.spin) // 2
        cas = mcscf.CASCI(hartree_fock, norb, (n_alpha, n_beta))
        mo = cas.sort_mo(active_space, base=0)
        one_body_tensor, core_energy = cas.get_h1cas(mo)
        two_body_integrals = cas.get_h2cas(mo)

        # run MP2 if requested
        frozen = [i for i in range(hartree_fock.mol.nao_nr()) if i not in active_space]
        mp2_energy = None
        mp2_t2 = None
        if mp2:
            mp2_solver = mp.MP2(hartree_fock, frozen=frozen)
            mp2_energy, mp2_t2 = mp2_solver.kernel()

        # run CCSD if requested
        ccsd_energy = None
        ccsd_t1 = None
        ccsd_t2 = None
        if ccsd:
            ccsd_solver = cc.CCSD(
                hartree_fock,
                frozen=frozen,
            )
            ccsd_energy, ccsd_t1, ccsd_t2 = ccsd_solver.kernel()

        # run FCI if requested
        fci_energy = None
        fci_vec = None
        if fci:
            fci_energy, _, fci_vec, _, _ = cas.kernel()

        # compute dipole integrals
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
            atom=hartree_fock.mol.atom,
            basis=hartree_fock.mol.basis,
            spin=hartree_fock.mol.spin,
            symmetry=hartree_fock.mol.symmetry or None,
            norb=norb,
            nelec=(n_alpha, n_beta),
            mo_coeff=hartree_fock.mo_coeff,
            mo_occ=hartree_fock.mo_occ,
            active_space=active_space,
            core_energy=core_energy,
            one_body_integrals=one_body_tensor,
            two_body_integrals=two_body_integrals,
            hf_energy=hf_energy,
            mp2_energy=mp2_energy,
            mp2_t2=mp2_t2,
            ccsd_energy=ccsd_energy,
            ccsd_t1=ccsd_t1,
            ccsd_t2=ccsd_t2,
            fci_energy=fci_energy,
            fci_vec=fci_vec,
            dipole_integrals=dipole_integrals,
            orbital_symmetries=orbsym,
        )

    @staticmethod
    def from_mole(
        molecule: gto.Mole,
        active_space: Iterable[int] | None = None,
        mp2: bool = False,
        ccsd: bool = False,
        fci: bool = False,
        scf_func=scf.RHF,
    ) -> "MolecularData":
        """Initialize a MolecularData object from a pySCF molecule.

        Args:
            molecule: The molecule.
            active_space: An optional list of orbitals to use for the active space.
            mp2: Whether to calculate and store the MP2 energy and t2 amplitudes.
            ccsd: Whether to calculate and store the CCSD energy, t1, and t2 amplitudes.
            fci: Whether to calculate and store the FCI energy and state vector.
            scf_func: The pySCF SCF function to use for the Hartree-Fock calculation.
        """
        hartree_fock = scf_func(molecule)
        hartree_fock.run()
        return MolecularData.from_scf(
            hartree_fock, active_space=active_space, mp2=mp2, ccsd=ccsd, fci=fci
        )
