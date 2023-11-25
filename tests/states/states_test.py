# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test states."""

from __future__ import annotations

import itertools

import numpy as np
import pyscf
import pytest

import ffsim


def test_slater_determinant():
    """Test Slater determinant."""
    norb = 5
    nelec = ffsim.testing.random_nelec(norb)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec)
    occ_a, occ_b = occupied_orbitals

    one_body_tensor = ffsim.random.random_hermitian(norb)
    eigs, orbital_rotation = np.linalg.eigh(one_body_tensor)
    eig = sum(eigs[occ_a]) + sum(eigs[occ_b])
    state = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    hamiltonian = ffsim.contract.one_body_linop(one_body_tensor, norb=norb, nelec=nelec)
    np.testing.assert_allclose(hamiltonian @ state, eig * state)


def test_hartree_fock_state():
    """Test Hartree-Fock state."""
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["H", (0, 0, 1.8)]],
        basis="sto-6g",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock_energy = hartree_fock.kernel()

    mol_data = ffsim.MolecularData.from_scf(hartree_fock)

    vec = ffsim.hartree_fock_state(mol_data.norb, mol_data.nelec)
    hamiltonian = ffsim.linear_operator(
        mol_data.hamiltonian, norb=mol_data.norb, nelec=mol_data.nelec
    )
    energy = np.vdot(vec, hamiltonian @ vec)

    np.testing.assert_allclose(energy, hartree_fock_energy)


@pytest.mark.parametrize(
    "norb, occupied_orbitals, spin_summed",
    [
        (4, ([0, 1], [0, 1]), True),
        (4, ([0, 1], [0, 1]), False),
        (3, ([0], [1, 2]), True),
        (3, ([0], [1, 2]), False),
        (2, ([], [0]), True),
        (2, ([], [0]), False),
    ],
)
def test_slater_determinant_one_rdm(
    norb: int, occupied_orbitals: tuple[list[int], list[int]], spin_summed: bool
):
    """Test Slater determinant 1-RDM."""
    occ_a, occ_b = occupied_orbitals
    nelec = len(occ_a), len(occ_b)

    rng = np.random.default_rng()
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)

    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )
    rdm = ffsim.slater_determinant_rdm(
        norb,
        occupied_orbitals,
        orbital_rotation=orbital_rotation,
        spin_summed=spin_summed,
    )
    expected = ffsim.rdm(vec, norb, nelec, spin_summed=spin_summed)

    np.testing.assert_allclose(rdm, expected, atol=1e-12)


def test_indices_to_strings():
    """Test converting statevector indices to strings."""
    norb = 3
    nelec = (2, 1)

    dim = ffsim.dim(norb, nelec)
    strings = ffsim.indices_to_strings(range(dim), norb, nelec)
    assert strings == [
        "011001",
        "011010",
        "011100",
        "101001",
        "101010",
        "101100",
        "110001",
        "110010",
        "110100",
    ]


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (3, (1, 2)),
        (2, (0, 1)),
        (1, (0, 0)),
    ],
)
def test_rdm_1_spin_summed(norb: int, nelec: tuple[int, int]):
    """Test computing spin-summed 1-RDM."""
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    rdm = ffsim.rdm(vec, norb, nelec)
    expected = _rdm1_spin_summed(vec, norb, nelec)
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (3, (1, 2)),
        (2, (0, 1)),
        (1, (0, 0)),
    ],
)
def test_rdm_1(norb: int, nelec: tuple[int, int]):
    """Test computing 1-RDM."""
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    rdm = ffsim.rdm(vec, norb, nelec, spin_summed=False)
    expected = _rdm1(vec, norb, nelec)
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (3, (1, 2)),
        (2, (0, 1)),
        (1, (0, 0)),
    ],
)
def test_rdm_2_spin_summed_reordered(norb: int, nelec: tuple[int, int]):
    """Test computing spin-summed reordered 2-RDM."""
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    rdm = ffsim.rdm(vec, norb, nelec, rank=2)
    expected = _rdm2_spin_summed_reordered(vec, norb, nelec)
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (3, (1, 2)),
        (2, (0, 1)),
        (1, (0, 0)),
    ],
)
def test_rdm_2_reordered(norb: int, nelec: tuple[int, int]):
    """Test computing reordered 2-RDM."""
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    rdm = ffsim.rdm(vec, norb, nelec, rank=2, spin_summed=False)
    expected = _rdm2_reordered(vec, norb, nelec)
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (3, (1, 2)),
        (2, (0, 1)),
        (1, (0, 0)),
    ],
)
def test_rdm_2_spin_summed(norb: int, nelec: tuple[int, int]):
    """Test computing spin-summed 2-RDM."""
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    rdm = ffsim.rdm(vec, norb, nelec, rank=2, reordered=False)
    expected = _rdm2_spin_summed(vec, norb, nelec)
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (4, (2, 2)),
        (3, (1, 2)),
        (2, (0, 1)),
        (1, (0, 0)),
    ],
)
def test_rdm_2(norb: int, nelec: tuple[int, int]):
    """Test computing 2-RDM."""
    rng = np.random.default_rng()
    vec = ffsim.random.random_statevector(ffsim.dim(norb, nelec), seed=rng)

    rdm = ffsim.rdm(vec, norb, nelec, rank=2, spin_summed=False, reordered=False)
    expected = _rdm2(vec, norb, nelec)
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


def _rdm1_spin_summed(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute spin-summed 1-RDM directly from its definition."""
    rdm = np.zeros((norb, norb), dtype=complex)
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(j)): 1,
                (ffsim.cre_b(i), ffsim.des_b(j)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[i, j] = val
        rdm[j, i] = val.conjugate()
    return rdm


def _rdm1(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute 1-RDM directly from its definition."""
    rdm = np.zeros((2 * norb, 2 * norb), dtype=complex)
    for i, j in itertools.combinations_with_replacement(range(norb), 2):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(i), ffsim.des_a(j)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[i, j] = val
        rdm[j, i] = val.conjugate()
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(i), ffsim.des_b(j)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + i, norb + j] = val
        rdm[norb + j, norb + i] = val.conjugate()
    return rdm


def _rdm2_spin_summed_reordered(
    vec: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    """Compute spin-summed reordered 2-RDM directly from its definition."""
    rdm = np.zeros((norb, norb, norb, norb), dtype=complex)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_a(q)): 1,
                (ffsim.cre_a(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_a(q)): 1,
                (ffsim.cre_b(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_b(q)): 1,
                (ffsim.cre_b(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_b(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, r, s] = val
    return rdm


def _rdm2_spin_summed(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute spin-summed 2-RDM directly from its definition."""
    rdm = np.zeros((norb, norb, norb, norb), dtype=complex)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.des_a(q), ffsim.cre_a(r), ffsim.des_a(s)): 1,
                (ffsim.cre_a(p), ffsim.des_a(q), ffsim.cre_b(r), ffsim.des_b(s)): 1,
                (ffsim.cre_b(p), ffsim.des_b(q), ffsim.cre_a(r), ffsim.des_a(s)): 1,
                (ffsim.cre_b(p), ffsim.des_b(q), ffsim.cre_b(r), ffsim.des_b(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, r, s] = val
    return rdm


def _rdm2_reordered(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute reordered 2-RDM directly from its definition."""
    rdm = np.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=complex)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_a(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, r, s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_a(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, norb + r, norb + s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(p), ffsim.cre_a(r), ffsim.des_a(s), ffsim.des_b(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + p, norb + q, r, s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(p), ffsim.cre_b(r), ffsim.des_b(s), ffsim.des_b(q)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + p, norb + q, norb + r, norb + s] = val
    return rdm


def _rdm2(vec: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """Compute 2-RDM directly from its definition."""
    rdm = np.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=complex)
    for p, q, r, s in itertools.product(range(norb), repeat=4):
        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.des_a(q), ffsim.cre_a(r), ffsim.des_a(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, r, s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_a(p), ffsim.des_a(q), ffsim.cre_b(r), ffsim.des_b(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[p, q, norb + r, norb + s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(p), ffsim.des_b(q), ffsim.cre_a(r), ffsim.des_a(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + p, norb + q, r, s] = val

        op = ffsim.FermionOperator(
            {
                (ffsim.cre_b(p), ffsim.des_b(q), ffsim.cre_b(r), ffsim.des_b(s)): 1,
            }
        )
        linop = ffsim.linear_operator(op, norb, nelec)
        val = np.vdot(vec, linop @ vec)
        rdm[norb + p, norb + q, norb + r, norb + s] = val
    return rdm
