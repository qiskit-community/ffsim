# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for molecular Hamiltonian."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pyscf
import pyscf.mcscf
import pytest
import scipy.sparse.linalg

import ffsim

RNG = np.random.default_rng(99421951268571780225542766582304428700)

NORB_NELEC = [
    (1, (0, 0)),
    (1, (0, 1)),
    (4, (2, 2)),
    (4, (2, 4)),
    (5, (3, 2)),
]
NORB_NELEC_SPINLESS = [
    (1, 0),
    (1, 1),
    (4, 2),
    (4, 4),
    (5, 3),
]


def _check_rotated(
    hamiltonian,
    rotated,
    orbital_rotation: np.ndarray | tuple[np.ndarray, np.ndarray],
    norb: int,
    nelec: Any,
):
    """Check that a rotated Hamiltonian definition and preserves expectation values."""
    if isinstance(orbital_rotation, tuple):
        mat_a, mat_b = orbital_rotation
        orbital_rotation_adjoint: np.ndarray | tuple[np.ndarray, np.ndarray] = (
            mat_a.T.conj(),
            mat_b.T.conj(),
        )
    else:
        orbital_rotation_adjoint = orbital_rotation.T.conj()

    linop = ffsim.linear_operator(hamiltonian, norb, nelec)
    linop_rotated = ffsim.linear_operator(rotated, norb, nelec)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    # test definition
    actual = linop_rotated @ vec
    expected = ffsim.apply_orbital_rotation(vec, orbital_rotation_adjoint, norb, nelec)
    expected = linop @ expected
    expected = ffsim.apply_orbital_rotation(expected, orbital_rotation, norb, nelec)
    np.testing.assert_allclose(actual, expected)

    # test expectation is preserved
    rotated_vec = ffsim.apply_orbital_rotation(vec, orbital_rotation, norb, nelec)
    original_expectation = np.vdot(vec, linop @ vec)
    rotated_expectation = np.vdot(rotated_vec, linop_rotated @ rotated_vec)
    np.testing.assert_allclose(original_expectation, rotated_expectation)


def test_linear_operator_water_molecule():
    """Test linear operator against PySCF FCI energy of a water molecule."""
    # Construct water molecule
    radius_1 = 0.958  # position for the first H atom
    radius_2 = 0.958  # position for the second H atom
    thetas_in_deg = 104.478  # bond angles.

    h1_x = radius_1
    h2_x = radius_2 * np.cos(np.pi / 180 * thetas_in_deg)
    h2_y = radius_2 * np.sin(np.pi / 180 * thetas_in_deg)

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[
            ["O", (0, 0, 0)],
            ["H", (h1_x, 0, 0)],
            ["H", (h2_x, h2_y, 0)],
        ],
        basis="sto-6g",
        spin=0,
        charge=0,
        symmetry="c2v",
    )
    hartree_fock = pyscf.scf.RHF(mol)
    hartree_fock.kernel()

    # Define active space
    active_space = [1, 2, 4, 5, 6]

    # Compute FCI energy using PySCF
    norb = len(active_space)
    n_electrons = int(np.sum(hartree_fock.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    cas = pyscf.mcscf.CASCI(hartree_fock, ncas=norb, nelecas=nelec)
    mo = cas.sort_mo(active_space, base=0)
    energy_fci = cas.kernel(mo)[0]

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_scf(hartree_fock, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # compute FCI energy from molecular Hamiltonian
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    eigs, _ = scipy.sparse.linalg.eigsh(hamiltonian, k=1, which="SA")
    eig = eigs[0]

    # Check that they match
    np.testing.assert_allclose(eig, energy_fci)


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
def test_diag_and_trace(norb: int, nelec: tuple[int, int]):
    """Test computing diagonal and trace."""
    # TODO remove dtype=float once complex is supported
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    hamiltonian_dense = linop @ np.eye(ffsim.dim(norb, nelec))
    diag = ffsim.diag(hamiltonian, norb=norb, nelec=nelec)
    np.testing.assert_allclose(diag, np.diag(hamiltonian_dense))
    np.testing.assert_allclose(
        ffsim.trace(hamiltonian, norb=norb, nelec=nelec), np.sum(diag)
    )


def test_diag_complex_raises():
    """Test that computing the diagonal of a complex Hamiltonian raises an error."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian(4, seed=RNG, dtype=complex)
    with pytest.raises(NotImplementedError, match="complex"):
        _ = ffsim.diag(hamiltonian, norb=4, nelec=(2, 2))


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
def test_fermion_operator(norb: int, nelec: tuple[int, int]):
    """Test FermionOperator."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb=norb, seed=RNG)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    op = ffsim.fermion_operator(mol_hamiltonian)
    linop = ffsim.linear_operator(op, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
@pytest.mark.parametrize("dtype", [float, complex])
def test_rotated(norb: int, nelec: tuple[int, int], dtype):
    """Test rotating orbitals."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian(
        norb=norb, seed=RNG, dtype=dtype
    )
    if dtype is complex:
        orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    else:
        orbital_rotation = ffsim.random.random_orthogonal(norb, seed=RNG)
    _check_rotated(
        hamiltonian,
        hamiltonian.rotated(orbital_rotation),
        orbital_rotation,
        norb,
        nelec,
    )


def test_approx_eq():
    """Test approximate equality."""
    norb = 4
    mol_ham_1 = ffsim.random.random_molecular_hamiltonian(norb=norb, seed=RNG)
    mol_ham_2 = dataclasses.replace(
        mol_ham_1,
        one_body_tensor=mol_ham_1.one_body_tensor + 1e-7,
        two_body_tensor=mol_ham_1.two_body_tensor + 1e-7,
    )
    assert ffsim.approx_eq(mol_ham_1, mol_ham_2, rtol=0, atol=1e-6)
    assert not ffsim.approx_eq(mol_ham_1, mol_ham_2, rtol=0, atol=1e-8)


@pytest.mark.parametrize("norb", range(1, 5))
def test_from_fermion_operator_roundtrip(norb: int):
    """Test converting fermion operator to molecular Hamiltonian."""
    mol_ham = ffsim.random.random_molecular_hamiltonian(norb=norb, seed=RNG)
    roundtripped = ffsim.MolecularHamiltonian.from_fermion_operator(
        ffsim.fermion_operator(mol_ham)
    )
    assert ffsim.approx_eq(roundtripped, mol_ham, atol=0)


def test_from_fermion_operator_invalid():
    """Test converting fermion operator with invalid terms."""
    op = ffsim.FermionOperator({(ffsim.cre_a(3), ffsim.cre_b(2)): 1.0})
    with pytest.raises(ValueError, match="quadratic"):
        _ = ffsim.MolecularHamiltonian.from_fermion_operator(op)
    op = ffsim.FermionOperator(
        {(ffsim.cre_a(3), ffsim.cre_b(2), ffsim.cre_a(3), ffsim.cre_b(2)): 1.0}
    )
    with pytest.raises(ValueError, match="quartic"):
        _ = ffsim.MolecularHamiltonian.from_fermion_operator(op)
    op = ffsim.FermionOperator({(ffsim.cre_a(3),): 1.0})
    with pytest.raises(ValueError, match="term"):
        _ = ffsim.MolecularHamiltonian.from_fermion_operator(op)


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
def test_linear_operator_spinless(norb: int, nelec: tuple[int, int]):
    """Test linear operator for MolecularHamiltonianSpinless."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb=norb, seed=RNG)
    hamiltonian_spinless = mol_hamiltonian.to_spinless()

    # the spinless tensor accessors agree with the converted Hamiltonian
    np.testing.assert_allclose(
        mol_hamiltonian.one_body_tensor_spinless, hamiltonian_spinless.one_body_tensor
    )
    np.testing.assert_allclose(
        mol_hamiltonian.two_body_tensor_spinless, hamiltonian_spinless.two_body_tensor
    )

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    result = ffsim.linear_operator(mol_hamiltonian, norb, nelec) @ vec
    linop_spinless = ffsim.linear_operator(hamiltonian_spinless, 2 * norb, sum(nelec))
    result_spinless = linop_spinless @ ffsim.spinful_to_spinless_vec(vec, norb, nelec)

    np.testing.assert_allclose(
        result_spinless, ffsim.spinful_to_spinless_vec(result, norb, nelec)
    )


@pytest.mark.parametrize("norb, nelec", NORB_NELEC_SPINLESS)
def test_diag_and_trace_spinless(norb: int, nelec: int):
    """Test computing diagonal and trace for MolecularHamiltonianSpinless."""
    # TODO remove dtype=float once complex is supported
    hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(
        norb, seed=RNG, dtype=float
    )
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    hamiltonian_dense = linop @ np.eye(ffsim.dim(norb, nelec))
    diag = ffsim.diag(hamiltonian, norb=norb, nelec=nelec)
    np.testing.assert_allclose(diag, np.diag(hamiltonian_dense))
    np.testing.assert_allclose(
        ffsim.trace(hamiltonian, norb=norb, nelec=nelec), np.sum(diag)
    )


def test_diag_complex_raises_spinless():
    """Test that computing the diagonal of a complex Hamiltonian raises an error."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(
        4, seed=RNG, dtype=complex
    )
    with pytest.raises(NotImplementedError, match="complex"):
        _ = ffsim.diag(hamiltonian, norb=4, nelec=2)


@pytest.mark.parametrize("norb, nelec", NORB_NELEC_SPINLESS)
def test_fermion_operator_spinless(norb: int, nelec: int):
    """Test FermionOperator for MolecularHamiltonianSpinless."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(
        norb=norb, seed=RNG
    )
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    op = ffsim.fermion_operator(mol_hamiltonian)
    linop = ffsim.linear_operator(op, norb, nelec)
    expected_linop = ffsim.linear_operator(mol_hamiltonian, norb, nelec)

    actual = linop @ vec
    expected = expected_linop @ vec
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("norb", range(1, 5))
def test_from_fermion_operator_roundtrip_spinless(norb: int):
    """Test converting fermion operator to spinless molecular Hamiltonian."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(norb, seed=RNG)
    roundtripped = ffsim.MolecularHamiltonianSpinless.from_fermion_operator(
        ffsim.fermion_operator(hamiltonian)
    )
    assert ffsim.approx_eq(roundtripped, hamiltonian, atol=0)


def test_from_fermion_operator_invalid_spinless():
    """Test converting fermion operator with invalid terms."""
    op = ffsim.FermionOperator({(ffsim.cre_a(3), ffsim.cre_b(2)): 1.0})
    with pytest.raises(ValueError, match="quadratic"):
        _ = ffsim.MolecularHamiltonianSpinless.from_fermion_operator(op)
    op = ffsim.FermionOperator(
        {(ffsim.cre_a(3), ffsim.cre_b(2), ffsim.cre_a(3), ffsim.cre_b(2)): 1.0}
    )
    with pytest.raises(ValueError, match="quartic"):
        _ = ffsim.MolecularHamiltonianSpinless.from_fermion_operator(op)
    op = ffsim.FermionOperator({(ffsim.cre_a(3),): 1.0})
    with pytest.raises(ValueError, match="term"):
        _ = ffsim.MolecularHamiltonianSpinless.from_fermion_operator(op)


@pytest.mark.parametrize("norb, nelec", NORB_NELEC_SPINLESS)
@pytest.mark.parametrize("dtype", [float, complex])
def test_rotated_spinless(norb: int, nelec: int, dtype):
    """Test rotating orbitals for MolecularHamiltonianSpinless."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(
        norb=norb, seed=RNG, dtype=dtype
    )
    if dtype is complex:
        orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    else:
        orbital_rotation = ffsim.random.random_orthogonal(norb, seed=RNG)
    _check_rotated(
        hamiltonian,
        hamiltonian.rotated(orbital_rotation),
        orbital_rotation,
        norb,
        nelec,
    )


def test_approx_eq_spinless():
    """Test approximate equality for MolecularHamiltonianSpinless."""
    norb = 4
    mol_ham_1 = ffsim.random.random_molecular_hamiltonian_spinless(norb=norb, seed=RNG)
    mol_ham_2 = dataclasses.replace(
        mol_ham_1,
        one_body_tensor=mol_ham_1.one_body_tensor + 1e-7,
        two_body_tensor=mol_ham_1.two_body_tensor + 1e-7,
    )
    assert ffsim.approx_eq(mol_ham_1, mol_ham_2, rtol=0, atol=1e-6)
    assert not ffsim.approx_eq(mol_ham_1, mol_ham_2, rtol=0, atol=1e-8)


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
@pytest.mark.parametrize("dtype", [float, complex])
def test_linear_operator_unrestricted(norb: int, nelec: tuple[int, int], dtype):
    """Test linear operator for MolecularHamiltonianUnrestricted."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(
        norb, seed=RNG, dtype=dtype
    )
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    linop = ffsim.linear_operator(hamiltonian, norb, nelec)
    linop_ferm = ffsim.linear_operator(ffsim.fermion_operator(hamiltonian), norb, nelec)
    np.testing.assert_allclose(linop @ vec, linop_ferm @ vec)
    np.testing.assert_allclose(linop.adjoint() @ vec, linop_ferm.adjoint() @ vec)


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
def test_diag_and_trace_unrestricted(norb: int, nelec: tuple[int, int]):
    """Test computing diagonal and trace for MolecularHamiltonianUnrestricted."""
    # TODO remove dtype=float once complex is supported
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(
        norb, seed=RNG, dtype=float
    )
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    hamiltonian_dense = linop @ np.eye(ffsim.dim(norb, nelec))
    diag = ffsim.diag(hamiltonian, norb=norb, nelec=nelec)
    np.testing.assert_allclose(diag, np.diag(hamiltonian_dense))
    np.testing.assert_allclose(
        ffsim.trace(hamiltonian, norb=norb, nelec=nelec), np.sum(diag)
    )


def test_diag_complex_raises_unrestricted():
    """Test that computing the diagonal of a complex Hamiltonian raises an error."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(
        4, seed=RNG, dtype=complex
    )
    with pytest.raises(NotImplementedError, match="complex"):
        _ = ffsim.diag(hamiltonian, norb=4, nelec=(2, 2))


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
def test_fermion_operator_unrestricted(norb: int, nelec: tuple[int, int]):
    """Test FermionOperator for MolecularHamiltonianUnrestricted."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(norb, seed=RNG)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)

    op = ffsim.fermion_operator(hamiltonian)
    actual = ffsim.linear_operator(op, norb, nelec) @ vec
    expected = ffsim.linear_operator(hamiltonian, norb, nelec) @ vec
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
@pytest.mark.parametrize("dtype", [float, complex])
def test_rotated_unrestricted(norb: int, nelec: tuple[int, int], dtype):
    """Test rotating with a different orbital rotation for each spin sector."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(
        norb, seed=RNG, dtype=dtype
    )
    if dtype is complex:
        rotation_a = ffsim.random.random_unitary(norb, seed=RNG)
        rotation_b = ffsim.random.random_unitary(norb, seed=RNG)
    else:
        rotation_a = ffsim.random.random_orthogonal(norb, seed=RNG)
        rotation_b = ffsim.random.random_orthogonal(norb, seed=RNG)
    orbital_rotation = (rotation_a, rotation_b)
    _check_rotated(
        hamiltonian,
        hamiltonian.rotated(orbital_rotation),
        orbital_rotation,
        norb,
        nelec,
    )


@pytest.mark.parametrize("norb", [2, 4, 5])
def test_rotated_input_forms_unrestricted(norb: int):
    """Test the accepted forms of the orbital rotation argument."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(norb, seed=RNG)
    mat_a = ffsim.random.random_unitary(norb, seed=RNG)
    mat_b = ffsim.random.random_unitary(norb, seed=RNG)
    eye = np.eye(norb)

    # a single array applies the same rotation to both spin sectors
    assert ffsim.approx_eq(
        hamiltonian.rotated(mat_a), hamiltonian.rotated((mat_a, mat_a))
    )

    # a stacked array is equivalent to a pair
    assert ffsim.approx_eq(
        hamiltonian.rotated(np.stack([mat_a, mat_b])),
        hamiltonian.rotated((mat_a, mat_b)),
    )

    # None means no operation is applied to that spin sector
    assert ffsim.approx_eq(
        hamiltonian.rotated((mat_a, None)), hamiltonian.rotated((mat_a, eye))
    )
    assert ffsim.approx_eq(
        hamiltonian.rotated((None, mat_b)), hamiltonian.rotated((eye, mat_b))
    )
    assert ffsim.approx_eq(hamiltonian.rotated((None, None)), hamiltonian)


def test_approx_eq_unrestricted():
    """Test approximate equality for MolecularHamiltonianUnrestricted."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(4, seed=RNG)
    other = dataclasses.replace(
        hamiltonian,
        one_body_tensors=hamiltonian.one_body_tensors + 1e-7,
        two_body_tensors=hamiltonian.two_body_tensors + 1e-7,
    )
    assert ffsim.approx_eq(hamiltonian, other, rtol=0, atol=1e-6)
    assert not ffsim.approx_eq(hamiltonian, other, rtol=0, atol=1e-8)


@pytest.mark.parametrize("norb", range(1, 5))
def test_from_fermion_operator_roundtrip_unrestricted(norb: int):
    """Test converting fermion operator to unrestricted molecular Hamiltonian."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(norb, seed=RNG)
    roundtripped = ffsim.MolecularHamiltonianUnrestricted.from_fermion_operator(
        ffsim.fermion_operator(hamiltonian)
    )
    assert ffsim.approx_eq(roundtripped, hamiltonian, atol=0)


def test_from_fermion_operator_invalid_unrestricted():
    """Test converting fermion operator with invalid terms."""
    op = ffsim.FermionOperator({(ffsim.cre_a(3), ffsim.cre_b(2)): 1.0})
    with pytest.raises(ValueError, match="quadratic"):
        _ = ffsim.MolecularHamiltonianUnrestricted.from_fermion_operator(op)
    op = ffsim.FermionOperator(
        {(ffsim.cre_a(3), ffsim.cre_b(2), ffsim.cre_a(3), ffsim.cre_b(2)): 1.0}
    )
    with pytest.raises(ValueError, match="quartic"):
        _ = ffsim.MolecularHamiltonianUnrestricted.from_fermion_operator(op)
    op = ffsim.FermionOperator({(ffsim.cre_a(3),): 1.0})
    with pytest.raises(ValueError, match="term"):
        _ = ffsim.MolecularHamiltonianUnrestricted.from_fermion_operator(op)


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
@pytest.mark.parametrize("dtype", [float, complex])
def test_to_spinless_unrestricted(norb: int, nelec: tuple[int, int], dtype):
    """Test converting an unrestricted Hamiltonian to a spinless one."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_unrestricted(
        norb, seed=RNG, dtype=dtype
    )
    hamiltonian_spinless = hamiltonian.to_spinless()

    assert hamiltonian_spinless.norb == 2 * norb

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    result = ffsim.linear_operator(hamiltonian, norb, nelec) @ vec
    linop_spinless = ffsim.linear_operator(hamiltonian_spinless, 2 * norb, sum(nelec))
    result_spinless = linop_spinless @ ffsim.spinful_to_spinless_vec(vec, norb, nelec)

    np.testing.assert_allclose(
        result_spinless, ffsim.spinful_to_spinless_vec(result, norb, nelec)
    )


@pytest.mark.parametrize("norb, nelec", NORB_NELEC)
def test_to_unrestricted(norb: int, nelec: tuple[int, int]):
    """Test converting a molecular Hamiltonian to unrestricted form."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    actual = ffsim.linear_operator(mol_hamiltonian.to_unrestricted(), norb, nelec) @ vec
    expected = ffsim.linear_operator(mol_hamiltonian, norb, nelec) @ vec
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("norb", [1, 4, 5])
def test_rotated_matches_unrestricted(norb: int):
    """Test that rotating commutes with conversion for a spin-independent rotation."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    rotation = ffsim.random.random_unitary(norb, seed=RNG)
    assert ffsim.approx_eq(
        mol_hamiltonian.to_unrestricted().rotated(rotation),
        mol_hamiltonian.rotated(rotation).to_unrestricted(),
    )


@pytest.mark.parametrize("norb", [1, 4, 5])
def test_to_spinless_matches_unrestricted(norb: int):
    """Test that converting to spinless commutes with converting to unrestricted."""
    mol_hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    assert ffsim.approx_eq(
        mol_hamiltonian.to_unrestricted().to_spinless(),
        mol_hamiltonian.to_spinless(),
    )
