# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for UCCSD ansatz."""

import dataclasses

import numpy as np
import pyscf
import pyscf.cc
import pytest

import ffsim

RNG = np.random.default_rng(241296678209900701723124712348955121818)


RESTRICTED_CLASSES = [
    (ffsim.UCCSDOpRestrictedReal, ffsim.random.random_uccsd_op_restricted_real),
    (ffsim.UCCSDOpRestricted, ffsim.random.random_uccsd_op_restricted),
]
UNRESTRICTED_CLASSES = [
    (ffsim.UCCSDOpUnrestrictedReal, ffsim.random.random_uccsd_op_unrestricted_real),
    (ffsim.UCCSDOpUnrestricted, ffsim.random.random_uccsd_op_unrestricted),
]
SPINLESS_CLASSES = [
    (ffsim.UCCSDOpSpinlessReal, ffsim.random.random_uccsd_op_spinless_real),
    (ffsim.UCCSDOpSpinless, ffsim.random.random_uccsd_op_spinless),
]


@pytest.mark.parametrize("cls, random_op", RESTRICTED_CLASSES)
def test_uccsd_restricted_norb(cls, random_op):
    """Test norb property."""
    norb = 5
    nocc = 3
    operator = random_op(norb, nocc, seed=RNG)
    assert isinstance(operator, cls)
    assert operator.norb == norb


@pytest.mark.parametrize("cls, random_op", RESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_restricted_n_params(cls, random_op, with_final_orbital_rotation):
    """Test computing number of parameters."""
    norb = 5
    nocc = 3
    operator = random_op(
        norb,
        nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    actual = cls.n_params(
        norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
    )
    expected = len(operator.to_parameters())
    assert actual == expected


@pytest.mark.parametrize("cls, random_op", RESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_restricted_parameters_roundtrip(
    cls, random_op, with_final_orbital_rotation
):
    """Test parameters roundtrip."""
    norb = 5
    nocc = 3
    operator = random_op(
        norb,
        nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    roundtripped = cls.from_parameters(
        operator.to_parameters(),
        norb=norb,
        nocc=nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    assert ffsim.approx_eq(roundtripped, operator)


@pytest.mark.parametrize("cls, random_op", RESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_restricted_from_parameters_roundtrip(
    cls, random_op, with_final_orbital_rotation
):
    """Test parameters roundtrip starting from a random parameter vector."""
    norb = 5
    nocc = 3
    n_params = cls.n_params(
        norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
    )
    params = RNG.standard_normal(n_params)
    operator = cls.from_parameters(
        params,
        norb=norb,
        nocc=nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    np.testing.assert_array_equal(operator.t2, operator.t2.transpose(1, 0, 3, 2))
    # The amplitude parameters roundtrip exactly. The final orbital rotation
    # parameters need not, because the matrix logarithm picks a principal branch.
    n_amplitudes = n_params - with_final_orbital_rotation * norb**2
    roundtripped_params = operator.to_parameters()
    np.testing.assert_allclose(
        roundtripped_params[:n_amplitudes], params[:n_amplitudes]
    )
    roundtripped = cls.from_parameters(
        roundtripped_params,
        norb=norb,
        nocc=nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    assert ffsim.approx_eq(roundtripped, operator)


@pytest.mark.parametrize("cls, random_op", RESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_restricted_approx_eq(cls, random_op, with_final_orbital_rotation):
    """Test approximate equality."""
    norb = 5
    nocc = 3
    operator = random_op(
        norb,
        nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    roundtripped = cls.from_parameters(
        operator.to_parameters(),
        norb=norb,
        nocc=nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    assert ffsim.approx_eq(operator, roundtripped)
    assert not ffsim.approx_eq(
        operator, dataclasses.replace(operator, t1=2 * operator.t1)
    )
    assert not ffsim.approx_eq(
        operator, dataclasses.replace(operator, t2=2 * operator.t2)
    )


@pytest.mark.parametrize("cls, random_op", RESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_restricted_apply_unitary(cls, random_op, with_final_orbital_rotation):
    """Test unitary."""
    norb = 5
    nocc = 3
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, (nocc, nocc)), seed=RNG)
    operator = random_op(
        norb,
        nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    result = ffsim.apply_unitary(vec, operator, norb=norb, nelec=(nocc, nocc))
    np.testing.assert_allclose(np.linalg.norm(result), 1.0)


def test_uccsd_restricted_validate():
    """Test validation of restricted UCCSD operators."""
    norb = 5
    nocc = 3
    nvrt = norb - nocc
    t1 = RNG.standard_normal((nocc, nvrt))
    t2 = RNG.standard_normal((nocc, nocc, nvrt, nvrt))
    for cls in [ffsim.UCCSDOpRestrictedReal, ffsim.UCCSDOpRestricted]:
        with pytest.raises(ValueError, match="t2 amplitudes were not symmetric"):
            cls(t1=t1, t2=t2)
        # Validation can be skipped
        cls(t1=t1, t2=t2, validate=False)
    operator = ffsim.random.random_uccsd_op_restricted(norb, nocc, seed=RNG)
    with pytest.raises(
        TypeError, match="UCCSDOpRestrictedReal only accepts real-valued t1"
    ):
        ffsim.UCCSDOpRestrictedReal(t1=operator.t1, t2=operator.t2.real)
    with pytest.raises(
        TypeError, match="UCCSDOpRestrictedReal only accepts real-valued t2"
    ):
        ffsim.UCCSDOpRestrictedReal(t1=operator.t1.real, t2=operator.t2)


@pytest.mark.parametrize("cls", [cls for cls, _ in RESTRICTED_CLASSES])
def test_uccsd_restricted_energy(cls):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, -0.5)], ["N", (0, 0, 0.5)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()

    # Get molecular data and molecular Hamiltonian
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    norb = mol_data.norb
    nelec = mol_data.nelec
    assert norb == 8
    assert nelec == (5, 5)
    mol_hamiltonian = mol_data.hamiltonian

    # Construct UCCSD operator
    operator = cls(t1=ccsd.t1, t2=ccsd.t2)

    # Construct the Hartree-Fock state to use as the reference state
    n_alpha, n_beta = nelec
    reference_state = ffsim.slater_determinant(
        norb=norb, occupied_orbitals=(range(n_alpha), range(n_beta))
    )

    # Apply the operator to the reference state
    ansatz_state = ffsim.apply_unitary(
        reference_state, operator, norb=norb, nelec=nelec
    )

    # Compute the energy ⟨ψ|H|ψ⟩ of the ansatz state
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))
    np.testing.assert_allclose(energy, -108.594284)


@pytest.mark.parametrize("cls, random_op", UNRESTRICTED_CLASSES)
def test_uccsd_unrestricted_norb(cls, random_op):
    """Test norb property."""
    norb = 5
    nelec = (3, 2)
    operator = random_op(norb, nelec, seed=RNG)
    assert isinstance(operator, cls)
    assert operator.norb == norb


@pytest.mark.parametrize("cls, random_op", UNRESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_unrestricted_n_params(cls, random_op, with_final_orbital_rotation):
    """Test computing number of parameters."""
    norb = 5
    nelec = (3, 2)
    # t1_a and t1_b have 3 * 2 + 2 * 3 parameters, t2_aa has comb(3, 2) * comb(2, 2)
    # parameters, t2_ab has 3 * 2 * 2 * 3 parameters, and t2_bb has
    # comb(2, 2) * comb(3, 2) parameters
    n_amplitudes = 3 * 2 + 2 * 3 + 3 * 1 + 3 * 2 * 2 * 3 + 1 * 3
    if cls is ffsim.UCCSDOpUnrestricted:
        n_amplitudes *= 2
    operator = random_op(
        norb,
        nelec,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    actual = cls.n_params(
        norb, nelec, with_final_orbital_rotation=with_final_orbital_rotation
    )
    assert actual == len(operator.to_parameters())
    assert actual == n_amplitudes + with_final_orbital_rotation * 2 * norb**2


@pytest.mark.parametrize("cls, random_op", UNRESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_unrestricted_parameters_roundtrip(
    cls, random_op, with_final_orbital_rotation
):
    """Test parameters roundtrip."""
    norb = 5
    nelec = (3, 2)
    operator = random_op(
        norb,
        nelec,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    roundtripped = cls.from_parameters(
        operator.to_parameters(),
        norb=norb,
        nelec=nelec,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    assert ffsim.approx_eq(roundtripped, operator)


@pytest.mark.parametrize("cls, random_op", UNRESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_unrestricted_from_parameters_roundtrip(
    cls, random_op, with_final_orbital_rotation
):
    """Test parameters roundtrip starting from a random parameter vector."""
    norb = 5
    nelec = (3, 2)
    n_params = cls.n_params(
        norb, nelec, with_final_orbital_rotation=with_final_orbital_rotation
    )
    params = RNG.standard_normal(n_params)
    operator = cls.from_parameters(
        params,
        norb=norb,
        nelec=nelec,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    t2_aa, _, t2_bb = operator.t2
    for t2 in [t2_aa, t2_bb]:
        np.testing.assert_array_equal(t2, -t2.transpose(1, 0, 2, 3))
        np.testing.assert_array_equal(t2, -t2.transpose(0, 1, 3, 2))
    # The amplitude parameters roundtrip exactly. The final orbital rotation
    # parameters need not, because the matrix logarithm picks a principal branch.
    n_amplitudes = n_params - with_final_orbital_rotation * 2 * norb**2
    roundtripped_params = operator.to_parameters()
    np.testing.assert_allclose(
        roundtripped_params[:n_amplitudes], params[:n_amplitudes]
    )
    roundtripped = cls.from_parameters(
        roundtripped_params,
        norb=norb,
        nelec=nelec,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    assert ffsim.approx_eq(roundtripped, operator)


def test_uccsd_unrestricted_validate():
    """Test validation of unrestricted UCCSD operators."""
    norb = 5
    nelec = (3, 2)
    operator = ffsim.random.random_uccsd_op_unrestricted_real(norb, nelec, seed=RNG)
    t2_aa, t2_ab, t2_bb = operator.t2
    bad_t2_aa = RNG.standard_normal(t2_aa.shape)
    bad_t2_bb = RNG.standard_normal(t2_bb.shape)
    for cls in [ffsim.UCCSDOpUnrestrictedReal, ffsim.UCCSDOpUnrestricted]:
        with pytest.raises(ValueError, match="t2_aa amplitudes were not antisymmetric"):
            cls(t1=operator.t1, t2=(bad_t2_aa, t2_ab, t2_bb))
        with pytest.raises(ValueError, match="t2_bb amplitudes were not antisymmetric"):
            cls(t1=operator.t1, t2=(t2_aa, t2_ab, bad_t2_bb))
        # Validation can be skipped
        cls(t1=operator.t1, t2=(bad_t2_aa, t2_ab, bad_t2_bb), validate=False)
    complex_operator = ffsim.random.random_uccsd_op_unrestricted(norb, nelec, seed=RNG)
    t1_a, t1_b = complex_operator.t1
    t2_aa, t2_ab, t2_bb = complex_operator.t2
    with pytest.raises(
        TypeError, match="UCCSDOpUnrestrictedReal only accepts real-valued t1"
    ):
        ffsim.UCCSDOpUnrestrictedReal(
            t1=(t1_a, t1_b), t2=(t2_aa.real, t2_ab.real, t2_bb.real)
        )
    with pytest.raises(
        TypeError, match="UCCSDOpUnrestrictedReal only accepts real-valued t2"
    ):
        ffsim.UCCSDOpUnrestrictedReal(
            t1=(t1_a.real, t1_b.real), t2=(t2_aa, t2_ab, t2_bb)
        )


@pytest.mark.parametrize("cls, random_op", UNRESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_unrestricted_approx_eq(cls, random_op, with_final_orbital_rotation):
    """Test approximate equality."""
    norb = 5
    nelec = (3, 2)
    operator = random_op(
        norb,
        nelec,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    roundtripped = cls.from_parameters(
        operator.to_parameters(),
        norb=norb,
        nelec=nelec,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    assert ffsim.approx_eq(operator, roundtripped)
    assert not ffsim.approx_eq(
        operator,
        dataclasses.replace(operator, t1=(2 * operator.t1[0], operator.t1[1])),
    )
    assert not ffsim.approx_eq(
        operator,
        dataclasses.replace(
            operator, t2=(2 * operator.t2[0], operator.t2[1], operator.t2[2])
        ),
    )


@pytest.mark.parametrize("cls, random_op", UNRESTRICTED_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_unrestricted_apply_unitary(cls, random_op, with_final_orbital_rotation):
    """Test unitary."""
    norb = 5
    nelec = (3, 2)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    operator = random_op(
        norb,
        nelec,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    result = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nelec)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0)


@pytest.mark.parametrize("cls", [cls for cls, _ in UNRESTRICTED_CLASSES])
def test_uccsd_unrestricted_energy(cls):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[("H", (0, 0, i)) for i in range(5)],
        basis="sto-6g",
        spin=1,
        symmetry="Dooh",
        verbose=0,
    )
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()

    # Get molecular data and molecular Hamiltonian
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb
    nelec = mol_data.nelec
    assert norb == 5
    assert nelec == (3, 2)
    mol_hamiltonian = mol_data.hamiltonian

    # Construct UCCSD operator
    operator = cls(t1=ccsd.t1, t2=ccsd.t2)

    # Construct the Hartree-Fock state to use as the reference state
    n_alpha, n_beta = nelec
    reference_state = ffsim.slater_determinant(
        norb=norb, occupied_orbitals=(range(n_alpha), range(n_beta))
    )

    # Apply the operator to the reference state
    ansatz_state = ffsim.apply_unitary(
        reference_state, operator, norb=norb, nelec=nelec
    )

    # Compute the energy ⟨ψ|H|ψ⟩ of the ansatz state
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    energy = np.real(np.vdot(ansatz_state, hamiltonian @ ansatz_state))
    np.testing.assert_allclose(energy, -2.672171)


@pytest.mark.parametrize("cls, random_op", SPINLESS_CLASSES)
def test_uccsd_spinless_norb(cls, random_op):
    """Test norb property."""
    norb = 5
    nocc = 3
    operator = random_op(norb, nocc, seed=RNG)
    assert isinstance(operator, cls)
    assert operator.norb == norb


@pytest.mark.parametrize("cls, random_op", SPINLESS_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_spinless_n_params(cls, random_op, with_final_orbital_rotation):
    """Test computing number of parameters."""
    norb = 5
    nocc = 3
    # t1 has 3 * 2 parameters and t2 has comb(3, 2) * comb(2, 2) parameters
    n_amplitudes = 3 * 2 + 3 * 1
    if cls is ffsim.UCCSDOpSpinless:
        n_amplitudes *= 2
    operator = random_op(
        norb,
        nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    actual = cls.n_params(
        norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
    )
    assert actual == len(operator.to_parameters())
    assert actual == n_amplitudes + with_final_orbital_rotation * norb**2


@pytest.mark.parametrize("cls, random_op", SPINLESS_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_spinless_parameters_roundtrip(
    cls, random_op, with_final_orbital_rotation
):
    """Test parameters roundtrip."""
    norb = 5
    nocc = 3
    operator = random_op(
        norb,
        nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    roundtripped = cls.from_parameters(
        operator.to_parameters(),
        norb=norb,
        nocc=nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    assert ffsim.approx_eq(roundtripped, operator)


@pytest.mark.parametrize("cls, random_op", SPINLESS_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_spinless_from_parameters_roundtrip(
    cls, random_op, with_final_orbital_rotation
):
    """Test parameters roundtrip starting from a random parameter vector."""
    norb = 5
    nocc = 3
    n_params = cls.n_params(
        norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
    )
    params = RNG.standard_normal(n_params)
    operator = cls.from_parameters(
        params,
        norb=norb,
        nocc=nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    np.testing.assert_array_equal(operator.t2, -operator.t2.transpose(1, 0, 2, 3))
    np.testing.assert_array_equal(operator.t2, -operator.t2.transpose(0, 1, 3, 2))
    # The amplitude parameters roundtrip exactly. The final orbital rotation
    # parameters need not, because the matrix logarithm picks a principal branch.
    n_amplitudes = n_params - with_final_orbital_rotation * norb**2
    roundtripped_params = operator.to_parameters()
    np.testing.assert_allclose(
        roundtripped_params[:n_amplitudes], params[:n_amplitudes]
    )
    roundtripped = cls.from_parameters(
        roundtripped_params,
        norb=norb,
        nocc=nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    assert ffsim.approx_eq(roundtripped, operator)


@pytest.mark.parametrize("cls, random_op", SPINLESS_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_spinless_approx_eq(cls, random_op, with_final_orbital_rotation):
    """Test approximate equality."""
    norb = 5
    nocc = 3
    operator = random_op(
        norb,
        nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    roundtripped = cls.from_parameters(
        operator.to_parameters(),
        norb=norb,
        nocc=nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    assert ffsim.approx_eq(operator, roundtripped)
    assert not ffsim.approx_eq(
        operator, dataclasses.replace(operator, t1=2 * operator.t1)
    )
    assert not ffsim.approx_eq(
        operator, dataclasses.replace(operator, t2=2 * operator.t2)
    )


@pytest.mark.parametrize("cls, random_op", SPINLESS_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_spinless_apply_unitary(cls, random_op, with_final_orbital_rotation):
    """Test unitary."""
    norb = 5
    nocc = 3
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nocc), seed=RNG)
    operator = random_op(
        norb,
        nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    result = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nocc)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0)


def test_uccsd_spinless_validate():
    """Test validation of spinless UCCSD operators."""
    norb = 5
    nocc = 3
    nvrt = norb - nocc
    t1 = RNG.standard_normal((nocc, nvrt))
    t2 = RNG.standard_normal((nocc, nocc, nvrt, nvrt))
    for cls in [ffsim.UCCSDOpSpinlessReal, ffsim.UCCSDOpSpinless]:
        with pytest.raises(ValueError, match="antisymmetric"):
            cls(t1=t1, t2=t2)
        # Validation can be skipped
        cls(t1=t1, t2=t2, validate=False)
    operator = ffsim.random.random_uccsd_op_spinless(norb, nocc, seed=RNG)
    with pytest.raises(TypeError, match="t1"):
        ffsim.UCCSDOpSpinlessReal(t1=operator.t1, t2=operator.t2.real)
    with pytest.raises(TypeError, match="t2"):
        ffsim.UCCSDOpSpinlessReal(t1=operator.t1.real, t2=operator.t2)


@pytest.mark.parametrize("cls, random_op", SPINLESS_CLASSES)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_uccsd_spinless_matches_unrestricted(
    cls, random_op, with_final_orbital_rotation
):
    """Test spinless UCCSD agrees with unrestricted UCCSD with no beta electrons."""
    norb = 5
    nocc = 3
    nvrt = norb - nocc
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nocc), seed=RNG)
    operator = random_op(
        norb,
        nocc,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    unrestricted_cls = (
        ffsim.UCCSDOpUnrestrictedReal
        if cls is ffsim.UCCSDOpSpinlessReal
        else ffsim.UCCSDOpUnrestricted
    )
    final_orbital_rotation = None
    if with_final_orbital_rotation:
        final_orbital_rotation = np.stack(
            [operator.final_orbital_rotation, np.eye(norb)]
        )
    unrestricted_op = unrestricted_cls(
        t1=(operator.t1, np.zeros((0, norb))),
        t2=(
            operator.t2,
            np.zeros((nocc, 0, nvrt, norb)),
            np.zeros((0, 0, norb, norb)),
        ),
        final_orbital_rotation=final_orbital_rotation,
    )
    actual = ffsim.apply_unitary(vec, operator, norb=norb, nelec=nocc)
    expected = ffsim.apply_unitary(vec, unrestricted_op, norb=norb, nelec=(nocc, 0))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
