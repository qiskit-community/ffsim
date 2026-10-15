# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Slater determinant functions."""

from __future__ import annotations

import itertools
from typing import cast

import numpy as np
import pytest

import ffsim

RNG = np.random.default_rng(326051405585540571929556828082477638948)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_hartree_fock_state_spinful(norb: int, nelec: tuple[int, int]):
    """Test Hartree-Fock state."""
    vec = ffsim.hartree_fock_state(norb, nelec)
    dim = ffsim.dim(norb, nelec)
    assert vec.shape == (dim,)
    assert vec[0] == 1
    assert all(vec[1:] == 0)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_hartree_fock_state_spinless(norb: int, nelec: int):
    """Test Hartree-Fock state."""
    vec = ffsim.hartree_fock_state(norb, nelec)
    dim = ffsim.dim(norb, nelec)
    assert vec.shape == (dim,)
    assert vec[0] == 1
    assert all(vec[1:] == 0)


def test_slater_determinant_sign_spinful():
    """Test Slater determinant sign, spinful."""
    vec = ffsim.slater_determinant(4, ([0, 1], [0, 1]))
    assert vec[0] == 1

    vec = ffsim.slater_determinant(4, ([0, 1], [1, 0]))
    assert vec[0] == -1

    vec = ffsim.slater_determinant(4, ([1, 0], [1, 0]))
    assert vec[0] == 1


def test_slater_determinant_sign_spinless():
    """Test Slater determinant sign, spinless."""
    vec = ffsim.slater_determinant(8, [0, 1, 2, 3])
    assert vec[0] == 1

    vec = ffsim.slater_determinant(8, [0, 1, 3, 2])
    assert vec[0] == -1


@pytest.mark.parametrize(
    "norb, nocc", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_slater_determinant_spinless(norb: int, nocc: int):
    """Test Slater determinant with same rotation for both spins."""
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nocc, seed=RNG)

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=RNG)
    eigs, orbital_rotation = np.linalg.eigh(one_body_tensor)
    eig = sum(eigs[occupied_orbitals])
    state = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    hamiltonian = ffsim.contract.one_body_linop(
        one_body_tensor, norb=norb, nelec=(nocc, 0)
    )
    np.testing.assert_allclose(hamiltonian @ state, eig * state)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_slater_determinant_same_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant with same rotation for both spins."""
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=RNG)
    occ_a, occ_b = occupied_orbitals

    one_body_tensor = ffsim.random.random_hermitian(norb, seed=RNG)
    eigs, orbital_rotation = np.linalg.eigh(one_body_tensor)
    eig = sum(eigs[occ_a]) + sum(eigs[occ_b])
    state = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    hamiltonian = ffsim.contract.one_body_linop(one_body_tensor, norb=norb, nelec=nelec)
    np.testing.assert_allclose(hamiltonian @ state, eig * state)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_slater_determinant_diff_rotation(norb: int, nelec: tuple[int, int]):
    """Test Slater determinant with different rotations for each spin."""
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=RNG)
    occ_a, occ_b = occupied_orbitals

    orbital_rotation_a = ffsim.random.random_unitary(norb, seed=RNG)
    orbital_rotation_b = ffsim.random.random_unitary(norb, seed=RNG)

    state = ffsim.slater_determinant(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
    )
    state_a = ffsim.slater_determinant(
        norb,
        (occ_a, []),
        orbital_rotation=orbital_rotation_a,
    )
    state_b = ffsim.slater_determinant(
        norb,
        ([], occ_b),
        orbital_rotation=orbital_rotation_b,
    )

    np.testing.assert_allclose(state, np.kron(state_a, state_b))


def _as_tuple(rdms: np.ndarray | tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return rdms if isinstance(rdms, tuple) else (rdms,)


@pytest.mark.parametrize(
    "norb, nelec, rank, reorder",
    [
        (norb, nelec, rank, reorder)
        for norb, nelec in ffsim.testing.generate_norb_nelec(exhaustive=False)
        for rank in [1, 2, 3, 4]
        for reorder in [True, False]
        if norb > 0
    ],
)
def test_slater_determinant_rdms_same_rotation(
    norb: int, nelec: tuple[int, int], rank: int, reorder: bool
):
    """Test Slater determinant RDMs, same rotation for both spins."""
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=RNG)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)

    rdms = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=orbital_rotation,
        rank=rank,
        reorder=reorder,
    )
    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )
    expected_rdms = ffsim.rdms(vec, norb, nelec, rank=rank, reorder=reorder)

    assert len(_as_tuple(rdms)) == len(_as_tuple(expected_rdms))
    for rdm, expected_rdm in zip(_as_tuple(rdms), _as_tuple(expected_rdms)):
        np.testing.assert_allclose(rdm, expected_rdm, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec, rank, reorder",
    [
        (norb, nelec, rank, reorder)
        for norb, nelec in ffsim.testing.generate_norb_nelec(exhaustive=False)
        for rank in [1, 2, 3, 4]
        for reorder in [True, False]
        if norb > 0
    ],
)
def test_slater_determinant_rdms_diff_rotation(
    norb: int, nelec: tuple[int, int], rank: int, reorder: bool
):
    """Test Slater determinant RDMs, independent rotations for each spin."""
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=RNG)
    orbital_rotation_a = ffsim.random.random_unitary(norb, seed=RNG)
    orbital_rotation_b = ffsim.random.random_unitary(norb, seed=RNG)

    rdms = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
        rank=rank,
        reorder=reorder,
    )
    vec = ffsim.slater_determinant(
        norb,
        occupied_orbitals,
        orbital_rotation=(orbital_rotation_a, orbital_rotation_b),
    )
    expected_rdms = ffsim.rdms(vec, norb, nelec, rank=rank, reorder=reorder)

    assert len(_as_tuple(rdms)) == len(_as_tuple(expected_rdms))
    for rdm, expected_rdm in zip(_as_tuple(rdms), _as_tuple(expected_rdms)):
        np.testing.assert_allclose(rdm, expected_rdm, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec, rank, reorder",
    [
        (norb, nelec, rank, reorder)
        for norb, nelec in ffsim.testing.generate_norb_nocc(exhaustive=False)
        for rank in [1, 2, 3, 4]
        for reorder in [True, False]
        if norb > 0
    ],
)
def test_slater_determinant_rdms_spinless(
    norb: int, nelec: int, rank: int, reorder: bool
):
    """Test Slater determinant RDMs, spinless."""
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=RNG)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)

    rdms = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=orbital_rotation,
        rank=rank,
        reorder=reorder,
    )
    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )
    expected_rdms = ffsim.rdms(
        vec, norb, (nelec, 0), rank=rank, spin_summed=True, reorder=reorder
    )

    assert len(_as_tuple(rdms)) == len(_as_tuple(expected_rdms))
    for rdm, expected_rdm in zip(_as_tuple(rdms), _as_tuple(expected_rdms)):
        np.testing.assert_allclose(rdm, expected_rdm, atol=1e-12)


def _rdm_entry(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    orbitals: tuple[int, ...],
    n_beta: int,
    reorder: bool,
) -> complex:
    """Compute one RDM entry from explicit fermionic operators.

    The last ``n_beta`` creation-annihilation pairs act on spin beta and the rest
    act on spin alpha.
    """
    rank = len(orbitals) // 2
    spins = [False] * (rank - n_beta) + [True] * n_beta
    creators = [
        ffsim.cre_b(p) if spin else ffsim.cre_a(p)
        for p, spin in zip(orbitals[::2], spins)
    ]
    annihilators = [
        ffsim.des_b(q) if spin else ffsim.des_a(q)
        for q, spin in zip(orbitals[1::2], spins)
    ]
    if reorder:
        term = creators + annihilators[::-1]
    else:
        term = [op for pair in zip(creators, annihilators) for op in pair]
    op = ffsim.FermionOperator({tuple(term): 1})
    return np.vdot(vec, ffsim.linear_operator(op, norb, nelec) @ vec)


@pytest.mark.parametrize(
    "norb, nelec, closed_shell, reorder",
    [
        (norb, nelec, closed_shell, reorder)
        for norb, nelec, closed_shell in [
            (4, 3, False),
            (4, (3, 2), False),
            (4, (3, 3), False),
            (4, (3, 3), True),
        ]
        for reorder in [True, False]
    ],
)
def test_slater_determinant_rdms_high_rank(
    norb: int, nelec: int | tuple[int, int], closed_shell: bool, reorder: bool
):
    """Test sampled entries of a Slater determinant RDM beyond rank 4."""
    rank = 5
    spinless = isinstance(nelec, int)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=RNG)
    if closed_shell:
        assert isinstance(occupied_orbitals, tuple)
        occ_a, _ = occupied_orbitals
        occupied_orbitals = (occ_a, occ_a)
    nelec_spinful = (nelec, 0) if isinstance(nelec, int) else nelec
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)

    rdms = ffsim.slater_determinant_rdms(
        norb,
        occupied_orbitals,
        orbital_rotation=orbital_rotation,
        rank=rank,
        reorder=reorder,
    )
    assert len(rdms) == rank
    rdm = rdms[-1]
    vec = ffsim.slater_determinant(
        norb, occupied_orbitals, orbital_rotation=orbital_rotation
    )

    blocks = [0] if spinless else list(range(rank + 1))
    if spinless:
        rdm = rdm[None]
    assert rdm.shape == (len(blocks),) + (norb,) * (2 * rank)
    for n_beta in blocks:
        for i in range(10):
            if i % 2:
                # Repeated orbitals give nonzero entries for reorder=False
                orbitals = tuple(int(x) for x in RNG.integers(norb, size=2 * rank))
            else:
                # Distinct orbitals within each spin give nonzero entries for
                # reorder=True
                creators, annihilators = [], []
                for n_pairs in [rank - n_beta, n_beta]:
                    replace = n_pairs > norb
                    creators += list(RNG.choice(norb, n_pairs, replace=replace))
                    annihilators += list(RNG.choice(norb, n_pairs, replace=replace))
                orbitals = tuple(
                    int(x) for pair in zip(creators, annihilators) for x in pair
                )
            expected = _rdm_entry(vec, norb, nelec_spinful, orbitals, n_beta, reorder)
            np.testing.assert_allclose(rdm[(n_beta, *orbitals)], expected, atol=1e-12)


def test_slater_determinant_rdms_invalid_rank():
    """Test that a rank below 1 raises an error."""
    with pytest.raises(ValueError, match="rank"):
        ffsim.slater_determinant_rdms(3, [0, 1], rank=0)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_slater_determinant_amplitudes_spinless(norb: int, nelec: int):
    """Test computing Slater determinant amplitudes, spinless."""
    dim = ffsim.dim(norb, nelec)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)

    for occupied_orbitals in itertools.combinations(range(norb), nelec):
        bitstrings = ffsim.addresses_to_strings(range(dim), norb=norb, nelec=nelec)
        actual = ffsim.slater_determinant_amplitudes(
            bitstrings, norb, occupied_orbitals, orbital_rotation
        )
        expected = ffsim.slater_determinant(
            norb, occupied_orbitals=occupied_orbitals, orbital_rotation=orbital_rotation
        )
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_slater_determinant_amplitudes_spinful(norb: int, nelec: tuple[int, int]):
    """Test computing Slater determinant amplitudes, spinful."""
    dim_a, dim_b = ffsim.dims(norb, nelec)
    n_alpha, n_beta = nelec
    orb_rot_a = ffsim.random.random_unitary(norb, seed=RNG)
    orb_rot_b = ffsim.random.random_unitary(norb, seed=RNG)

    bitstrings_a = ffsim.addresses_to_strings(range(dim_a), norb=norb, nelec=n_alpha)
    bitstrings_b = ffsim.addresses_to_strings(range(dim_b), norb=norb, nelec=n_beta)
    bitstrings = cast(
        tuple[tuple[int], tuple[int]],
        tuple(zip(*itertools.product(bitstrings_a, bitstrings_b))),
    )

    for occ_a in itertools.combinations(range(norb), n_alpha):
        for occ_b in itertools.combinations(range(norb), n_beta):
            actual = ffsim.slater_determinant_amplitudes(
                bitstrings, norb, (occ_a, occ_b), (orb_rot_a, orb_rot_b)
            )
            expected = ffsim.slater_determinant(
                norb,
                occupied_orbitals=(occ_a, occ_b),
                orbital_rotation=(orb_rot_a, orb_rot_b),
            )
            ffsim.testing.assert_allclose_up_to_global_phase(actual, expected)


def test_slater_determinant_amplitudes_sign_spinful():
    """Test Slater determinant amplitudes sign, spinful."""
    amps = ffsim.slater_determinant_amplitudes(
        ([0b0011], [0b0011]), 4, ([0, 1], [0, 1]), np.eye(4)
    )
    assert amps[0] == 1

    amps = ffsim.slater_determinant_amplitudes(
        ([0b0011], [0b0011]), 4, ([0, 1], [1, 0]), np.eye(4)
    )
    assert amps[0] == -1

    amps = ffsim.slater_determinant_amplitudes(
        ([0b0011], [0b0011]), 4, ([1, 0], [1, 0]), np.eye(4)
    )
    assert amps[0] == 1


def test_slater_determinant_amplitudes_sign_spinless():
    """Test Slater determinant amplitudes sign, spinless."""
    amps = ffsim.slater_determinant_amplitudes([0b00001111], 8, [0, 1, 2, 3], np.eye(8))
    assert amps[0] == 1

    amps = ffsim.slater_determinant_amplitudes([0b00001111], 8, [0, 1, 3, 2], np.eye(8))
    assert amps[0] == -1
