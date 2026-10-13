# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test reduced density matrices."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import ffsim

RNG = np.random.default_rng(250368525468016360570772321326665989865)


def test_reduced_density_matrices_expectation():
    """Test ReducedDensityMatrix expectation method."""
    norb = 5
    nelec = (3, 2)

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    rdms = ffsim.ReducedDensityMatrix(
        *ffsim.rdms(vec, norb=norb, nelec=nelec, rank=2, spin_summed=True)
    )
    mol_ham = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    linop = ffsim.linear_operator(mol_ham, norb=norb, nelec=nelec)

    expected = np.vdot(vec, linop @ vec)
    actual = rdms.expectation(mol_ham)

    np.testing.assert_allclose(actual, expected)


def test_reduced_density_matrices_rotated():
    """Test ReducedDensityMatrix rotated method."""
    norb = 5
    nelec = (3, 2)

    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    rdms = ffsim.ReducedDensityMatrix(
        *ffsim.rdms(vec, norb=norb, nelec=nelec, rank=2, spin_summed=True)
    )
    mol_ham = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    energy = rdms.expectation(mol_ham)

    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    energy_rotated = rdms.rotated(orbital_rotation).expectation(
        mol_ham.rotated(orbital_rotation)
    )

    np.testing.assert_allclose(energy_rotated, energy)


@pytest.mark.parametrize("norb", range(1, 5))
def test_reduced_density_matrices_rotated_one_body_matrix_product(norb: int):
    """Test rotating the one-body RDM against an explicit matrix product."""
    one_rdm = RNG.normal(size=(norb, norb)) + 1j * RNG.normal(size=(norb, norb))
    two_rdm = RNG.normal(size=(norb,) * 4) + 1j * RNG.normal(size=(norb,) * 4)
    rdms = ffsim.ReducedDensityMatrix(one_rdm=one_rdm, two_rdm=two_rdm)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    np.testing.assert_allclose(
        rdms.rotated(orbital_rotation).one_rdm,
        orbital_rotation.conj() @ one_rdm @ orbital_rotation.T,
        atol=1e-12,
    )


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("spin_summed", [False, True])
def test_rdms_return_shapes(rank: int, spin_summed: bool):
    """A bare array is returned for rank 1 and a tuple of all ranks otherwise."""
    norb, nelec = 2, (1, 1)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    result = ffsim.rdms(vec, norb, nelec, rank=rank, spin_summed=spin_summed)
    if rank == 1:
        assert isinstance(result, np.ndarray)
    else:
        assert isinstance(result, tuple)
    rdms = _as_tuple(result)
    assert len(rdms) == rank
    for k, rdm in enumerate(rdms, start=1):
        assert rdm.dtype == complex
        assert rdm.shape == _rdm_shape(norb, k, spin_summed)


@pytest.mark.parametrize(
    "norb, nelec, rank",
    [
        (4, (2, 2), 1),
        (4, (2, 2), 2),
        (3, (1, 2), 2),
        (2, (2, 1), 3),
        (2, (2, 1), 4),
    ],
)
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("dtype", [float, complex])
def test_rdms_match_definition(
    norb: int,
    nelec: tuple[int, int],
    rank: int,
    spin_summed: bool,
    reorder: bool,
    dtype: type,
):
    """Every entry of every returned RDM matches its operator definition."""
    vec = ffsim.random.random_state_vector(
        ffsim.dim(norb, nelec), seed=RNG, dtype=dtype
    )
    rdms = _as_tuple(
        ffsim.rdms(
            vec, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
    )
    for k, rdm in enumerate(rdms, start=1):
        expected = _expected_rdm(vec, norb, nelec, k, spin_summed, reorder)
        np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize("norb, nelec", [(3, (2, 2)), (3, (3, 1))])
@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
def test_rdms_match_definition_sampled(
    norb: int, nelec: tuple[int, int], rank: int, spin_summed: bool, reorder: bool
):
    """Sampled higher-rank entries match their definition in a larger system."""
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    rdm = _as_tuple(
        ffsim.rdms(
            vec, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
    )[-1]
    # Random entries are often zero, so also check the entry with the largest
    # imaginary part, which only a coherence between determinants can produce.
    coherence = tuple(
        int(i) for i in np.unravel_index(np.argmax(np.abs(rdm.imag)), rdm.shape)
    )
    assert abs(rdm[coherence].imag) > 1e-4
    sampled = [tuple(int(RNG.integers(dim)) for dim in rdm.shape) for _ in range(30)]
    for index in [coherence, *sampled]:
        expected = _expected_rdm_entry(vec, norb, nelec, index, spin_summed, reorder)
        np.testing.assert_allclose(rdm[index], expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (1, (0, 0)),
        (1, (1, 1)),
        (2, (0, 1)),
        (2, (1, 0)),
        (2, (0, 2)),
        (2, (2, 0)),
        (2, (2, 2)),
    ],
)
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
def test_rdms_match_definition_particle_sectors(
    norb: int, nelec: tuple[int, int], rank: int, spin_summed: bool, reorder: bool
):
    """Vacuum, empty spin sectors, and full occupation match their definition."""
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    rdm = _as_tuple(
        ffsim.rdms(
            vec, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
    )[-1]
    expected = _expected_rdm(vec, norb, nelec, rank, spin_summed, reorder)
    np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec", [(1, (0, 0)), (2, (1, 0)), (2, (0, 2)), (3, (2, 1)), (3, (1, 1))]
)
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_rdms_vanishing_blocks(norb: int, nelec: tuple[int, int], rank: int):
    """A normal-ordered block vanishes exactly when it needs too many particles."""
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    blocks = _as_tuple(ffsim.rdms(vec, norb, nelec, rank=rank))[-1]
    for n_beta, block in enumerate(blocks):
        if rank - n_beta > nelec[0] or n_beta > nelec[1]:
            np.testing.assert_allclose(block, 0, atol=1e-12)
        else:
            assert np.linalg.norm(block) > 1e-4
    summed = _as_tuple(ffsim.rdms(vec, norb, nelec, rank=rank, spin_summed=True))[-1]
    if rank > sum(nelec):
        np.testing.assert_allclose(summed, 0, atol=1e-12)
    else:
        assert np.linalg.norm(summed) > 1e-4


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_rdms_strided_single_precision_input(rank: int, spin_summed: bool, dtype: type):
    """Non-contiguous single-precision input gives the same result as complex128."""
    norb, nelec = 3, (2, 1)
    dim = ffsim.dim(norb, nelec)
    data: np.ndarray = np.empty(2 * dim, dtype=dtype)
    vec = data[::2]
    vec[:] = RNG.standard_normal(dim)
    if np.iscomplexobj(vec):
        vec += 1j * RNG.standard_normal(dim)
    assert not vec.flags.c_contiguous
    actual = _as_tuple(ffsim.rdms(vec, norb, nelec, rank=rank, spin_summed=spin_summed))
    expected = _as_tuple(
        ffsim.rdms(vec.astype(complex), norb, nelec, rank=rank, spin_summed=spin_summed)
    )
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, e, atol=1e-12)


@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
def test_rdms_lower_ranks_agree(spin_summed: bool, reorder: bool):
    """Requesting a higher rank does not change the lower-rank RDMs."""
    norb, nelec = 3, (2, 2)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    rdms4 = ffsim.rdms(
        vec, norb, nelec, rank=4, spin_summed=spin_summed, reorder=reorder
    )
    for rank in [1, 2, 3]:
        lower = _as_tuple(
            ffsim.rdms(
                vec, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
            )
        )
        for a, e in zip(rdms4[:rank], lower, strict=True):
            np.testing.assert_allclose(a, e, atol=1e-12)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
def test_rdms_scale_with_squared_norm(rank: int, spin_summed: bool, reorder: bool):
    """Scaling the vector by c scales every RDM by |c|**2, ignoring the phase."""
    norb, nelec = 3, (2, 2)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    c = 1.7 * np.exp(0.72j)
    scaled = _as_tuple(
        ffsim.rdms(
            c * vec, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
    )
    original = _as_tuple(
        ffsim.rdms(
            vec, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
    )
    for a, e in zip(scaled, original, strict=True):
        np.testing.assert_allclose(a, abs(c) ** 2 * e, atol=1e-12)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("reorder", [False, True])
def test_rdms_spin_blocks_sum_to_spin_summed(rank: int, reorder: bool):
    """Summing the spin blocks over all spin assignments gives the spin-summed RDM."""
    norb, nelec = 3, (2, 2)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    blocks = _as_tuple(ffsim.rdms(vec, norb, nelec, rank=rank, reorder=reorder))[-1]
    summed = _as_tuple(
        ffsim.rdms(vec, norb, nelec, rank=rank, spin_summed=True, reorder=reorder)
    )[-1]
    reconstructed = np.zeros_like(summed)
    for spins in itertools.product([False, True], repeat=rank):
        # Blocks store alpha pairs before beta pairs. Pairs of different spin
        # commute in either ordering convention, so permute the pairs of the block
        # into the order given by spins. The stable sort keeps the order of
        # same-spin pairs.
        block_order = sorted(range(rank), key=lambda pair: spins[pair])
        pair_in_block = np.argsort(block_order)
        axes = [2 * pair_in_block[pair] + j for pair in range(rank) for j in range(2)]
        reconstructed += blocks[sum(spins)].transpose(axes)
    np.testing.assert_allclose(summed, reconstructed, atol=1e-12)


@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("reorder", [False, True])
def test_rdms_partial_trace(rank: int, reorder: bool):
    """Tracing out the first or last pair of indices gives the next lower RDM.

    Without reordering, each traced pair is a number operator, contributing a factor
    of N. With reordering, the traced creation-annihilation pair acts on the state
    after the remaining same-spin annihilators, so the factor is reduced by their
    number.
    """
    norb, nelec = 3, (2, 2)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    *_, lower_summed, summed = ffsim.rdms(
        vec, norb, nelec, rank=rank, spin_summed=True, reorder=reorder
    )
    *_, lower_blocks, blocks = ffsim.rdms(vec, norb, nelec, rank=rank, reorder=reorder)
    other_pairs = rank - 1 if reorder else 0
    for axis in [0, 2 * rank - 2]:
        np.testing.assert_allclose(
            np.trace(summed, axis1=axis, axis2=axis + 1),
            (sum(nelec) - other_pairs) * lower_summed,
            atol=1e-12,
        )

    for n_beta, block in enumerate(blocks):
        n_alpha = rank - n_beta
        # Blocks list the alpha pairs first, so the first pair is beta only when
        # there are no alpha pairs, and the last pair is alpha only when there are
        # no beta pairs.
        for pair, is_beta in [(0, n_alpha == 0), (rank - 1, n_beta > 0)]:
            n_same_spin = n_beta if is_beta else n_alpha
            factor = nelec[is_beta] - (n_same_spin - 1 if reorder else 0)
            np.testing.assert_allclose(
                np.trace(block, axis1=2 * pair, axis2=2 * pair + 1),
                factor * lower_blocks[n_beta - is_beta],
                atol=1e-12,
            )


@pytest.mark.parametrize("rank", [-1, 0, 5])
def test_rdms_unsupported_rank(rank: int):
    """Unsupported ranks raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match=f"rank {rank}"):
        ffsim.rdms(np.ones(1), 1, (0, 0), rank=rank)


@pytest.mark.parametrize(
    "norb, nelec, rank",
    [(4, (2, 2), 1), (3, (1, 2), 2), (2, (2, 1), 3), (2, (2, 1), 4)],
)
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("bra_dtype", [float, complex])
@pytest.mark.parametrize("ket_dtype", [float, complex])
def test_transition_rdms_match_definition(
    norb: int,
    nelec: tuple[int, int],
    rank: int,
    spin_summed: bool,
    reorder: bool,
    bra_dtype: type,
    ket_dtype: type,
):
    """Transition RDMs match explicit matrix elements for independent states."""
    bra = ffsim.random.random_state_vector(
        ffsim.dim(norb, nelec), seed=RNG, dtype=bra_dtype
    )
    ket = ffsim.random.random_state_vector(
        ffsim.dim(norb, nelec), seed=RNG, dtype=ket_dtype
    )
    result = ffsim.transition_rdms(
        bra, ket, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
    )
    assert isinstance(result, np.ndarray if rank == 1 else tuple)
    rdms = _as_tuple(result)
    assert len(rdms) == rank
    for k, rdm in enumerate(rdms, start=1):
        assert rdm.dtype == complex
        expected = _expected_rdm(bra, norb, nelec, k, spin_summed, reorder, ket=ket)
        np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize("norb, nelec", [(3, (2, 2)), (3, (3, 1))])
@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
def test_transition_rdms_match_definition_sampled(
    norb: int, nelec: tuple[int, int], rank: int, spin_summed: bool, reorder: bool
):
    """Higher-rank transition RDMs match matrix elements in larger systems."""
    bra, ket = [
        ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
        for _ in range(2)
    ]
    rdms = _as_tuple(
        ffsim.transition_rdms(
            bra, ket, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
    )
    for rdm in rdms:
        largest = tuple(
            int(i) for i in np.unravel_index(np.argmax(np.abs(rdm)), rdm.shape)
        )
        assert abs(rdm[largest]) > 1e-4
        sampled = [
            tuple(int(RNG.integers(dim)) for dim in rdm.shape) for _ in range(30)
        ]
        for index in [largest, *sampled]:
            expected = _expected_rdm_entry(
                bra, norb, nelec, index, spin_summed, reorder, ket=ket
            )
            np.testing.assert_allclose(rdm[index], expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec", [(1, (0, 0)), (1, (1, 1)), (2, (0, 1)), (2, (1, 0)), (2, (2, 2))]
)
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
def test_transition_rdms_particle_sectors(
    norb: int, nelec: tuple[int, int], spin_summed: bool, reorder: bool
):
    """Transition RDMs handle vacuum, empty spin sectors, and full occupation."""
    bra, ket = [
        ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
        for _ in range(2)
    ]
    for rank in [2, 4]:
        rdms = ffsim.transition_rdms(
            bra, ket, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
        for k, rdm in enumerate(rdms, start=1):
            expected = _expected_rdm(bra, norb, nelec, k, spin_summed, reorder, ket=ket)
            np.testing.assert_allclose(rdm, expected, atol=1e-12)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize("dtype", [float, complex])
def test_transition_rdms_equal_states(
    rank: int, spin_summed: bool, reorder: bool, dtype: type
):
    """Equal bra and ket give ordinary RDMs, including for unnormalized states."""
    norb, nelec = 3, (2, 2)
    vec = 1.7 * ffsim.random.random_state_vector(
        ffsim.dim(norb, nelec), seed=RNG, dtype=dtype
    )
    actual = _as_tuple(
        ffsim.transition_rdms(
            vec, vec, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
    )
    expected = _as_tuple(
        ffsim.rdms(
            vec, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
    )
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, e, atol=1e-12)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("reorder", [False, True])
@pytest.mark.parametrize(
    "scales", [(1.2 + 0.7j, -0.3 + 0.9j), (1j, 1), (1, 1j), (0, 1)]
)
def test_transition_rdms_scaling(
    rank: int, spin_summed: bool, reorder: bool, scales: tuple[complex, complex]
):
    """Scaling is conjugate-linear in bra and linear in ket, even for zero input."""
    norb, nelec = 3, (2, 2)
    bra, ket = [
        ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG, dtype=float)
        for _ in range(2)
    ]
    a, b = scales
    actual = _as_tuple(
        ffsim.transition_rdms(
            a * bra,
            b * ket,
            norb,
            nelec,
            rank=rank,
            spin_summed=spin_summed,
            reorder=reorder,
        )
    )
    original = _as_tuple(
        ffsim.transition_rdms(
            bra, ket, norb, nelec, rank=rank, spin_summed=spin_summed, reorder=reorder
        )
    )
    for rdm, unscaled in zip(actual, original, strict=True):
        np.testing.assert_allclose(rdm, a.conjugate() * b * unscaled, atol=1e-12)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("spin_summed", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_transition_rdms_strided_single_precision_input(
    rank: int, spin_summed: bool, dtype: type
):
    """Both states can have non-contiguous single-precision amplitudes."""
    norb, nelec = 3, (2, 1)
    data: np.ndarray = np.empty((ffsim.dim(norb, nelec), 2), dtype=dtype)
    data[:] = RNG.standard_normal(data.shape)
    if np.iscomplexobj(data):
        data += 1j * RNG.standard_normal(data.shape)
    bra, ket = data.T
    assert not bra.flags.c_contiguous
    assert not ket.flags.c_contiguous
    actual = _as_tuple(
        ffsim.transition_rdms(bra, ket, norb, nelec, rank=rank, spin_summed=spin_summed)
    )
    expected = _as_tuple(
        ffsim.transition_rdms(
            bra.astype(complex),
            ket.astype(complex),
            norb,
            nelec,
            rank=rank,
            spin_summed=spin_summed,
        )
    )
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, e, atol=1e-12)


def test_transition_rdms_orthogonal_states():
    """Orthogonal determinants can have a nonzero transition RDM."""
    norb, nelec = 2, (1, 0)
    bra = np.array([1, 0])
    ket = np.array([0, 1j])
    rdm = ffsim.transition_rdms(bra, ket, norb, nelec)
    np.testing.assert_allclose(rdm[0], np.array([[0, 1j], [0, 0]]), atol=1e-12)
    np.testing.assert_allclose(rdm[1], 0, atol=1e-12)


def test_transition_rdms_hamiltonian_matrix_element():
    """Contracting transition RDMs gives a molecular Hamiltonian matrix element."""
    norb, nelec = 4, (2, 1)
    bra, ket = [
        ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
        for _ in range(2)
    ]
    mol_ham = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    rdm1, rdm2 = ffsim.transition_rdms(bra, ket, norb, nelec, rank=2, spin_summed=True)
    actual = (
        mol_ham.constant * np.vdot(bra, ket)
        + np.einsum("pq,pq->", mol_ham.one_body_tensor, rdm1)
        + 0.5 * np.einsum("pqrs,pqrs->", mol_ham.two_body_tensor, rdm2)
    )
    expected = np.vdot(bra, ffsim.linear_operator(mol_ham, norb, nelec) @ ket)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("rank", [-1, 0, 5])
def test_transition_rdms_unsupported_rank(rank: int):
    """Unsupported ranks raise NotImplementedError for transition RDMs."""
    with pytest.raises(NotImplementedError, match=f"rank {rank}"):
        ffsim.transition_rdms(np.ones(1), np.ones(1), 1, (0, 0), rank=rank)


def test_transition_rdms_mismatched_shapes():
    """Bra and ket vectors with different shapes raise ValueError."""
    norb, nelec = 3, (1, 1)
    bra = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=RNG)
    with pytest.raises(ValueError, match="same shape"):
        ffsim.transition_rdms(bra, bra[:3], norb, nelec)


def _as_tuple(rdms: np.ndarray | tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Normalize the return value of ffsim.rdms to a tuple of RDMs."""
    return (rdms,) if isinstance(rdms, np.ndarray) else rdms


def _rdm_shape(norb: int, rank: int, spin_summed: bool) -> tuple[int, ...]:
    """Return the shape of the rank-``rank`` RDM."""
    return (() if spin_summed else (rank + 1,)) + (norb,) * (2 * rank)


def _expected_rdm(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    rank: int,
    spin_summed: bool,
    reorder: bool,
    *,
    ket: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a whole RDM from explicit fermionic operators."""
    shape = _rdm_shape(norb, rank, spin_summed)
    expected = np.zeros(shape, dtype=complex)
    for index in np.ndindex(shape):
        expected[index] = _expected_rdm_entry(
            vec, norb, nelec, index, spin_summed, reorder, ket=ket
        )
    return expected


def _expected_rdm_entry(
    vec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    index: tuple[int, ...],
    spin_summed: bool,
    reorder: bool,
    *,
    ket: np.ndarray | None = None,
) -> complex:
    """Compute one RDM entry from explicit fermionic operators.

    For a spin-resolved RDM, the first index selects the spin block, whose alpha
    pairs come before its beta pairs. True denotes a beta pair.
    """
    orbitals = index if spin_summed else index[1:]
    rank = len(orbitals) // 2
    if spin_summed:
        spin_assignments = list(itertools.product([False, True], repeat=rank))
    else:
        n_beta = index[0]
        spin_assignments = [(False,) * (rank - n_beta) + (True,) * n_beta]
    op = ffsim.FermionOperator(
        {_rdm_term(orbitals, spins, reorder): 1 for spins in spin_assignments}
    )
    return np.vdot(
        vec, ffsim.linear_operator(op, norb, nelec) @ (vec if ket is None else ket)
    )


def _rdm_term(
    orbitals: tuple[int, ...], spins: tuple[bool, ...], reorder: bool
) -> tuple[tuple[bool, bool, int], ...]:
    """Return the operator term whose expectation is the RDM entry."""
    creators = [
        ffsim.cre_b(p) if spin else ffsim.cre_a(p)
        for p, spin in zip(orbitals[::2], spins, strict=True)
    ]
    annihilators = [
        ffsim.des_b(q) if spin else ffsim.des_a(q)
        for q, spin in zip(orbitals[1::2], spins, strict=True)
    ]
    if reorder:
        term = creators + annihilators[::-1]
    else:
        term = [op for pair in zip(creators, annihilators) for op in pair]
    return tuple(term)
