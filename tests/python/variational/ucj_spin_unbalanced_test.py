# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for spin-unbalanced unitary cluster Jastrow ansatz."""

import itertools
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
import pyscf
import pyscf.cc
import pytest
import scipy.optimize

import ffsim
from ffsim.variational.ucj_spin_unbalanced import (
    ucj_op_spin_unbalanced_from_parameters_jax,
)

RNG = np.random.default_rng(155313014187200663515254027006546431903)


def test_n_params():
    for norb, n_reps, with_final_orbital_rotation in itertools.product(
        [1, 2, 3], [1, 2, 3], [False, True]
    ):
        operator = ffsim.random.random_ucj_op_spin_unbalanced(
            norb, n_reps=n_reps, with_final_orbital_rotation=with_final_orbital_rotation
        )
        actual = ffsim.UCJOpSpinUnbalanced.n_params(
            norb, n_reps, with_final_orbital_rotation=with_final_orbital_rotation
        )
        expected = len(operator.to_parameters())
        assert actual == expected

        pairs_aa = list(itertools.combinations_with_replacement(range(norb), 2))[:norb]
        pairs_ab = list(itertools.combinations_with_replacement(range(norb), 2))[:norb]
        pairs_bb = list(itertools.combinations_with_replacement(range(norb), 2))[:norb]

        actual = ffsim.UCJOpSpinUnbalanced.n_params(
            norb,
            n_reps,
            interaction_pairs=(pairs_aa, pairs_ab, pairs_bb),
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        expected = len(
            operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab, pairs_bb))
        )
        assert actual == expected

        with pytest.raises(ValueError, match="triangular"):
            actual = ffsim.UCJOpSpinUnbalanced.n_params(
                norb,
                n_reps,
                interaction_pairs=([(1, 0)], pairs_ab, pairs_bb),
            )
        with pytest.raises(ValueError, match="triangular"):
            actual = ffsim.UCJOpSpinUnbalanced.n_params(
                norb,
                n_reps,
                interaction_pairs=(pairs_aa, pairs_ab, [(1, 0)]),
            )
        with pytest.raises(ValueError, match="Duplicate"):
            actual = ffsim.UCJOpSpinUnbalanced.n_params(
                norb,
                n_reps,
                interaction_pairs=(pairs_aa, [(1, 0), (1, 0)], pairs_bb),
            )
        with pytest.raises(ValueError, match="outside the range"):
            actual = ffsim.UCJOpSpinUnbalanced.n_params(
                norb,
                n_reps,
                interaction_pairs=(pairs_aa, [(norb, 0)], pairs_bb),
            )


def test_t_amplitudes_validate_interaction_pairs():
    """Test from_t_amplitudes validates each list of interaction pairs."""
    norb = 4
    nelec = (2, 1)
    t2 = ffsim.random.random_t2_amplitudes_unrestricted(
        norb, nelec, seed=RNG, dtype=float
    )

    # The alpha-beta pairs are allowed to be lower triangular, but not duplicated.
    with pytest.raises(ValueError, match="Duplicate"):
        _ = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2, n_reps=2, interaction_pairs=(None, [(0, 1), (0, 1)], None)
        )
    with pytest.raises(ValueError, match="triangular"):
        _ = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2, n_reps=2, interaction_pairs=([(1, 0)], None, None)
        )
    with pytest.raises(ValueError, match="triangular"):
        _ = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2, n_reps=2, interaction_pairs=(None, None, [(1, 0)])
        )
    with pytest.raises(ValueError, match="outside the range"):
        _ = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2, n_reps=2, interaction_pairs=(None, [(0, norb)], None)
        )


def test_parameters_roundtrip_all_to_all():
    norb = 5
    n_reps = 2

    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_ucj_op_spin_unbalanced(
            norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = ffsim.UCJOpSpinUnbalanced.from_parameters(
            operator.to_parameters(),
            norb=norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(roundtripped, operator)


def test_parameters_roundtrip_interaction_pairs():
    norb = 5
    n_reps = 2
    interaction_pairs_list = [
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        ([(0, 1)], [(0, 1)], []),
        ([(0, 1)], [], []),
        ([], [], []),
    ]

    for interaction_pairs in interaction_pairs_list:
        for with_final_orbital_rotation in [False, True]:
            operator = ffsim.random.random_ucj_op_spin_unbalanced(
                norb,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs,
                with_final_orbital_rotation=with_final_orbital_rotation,
                seed=RNG,
            )
            roundtripped = ffsim.UCJOpSpinUnbalanced.from_parameters(
                operator.to_parameters(interaction_pairs=interaction_pairs),
                norb=norb,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs,
                with_final_orbital_rotation=with_final_orbital_rotation,
            )
            assert ffsim.approx_eq(roundtripped, operator)


def test_from_parameters_jax():
    """Test JAX parameter unpacking matches from_parameters."""
    norb = 4
    interaction_pairs_list: list[
        tuple[
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
            list[tuple[int, int]] | None,
        ]
        | None
    ] = [
        None,
        ([(0, 1)], [(3, 1), (0, 2)], None),
        (None, [], [(2, 3)]),
        ([], [], []),
    ]
    for n_reps, interaction_pairs, with_final_orbital_rotation in itertools.product(
        [0, 1, 2], interaction_pairs_list, [False, True]
    ):
        n_params = ffsim.UCJOpSpinUnbalanced.n_params(
            norb,
            n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        params = RNG.standard_normal(n_params)
        operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
            params,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        diag_coulomb_mats, orbital_rotations, final_orbital_rotation = (
            ucj_op_spin_unbalanced_from_parameters_jax(
                jnp.asarray(params),
                norb=norb,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs,
                with_final_orbital_rotation=with_final_orbital_rotation,
            )
        )
        np.testing.assert_allclose(diag_coulomb_mats, operator.diag_coulomb_mats)
        np.testing.assert_allclose(
            orbital_rotations, operator.orbital_rotations, atol=1e-12
        )
        if with_final_orbital_rotation:
            assert final_orbital_rotation is not None
            assert operator.final_orbital_rotation is not None
            np.testing.assert_allclose(
                final_orbital_rotation, operator.final_orbital_rotation, atol=1e-12
            )
        else:
            assert final_orbital_rotation is None


def test_t_amplitudes_energy():
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["Be", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    scf = pyscf.scf.ROHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()

    # Get molecular data and molecular Hamiltonian
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # Construct UCJ operator
    n_reps: int | tuple[int, int] = 4
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, t1=ccsd.t1, n_reps=n_reps
    )

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
    linop = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    energy = np.real(np.vdot(ansatz_state, linop @ ansatz_state))
    np.testing.assert_allclose(energy, -15.128401)

    # Test setting number of reps as tuple
    n_reps = (4, 2)
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, t1=ccsd.t1, n_reps=n_reps
    )
    ansatz_state = ffsim.apply_unitary(
        reference_state, operator, norb=norb, nelec=nelec
    )
    energy = np.real(np.vdot(ansatz_state, linop @ ansatz_state))
    np.testing.assert_allclose(energy, -15.128659)

    # Test setting number of reps as None
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(ccsd.t2, t1=ccsd.t1)
    ansatz_state = ffsim.apply_unitary(
        reference_state, operator, norb=norb, nelec=nelec
    )
    energy = np.real(np.vdot(ansatz_state, linop @ ansatz_state))
    np.testing.assert_allclose(energy, -15.140248)


def test_t_amplitudes_first_order_matches_uccsd():
    """Test from_t_amplitudes agrees with UCCSD to first order in the t2 amplitudes."""
    norb = 5
    nelec = (3, 2)
    nocc_a, nocc_b = nelec
    nvrt_a = norb - nocc_a
    nvrt_b = norb - nocc_b
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    t1 = (np.zeros((nocc_a, nvrt_a)), np.zeros((nocc_b, nvrt_b)))
    t2aa, t2ab, t2bb = ffsim.random.random_t2_amplitudes_unrestricted(
        norb, nelec, seed=RNG, dtype=float
    )
    zero_aa = np.zeros_like(t2aa)
    zero_ab = np.zeros_like(t2ab)
    zero_bb = np.zeros_like(t2bb)
    eps = 1e-6
    tol = 1e-12
    for aa, ab, bb in [
        (t2aa, zero_ab, zero_bb),
        (zero_aa, t2ab, zero_bb),
        (zero_aa, zero_ab, t2bb),
        (t2aa, t2ab, t2bb),
    ]:
        t2 = (eps * aa, eps * ab, eps * bb)
        uccsd_op = ffsim.UCCSDOpUnrestrictedReal(t1=t1, t2=t2)
        ucj_op = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, tol=tol)
        uccsd_state = ffsim.apply_unitary(
            reference_state, uccsd_op, norb=norb, nelec=nelec
        )
        ucj_state = ffsim.apply_unitary(reference_state, ucj_op, norb=norb, nelec=nelec)
        np.testing.assert_allclose(
            (ucj_state - reference_state) / eps,
            (uccsd_state - reference_state) / eps,
            atol=1e-4,
        )


def test_t_amplitudes_validate():
    """Test from_t_amplitudes validates the antisymmetry of the same-spin amplitudes."""
    norb = 5
    nocc_a, nocc_b = 3, 2
    nvrt_a = norb - nocc_a
    nvrt_b = norb - nocc_b
    t2aa = ffsim.random.random_t2_amplitudes_spinless(
        norb, nocc_a, seed=RNG, dtype=float
    )
    t2ab = RNG.standard_normal((nocc_a, nocc_b, nvrt_a, nvrt_b))
    t2bb = ffsim.random.random_t2_amplitudes_spinless(
        norb, nocc_b, seed=RNG, dtype=float
    )
    bad_t2aa = ffsim.random.random_t2_amplitudes(norb, nocc_a, seed=RNG, dtype=float)
    bad_t2bb = ffsim.random.random_t2_amplitudes(norb, nocc_b, seed=RNG, dtype=float)
    with pytest.raises(ValueError, match="t2aa amplitudes were not antisymmetric"):
        ffsim.UCJOpSpinUnbalanced.from_t_amplitudes((bad_t2aa, t2ab, t2bb))
    with pytest.raises(ValueError, match="t2bb amplitudes were not antisymmetric"):
        ffsim.UCJOpSpinUnbalanced.from_t_amplitudes((t2aa, t2ab, bad_t2bb))
    # Validation can be skipped
    ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        (bad_t2aa, t2ab, bad_t2bb), validate=False
    )


def test_t_amplitudes_random_n_reps():
    norb = 5
    nelec = (3, 2)
    nocc_a, nocc_b = nelec
    nvrt_a = norb - nocc_a
    nvrt_b = norb - nocc_b

    n_reps: int | tuple[int, int]
    for n_reps in [5, 50, (10, 5), (0, 1)]:  # type: ignore
        t2 = ffsim.random.random_t2_amplitudes_unrestricted(
            norb, nelec, seed=RNG, dtype=float
        )
        t1a = RNG.standard_normal((nocc_a, nvrt_a))
        t1b = RNG.standard_normal((nocc_b, nvrt_b))
        t1 = (t1a, t1b)
        operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, t1=t1, n_reps=n_reps)
        total_n_reps = n_reps if isinstance(n_reps, int) else sum(n_reps)
        assert operator.n_reps == total_n_reps
        actual = len(operator.to_parameters())
        expected = ffsim.UCJOpSpinUnbalanced.n_params(
            norb, total_n_reps, with_final_orbital_rotation=True
        )
        assert actual == expected


def test_t_amplitudes_numpy_integer_n_reps():
    """Test that n_reps can be a Numpy integer."""
    norb = 5
    nelec = (3, 2)
    t2 = ffsim.random.random_t2_amplitudes_unrestricted(
        norb, nelec, seed=RNG, dtype=float
    )
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        t2,
        n_reps=np.int64(3),  # type: ignore[arg-type]
    )
    expected = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, n_reps=3)
    assert ffsim.approx_eq(operator, expected)
    with pytest.raises(ValueError, match="n_reps"):
        ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2,
            n_reps=np.int64(0),  # type: ignore[arg-type]
        )


def test_t_amplitudes_integer_n_reps_split():
    """Test that an integer n_reps selects the terms with the largest weights."""
    norb = 5
    nelec = (3, 2)
    t2aa, t2ab, t2bb = ffsim.random.random_t2_amplitudes_unrestricted(
        norb, nelec, seed=RNG, dtype=float
    )

    # Large same-spin amplitudes are used before small alpha-beta amplitudes
    t2 = (t2aa, 1e-3 * t2ab, t2bb)
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, n_reps=3)
    expected = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, n_reps=(0, 3))
    assert ffsim.approx_eq(operator, expected)

    # Large alpha-beta amplitudes are used before small same-spin amplitudes
    t2 = (1e-3 * t2aa, t2ab, 1e-3 * t2bb)
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, n_reps=3)
    expected = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, n_reps=(3, 0))
    assert ffsim.approx_eq(operator, expected)

    # The full number of terms reproduces the full decomposition, and extra
    # repetitions are padded with identities
    t2 = (t2aa, t2ab, t2bb)
    full = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2)
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, n_reps=full.n_reps)
    assert ffsim.approx_eq(operator, full)
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2, n_reps=full.n_reps + 2)
    np.testing.assert_allclose(
        operator.diag_coulomb_mats[: full.n_reps], full.diag_coulomb_mats
    )
    np.testing.assert_allclose(operator.diag_coulomb_mats[full.n_reps :], 0)


def test_t_amplitudes_integer_n_reps_split_by_magnitude():
    """Test that an integer n_reps ranks by singular values and eigenvalues."""
    norb = 5
    nelec = (3, 2)
    t2aa, t2ab, t2bb = ffsim.random.random_t2_amplitudes_unrestricted(
        norb, nelec, seed=RNG, dtype=float
    )
    nocc_a, nocc_b = nelec
    nvrt_a, nvrt_b = norb - nocc_a, norb - nocc_b

    def magnitudes(t2ab):
        # 4 alpha-beta repetitions per singular value, 2 same-spin terms per eigenvalue.
        # Same-spin eigenvalues are scaled by 1/sqrt(2) so that every distinct
        # excitation counts equally in the spin-orbital norm.
        mat_ab = t2ab.transpose(0, 2, 1, 3).reshape(nocc_a * nvrt_a, nocc_b * nvrt_b)
        singular_vals = np.linalg.svd(mat_ab, compute_uv=False)
        mags_ab = np.repeat(singular_vals[singular_vals > 1e-8], 4)
        eigs = []
        for t2, nocc, nvrt in [(t2aa, nocc_a, nvrt_a), (t2bb, nocc_b, nvrt_b)]:
            these_eigs = np.linalg.eigvalsh(
                t2.transpose(0, 2, 1, 3).reshape(nocc * nvrt, nocc * nvrt) / np.sqrt(2)
            )
            these_eigs = np.sort(np.abs(these_eigs))[::-1]
            eigs.append(np.repeat(these_eigs[these_eigs > 1e-8], 2))
        eigs_aa, eigs_bb = eigs
        n_same_spin = max(len(eigs_aa), len(eigs_bb))
        eigs_aa = np.pad(eigs_aa, (0, n_same_spin - len(eigs_aa)))
        eigs_bb = np.pad(eigs_bb, (0, n_same_spin - len(eigs_bb)))
        return mags_ab, np.sqrt(eigs_aa**2 + eigs_bb**2)

    # Scale the alpha-beta amplitudes so that the largest same-spin magnitude lies
    # strictly between the two largest distinct singular values
    mags_ab, mags_same_spin = magnitudes(t2ab)
    levels = np.unique(np.round(mags_ab, 8))[::-1]
    t2ab = mags_same_spin[0] / (0.5 * (levels[0] + levels[1])) * t2ab
    mags_ab, mags_same_spin = magnitudes(t2ab)
    n_ab_above = int(np.count_nonzero(mags_ab > mags_same_spin[0]))
    assert 0 < n_ab_above < len(mags_ab)

    # Merge in decreasing order of magnitude, with ties going to alpha-beta
    mags = np.concatenate([mags_ab, mags_same_spin])
    is_same_spin = np.concatenate(
        [np.zeros(len(mags_ab), dtype=int), np.ones(len(mags_same_spin), dtype=int)]
    )
    order = np.lexsort((is_same_spin, -np.round(mags, 8)))
    for n_reps in range(1, len(mags) + 1):
        n_reps_same_spin = int(np.sum(is_same_spin[order[:n_reps]]))
        expected_split = (n_reps - n_reps_same_spin, n_reps_same_spin)
        if n_reps == n_ab_above + 1:
            # The alpha-beta repetitions above the largest same-spin magnitude come
            # first, followed by the largest same-spin repetition
            assert expected_split == (n_ab_above, 1)
        operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            (t2aa, t2ab, t2bb), n_reps=n_reps
        )
        expected = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            (t2aa, t2ab, t2bb), n_reps=expected_split
        )
        assert ffsim.approx_eq(operator, expected)


@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("method", [None, "SLSQP"])
def test_t_amplitudes_optimize_kwargs(optimize: bool, method: str | None):
    t2 = np.ones((1, 1, 1, 1))
    callback = Mock()
    bounds = scipy.optimize.Bounds(-2, 2)
    constraint = scipy.optimize.NonlinearConstraint(lambda x: np.sum(x**2), 0, 100)
    options = dict(maxiter=2)
    kwargs: dict = dict(
        callback=callback, bounds=bounds, constraints=constraint, options=options
    )
    if method is not None:
        kwargs["method"] = method

    # Observe every SciPy invocation without solving each intermediate optimization.
    with patch(
        "scipy.optimize.minimize",
        side_effect=lambda fun, x0, **kwargs: scipy.optimize.OptimizeResult(x=x0),
    ) as minimize:
        ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            (t2, t2, t2),
            n_reps=(1, 1),
            multi_stage_start=2,
            multi_stage_step=1,
            optimize=optimize,
            validate=False,
            **kwargs,
        )

    assert minimize.call_count == (6 if optimize else 0)
    for call in minimize.call_args_list:
        assert call.kwargs["method"] == (method or "L-BFGS-B")
        assert call.kwargs["callback"] is callback
        assert call.kwargs["bounds"] is bounds
        assert call.kwargs["constraints"] is constraint
        assert call.kwargs["options"] is options
    assert options == {"maxiter": 2}
    assert ("method" in kwargs) == (method is not None)


def test_t_amplitudes_random_optimize():
    norb = 5
    nelec = (3, 2)
    nocc_a, nocc_b = nelec
    nvrt_a = norb - nocc_a
    nvrt_b = norb - nocc_b

    n_reps: int | tuple[int, int]
    for n_reps in [5, (10, 5)]:  # type: ignore
        t2 = ffsim.random.random_t2_amplitudes_unrestricted(
            norb, nelec, seed=RNG, dtype=float
        )
        t1a = RNG.standard_normal((nocc_a, nvrt_a))
        t1b = RNG.standard_normal((nocc_b, nvrt_b))
        t1 = (t1a, t1b)
        operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2, t1=t1, n_reps=n_reps, optimize=True, options=dict(maxiter=10)
        )
        total_n_reps = n_reps if isinstance(n_reps, int) else sum(n_reps)
        assert operator.n_reps == total_n_reps
        actual = len(operator.to_parameters())
        expected = ffsim.UCJOpSpinUnbalanced.n_params(
            norb, total_n_reps, with_final_orbital_rotation=True
        )
        assert actual == expected


def test_t_amplitudes_zero_n_reps():
    norb = 5
    nelec = (3, 2)
    nocc_a, nocc_b = nelec
    nvrt_a = norb - nocc_a
    nvrt_b = norb - nocc_b

    n_reps: int | tuple[int, int]
    for n_reps, optimize in itertools.product([5, 50, (10, 5)], [False, True]):  # type: ignore
        t2aa = np.zeros((nocc_a, nocc_a, nvrt_a, nvrt_a))
        t2ab = np.zeros((nocc_a, nocc_b, nvrt_a, nvrt_b))
        t2bb = np.zeros((nocc_b, nocc_b, nvrt_b, nvrt_b))
        t1a = np.zeros((nocc_a, nvrt_a))
        t1b = np.zeros((nocc_b, nvrt_b))
        t2 = (t2aa, t2ab, t2bb)
        t1 = (t1a, t1b)
        operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2, t1=t1, n_reps=n_reps, optimize=optimize
        )
        total_n_reps = n_reps if isinstance(n_reps, int) else sum(n_reps)
        assert operator.n_reps == total_n_reps
        actual = len(operator.to_parameters())
        expected = ffsim.UCJOpSpinUnbalanced.n_params(
            norb, total_n_reps, with_final_orbital_rotation=True
        )
        assert actual == expected


def test_t_amplitudes_restrict_indices():
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["Be", (0, 0, 1.1)]],
        basis="6-31g",
        spin=1,
        symmetry="Coov",
    )
    scf = pyscf.scf.ROHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()

    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb

    n_reps = 2
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    pairs_bb = [(p, p + 1) for p in range(norb - 1)]
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)
    )
    other_operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
        operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)),
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab, pairs_bb),
    )
    assert ffsim.approx_eq(operator, other_operator, rtol=1e-12)

    n_reps = 2
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    pairs_bb = []
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)
    )
    other_operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
        operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)),
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab, pairs_bb),
    )
    assert ffsim.approx_eq(operator, other_operator, rtol=1e-12)

    n_reps = 2
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = []
    pairs_bb = [(p, p + 1) for p in range(norb - 1)]
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)
    )
    other_operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
        operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)),
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab, pairs_bb),
    )
    assert ffsim.approx_eq(operator, other_operator, rtol=1e-12)

    n_reps = 2
    pairs_aa = []
    pairs_ab = [(p, p) for p in range(norb)]
    pairs_bb = [(p, p + 1) for p in range(norb - 1)]
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)
    )
    other_operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
        operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)),
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab, pairs_bb),
    )
    assert ffsim.approx_eq(operator, other_operator, rtol=1e-12)

    n_reps = 2
    pairs_aa = []
    pairs_ab = []
    pairs_bb = []
    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        ccsd.t2, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)
    )
    other_operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
        operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab, pairs_bb)),
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab, pairs_bb),
    )
    assert ffsim.approx_eq(operator, other_operator, rtol=1e-12)


def test_validate():
    n_reps = 3
    norb = 4
    eye = np.eye(norb)
    diag_coulomb_mats = np.stack([np.stack([eye, eye, eye]) for _ in range(n_reps)])
    orbital_rotations = np.stack([np.stack([eye, eye]) for _ in range(n_reps)])

    _ = ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=RNG.standard_normal(10),
        orbital_rotations=orbital_rotations,
        validate=False,
    )

    _ = ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=RNG.standard_normal((n_reps, 3, norb, norb)),
        orbital_rotations=orbital_rotations,
        atol=10,
    )

    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=RNG.standard_normal(10),
            orbital_rotations=orbital_rotations,
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=RNG.standard_normal(10),
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=RNG.standard_normal(10),
        )
    with pytest.raises(ValueError, match="dimension"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=np.concatenate([orbital_rotations, orbital_rotations]),
        )
    with pytest.raises(ValueError, match="symmetric"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=RNG.standard_normal((n_reps, 3, norb, norb)),
            orbital_rotations=orbital_rotations,
        )
    with pytest.raises(ValueError, match="unitary"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=RNG.standard_normal((n_reps, 2, norb, norb)),
        )
    with pytest.raises(ValueError, match="unitary"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=RNG.standard_normal((2, norb, norb)),
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=np.stack([np.stack([eye, eye]) for _ in range(n_reps)]),
            orbital_rotations=orbital_rotations,
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=np.stack(
                [np.stack([eye, eye, eye]) for _ in range(n_reps)]
            ),
        )


def test_empty_interaction_pairs():
    """Test constructing operators with empty lists of interaction pairs."""
    norb = 4
    nelec = (2, 1)
    n_reps = 2
    empty_pairs: list[tuple[int, int]] = []
    interaction_pairs = (empty_pairs, empty_pairs, empty_pairs)
    t2 = ffsim.random.random_t2_amplitudes_unrestricted(
        norb, nelec, seed=RNG, dtype=float
    )

    operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        t2, n_reps=n_reps, interaction_pairs=interaction_pairs
    )
    np.testing.assert_allclose(
        operator.diag_coulomb_mats, np.zeros((n_reps, 3, norb, norb))
    )

    params = operator.to_parameters(interaction_pairs=interaction_pairs)
    assert len(params) == ffsim.UCJOpSpinUnbalanced.n_params(
        norb, n_reps, interaction_pairs=interaction_pairs
    )
    roundtrip = ffsim.UCJOpSpinUnbalanced.from_parameters(
        params, norb=norb, n_reps=n_reps, interaction_pairs=interaction_pairs
    )
    assert ffsim.approx_eq(operator, roundtrip)


@pytest.mark.parametrize(
    "n_reps, n_calls",
    [
        # Every sector is optimized
        (None, 3),
        ((2, 2), 3),
        ((1000, 1000), 3),
        # A sector with zero repetitions is not optimized
        ((0, 2), 2),
        ((2, 0), 1),
    ],
)
def test_t_amplitudes_optimize_without_truncation(n_reps, n_calls):
    """Test that optimize=True optimizes regardless of how n_reps compares."""
    norb = 4
    nelec = (2, 2)
    t2 = ffsim.random.random_t2_amplitudes_unrestricted(
        norb, nelec, seed=RNG, dtype=float
    )
    with patch("scipy.optimize.minimize", wraps=scipy.optimize.minimize) as minimize:
        operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2, n_reps=n_reps, optimize=True, options=dict(maxiter=5)
        )
    assert minimize.call_count == n_calls
    if n_reps is not None:
        assert operator.n_reps == sum(n_reps)
