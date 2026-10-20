# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for spinless unitary cluster Jastrow ansatz."""

import itertools
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
import pyscf
import pyscf.cc
import pytest
import scipy.optimize

import ffsim
from ffsim.variational.ucj_spinless import ucj_op_spinless_from_parameters_jax

RNG = np.random.default_rng(290171910829038264156178965281743000184)


def test_n_params():
    for norb, n_reps, with_final_orbital_rotation in itertools.product(
        [1, 2, 3], [1, 2, 3], [False, True]
    ):
        operator = ffsim.random.random_ucj_op_spinless(
            norb, n_reps=n_reps, with_final_orbital_rotation=with_final_orbital_rotation
        )
        actual = ffsim.UCJOpSpinless.n_params(
            norb, n_reps, with_final_orbital_rotation=with_final_orbital_rotation
        )
        expected = len(operator.to_parameters())
        assert actual == expected

        interaction_pairs = list(
            itertools.combinations_with_replacement(range(norb), 2)
        )[:norb]

        actual = ffsim.UCJOpSpinless.n_params(
            norb,
            n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        expected = len(operator.to_parameters(interaction_pairs=interaction_pairs))
        assert actual == expected

        with pytest.raises(ValueError, match="triangular"):
            actual = ffsim.UCJOpSpinless.n_params(
                norb,
                n_reps,
                interaction_pairs=[(1, 0)],
            )
        with pytest.raises(ValueError, match="Duplicate"):
            actual = ffsim.UCJOpSpinless.n_params(
                norb,
                n_reps,
                interaction_pairs=[(1, 0), (1, 0)],
            )
        with pytest.raises(ValueError, match="outside the range"):
            actual = ffsim.UCJOpSpinless.n_params(
                norb,
                n_reps,
                interaction_pairs=[(0, norb)],
            )


def test_parameters_roundtrip_all_to_all():
    norb = 5
    n_reps = 2

    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_ucj_op_spinless(
            norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = ffsim.UCJOpSpinless.from_parameters(
            operator.to_parameters(),
            norb=norb,
            n_reps=n_reps,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(roundtripped, operator)


def test_parameters_roundtrip_interaction_pairs():
    norb = 5
    n_reps = 2
    interaction_pairs = [(0, 1)]

    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_ucj_op_spinless(
            norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = ffsim.UCJOpSpinless.from_parameters(
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
    interaction_pairs_list = [None, [(0, 1), (2, 2)], []]
    for n_reps, interaction_pairs, with_final_orbital_rotation in itertools.product(
        [0, 1, 2], interaction_pairs_list, [False, True]
    ):
        n_params = ffsim.UCJOpSpinless.n_params(
            norb,
            n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        params = RNG.standard_normal(n_params)
        operator = ffsim.UCJOpSpinless.from_parameters(
            params,
            norb=norb,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        diag_coulomb_mats, orbital_rotations, final_orbital_rotation = (
            ucj_op_spinless_from_parameters_jax(
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

    # Construct UCJ operator. The restricted CCSD t2 amplitudes are converted to the
    # antisymmetric amplitudes of the spinless doubles operator.
    t2 = ccsd.t2 - ccsd.t2.transpose(1, 0, 2, 3)
    operator = ffsim.UCJOpSpinless.from_t_amplitudes(t2, t1=ccsd.t1)

    # Compute energy using entanglement forging
    reference_occupations_spatial = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 1, 4),
        (1, 2, 3),
        (1, 2, 4),
        (2, 3, 4),
    ]
    reference_occupations = list(
        zip(reference_occupations_spatial, reference_occupations_spatial)
    )
    energy, _ = ffsim.multireference_state_prod(
        mol_hamiltonian,
        (operator, operator),
        reference_occupations,
        norb=norb,
        nelec=nelec,
    )
    energy_alt, _ = ffsim.multireference_state(
        mol_hamiltonian,
        operator,
        reference_occupations,
        norb=norb,
        nelec=nelec,
    )
    np.testing.assert_allclose(energy, energy_alt)
    np.testing.assert_allclose(energy, -108.512645)


def test_t_amplitudes_first_order_matches_uccsd():
    """Test from_t_amplitudes agrees with UCCSD to first order in the t2 amplitudes."""
    norb = 5
    nocc = 3
    nvrt = norb - nocc
    eps = 1e-6
    tol = 1e-12
    t2 = eps * ffsim.random.random_t2_amplitudes_spinless(
        norb, nocc, seed=RNG, dtype=float
    )
    ucj_op = ffsim.UCJOpSpinless.from_t_amplitudes(t2, tol=tol)
    reference_state = ffsim.hartree_fock_state(norb, nocc)
    ucj_state = ffsim.apply_unitary(reference_state, ucj_op, norb=norb, nelec=nocc)
    uccsd_op = ffsim.UCCSDOpSpinlessReal(t1=np.zeros((nocc, nvrt)), t2=t2)
    uccsd_state = ffsim.apply_unitary(reference_state, uccsd_op, norb=norb, nelec=nocc)
    np.testing.assert_allclose(
        (ucj_state - reference_state) / eps,
        (uccsd_state - reference_state) / eps,
        atol=1e-4,
    )


def test_t_amplitudes_validate():
    """Test from_t_amplitudes validates the antisymmetry of the t2 amplitudes."""
    norb = 5
    nocc = 3
    # Restricted amplitudes are rejected, and the error explains how to convert them
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=RNG, dtype=float)
    with pytest.raises(ValueError, match=r"t2 - t2\.transpose\(1, 0, 2, 3\)"):
        ffsim.UCJOpSpinless.from_t_amplitudes(t2)
    ffsim.UCJOpSpinless.from_t_amplitudes(t2 - t2.transpose(1, 0, 2, 3))
    # Validation can be skipped
    ffsim.UCJOpSpinless.from_t_amplitudes(t2, validate=False)


def test_t_amplitudes_random_n_reps():
    norb = 5
    nocc = 3
    nvrt = norb - nocc

    for n_reps in [3, 15]:
        t2 = ffsim.random.random_t2_amplitudes_spinless(
            norb, nocc, seed=RNG, dtype=float
        )
        t1 = RNG.standard_normal((nocc, nvrt))
        operator = ffsim.UCJOpSpinless.from_t_amplitudes(t2, t1=t1, n_reps=n_reps)
        assert operator.n_reps == n_reps
        actual = len(operator.to_parameters())
        expected = ffsim.UCJOpSpinless.n_params(
            norb, n_reps, with_final_orbital_rotation=True
        )
        assert actual == expected


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
        ffsim.UCJOpSpinless.from_t_amplitudes(
            t2,
            n_reps=1,
            multi_stage_start=2,
            multi_stage_step=1,
            optimize=optimize,
            validate=False,
            **kwargs,
        )

    assert minimize.call_count == (2 if optimize else 0)
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
    nocc = 3
    nvrt = norb - nocc
    n_reps = 3
    t2 = ffsim.random.random_t2_amplitudes_spinless(norb, nocc, seed=RNG, dtype=float)
    t1 = RNG.standard_normal((nocc, nvrt))
    interaction_pairs = [(p, p) for p in range(norb)]
    interaction_pairs.extend([(p, p + 1) for p in range(norb - 1)])
    operator = ffsim.UCJOpSpinless.from_t_amplitudes(
        t2,
        t1=t1,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
        optimize=True,
        options={"maxiter": 10},
    )
    diag_coulomb_mask = np.zeros((norb, norb), dtype=bool)
    rows, cols = zip(*interaction_pairs)
    diag_coulomb_mask[rows, cols] = True
    diag_coulomb_mask[cols, rows] = True
    np.testing.assert_allclose(
        operator.diag_coulomb_mats, operator.diag_coulomb_mats * diag_coulomb_mask
    )
    assert operator.n_reps == n_reps
    actual = len(operator.to_parameters())
    expected = ffsim.UCJOpSpinless.n_params(
        norb, n_reps, with_final_orbital_rotation=True
    )
    assert actual == expected


def test_t_amplitudes_zero_n_reps():
    norb = 5
    nocc = 3
    nvrt = norb - nocc

    for n_reps, optimize in itertools.product([3, 15], [False, True]):
        t2 = np.zeros((nocc, nocc, nvrt, nvrt))
        t1 = np.zeros((nocc, nvrt))
        operator = ffsim.UCJOpSpinless.from_t_amplitudes(
            t2, t1=t1, n_reps=n_reps, optimize=optimize
        )
        assert operator.n_reps == n_reps
        actual = len(operator.to_parameters())
        expected = ffsim.UCJOpSpinless.n_params(
            norb, n_reps, with_final_orbital_rotation=True
        )
        assert actual == expected


def test_t_amplitudes_restrict_indices():
    # Build an H2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, -0.9)], ["H", (0, 0, 0.9)]],
        basis="sto-6g",
        symmetry="Dooh",
    )
    scf = pyscf.scf.RHF(mol).run()
    ccsd = pyscf.cc.CCSD(scf).run()

    # Get molecular data and molecular Hamiltonian (one- and two-body tensors)
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb = mol_data.norb

    # Construct UCJ operator
    n_reps = 2
    interaction_pairs = [(p, p + 1) for p in range(norb - 1)]

    t2 = ccsd.t2 - ccsd.t2.transpose(1, 0, 2, 3)
    operator = ffsim.UCJOpSpinless.from_t_amplitudes(
        t2, n_reps=n_reps, interaction_pairs=interaction_pairs
    )
    other_operator = ffsim.UCJOpSpinless.from_parameters(
        operator.to_parameters(interaction_pairs=interaction_pairs),
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
    )

    assert ffsim.approx_eq(operator, other_operator, rtol=1e-12)


def test_validate():
    n_reps = 3
    norb = 4
    eye = np.eye(norb)
    diag_coulomb_mats = np.stack([eye for _ in range(n_reps)])
    orbital_rotations = np.stack([eye for _ in range(n_reps)])

    _ = ffsim.UCJOpSpinless(
        diag_coulomb_mats=RNG.standard_normal(10),
        orbital_rotations=orbital_rotations,
        validate=False,
    )

    _ = ffsim.UCJOpSpinless(
        diag_coulomb_mats=RNG.standard_normal((n_reps, norb, norb)),
        orbital_rotations=orbital_rotations,
        atol=10,
    )

    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinless(
            diag_coulomb_mats=RNG.standard_normal(10),
            orbital_rotations=orbital_rotations,
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinless(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=RNG.standard_normal(10),
        )
    with pytest.raises(ValueError, match="shape"):
        _ = ffsim.UCJOpSpinless(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=RNG.standard_normal(10),
        )
    with pytest.raises(ValueError, match="dimension"):
        _ = ffsim.UCJOpSpinless(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=np.concatenate([orbital_rotations, orbital_rotations]),
        )
    with pytest.raises(ValueError, match="symmetric"):
        _ = ffsim.UCJOpSpinless(
            diag_coulomb_mats=RNG.standard_normal((n_reps, norb, norb)),
            orbital_rotations=orbital_rotations,
        )
    with pytest.raises(ValueError, match="unitary"):
        _ = ffsim.UCJOpSpinless(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=RNG.standard_normal((n_reps, norb, norb)),
        )
    with pytest.raises(ValueError, match="unitary"):
        _ = ffsim.UCJOpSpinless(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=RNG.standard_normal((norb, norb)),
        )


def test_empty_interaction_pairs():
    """Test constructing operators with an empty list of interaction pairs."""
    norb = 4
    nocc = 2
    n_reps = 2
    t2 = ffsim.random.random_t2_amplitudes_spinless(norb, nocc, seed=RNG, dtype=float)

    operator = ffsim.UCJOpSpinless.from_t_amplitudes(
        t2, n_reps=n_reps, interaction_pairs=[]
    )
    np.testing.assert_allclose(
        operator.diag_coulomb_mats, np.zeros((n_reps, norb, norb))
    )

    params = operator.to_parameters(interaction_pairs=[])
    assert len(params) == ffsim.UCJOpSpinless.n_params(
        norb, n_reps, interaction_pairs=[]
    )
    roundtrip = ffsim.UCJOpSpinless.from_parameters(
        params, norb=norb, n_reps=n_reps, interaction_pairs=[]
    )
    assert ffsim.approx_eq(operator, roundtrip)


@pytest.mark.parametrize("n_reps", [None, 3, "n_terms", 100])
def test_t_amplitudes_optimize_without_truncation(n_reps):
    """Test that optimize=True optimizes regardless of how n_reps compares."""
    norb, nocc = 4, 2
    t2 = ffsim.random.random_t2_amplitudes_spinless(norb, nocc, seed=RNG, dtype=float)
    n_terms = len(ffsim.linalg.double_factorized_t2(0.5 * t2)[0])
    if n_reps == "n_terms":
        n_reps = n_terms
    pairs = [(p, p + 1) for p in range(norb - 1)]
    with patch("scipy.optimize.minimize", wraps=scipy.optimize.minimize) as minimize:
        operator = ffsim.UCJOpSpinless.from_t_amplitudes(
            t2,
            n_reps=n_reps,
            interaction_pairs=pairs,
            optimize=True,
            options=dict(maxiter=5),
        )
    assert minimize.call_count == 1
    assert operator.n_reps == (n_terms if n_reps is None else n_reps)
