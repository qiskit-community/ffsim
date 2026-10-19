# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test orbital optimization."""

from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pyscf
import pyscf.ci
import pytest
import scipy.linalg
import scipy.optimize

import ffsim
from ffsim.linalg.util import (
    special_orthogonal_from_parameters,
    unitary_from_parameters,
)

RNG = np.random.default_rng(241127903049053917326682280086371763733)


@pytest.mark.parametrize("method", [None, "SLSQP"])
@pytest.mark.parametrize("initial_sign", [1, -1])
def test_optimize_orbitals_bounds(method: str | None, initial_sign: int) -> None:
    """Test that bounds limit the additional rotation relative to the initial guess."""
    # Rotating the occupied orbital toward the lower-energy orbital lowers the
    # energy, but the bound limits the additional rotation to 0.2 radians, for a
    # total rotation of 0.3 radians from the unrotated RDMs.
    rdm = ffsim.ReducedDensityMatrix(np.diag([1.0, 0.0]), np.zeros((2, 2, 2, 2)))
    hamiltonian = ffsim.MolecularHamiltonian(
        np.diag([1.0, -1.0]), np.zeros((2, 2, 2, 2))
    )
    initial_rotation = scipy.linalg.expm(np.array([[0.0, 0.1], [-0.1, 0.0]]))
    initial_rotation = initial_rotation @ np.diag([initial_sign, 1])
    iterates: list[np.ndarray] = []

    def callback(x: np.ndarray) -> None:
        iterates.append(x.copy())

    kwargs: dict = {} if method is None else dict(method=method)
    rotation, result = ffsim.optimize_orbitals(
        rdm,
        hamiltonian,
        initial_orbital_rotation=initial_rotation,
        bounds=scipy.optimize.Bounds(0.0, 0.2),
        callback=callback,
        options=dict(maxiter=10),
        return_optimize_result=True,
        **kwargs,
    )

    assert result.success
    assert iterates
    assert all(0 <= x[0] <= 0.2 for x in iterates)
    np.testing.assert_allclose(result.x, [0.2], atol=1e-8)
    np.testing.assert_allclose(result.fun, np.cos(0.6), atol=1e-8)
    np.testing.assert_allclose(
        rdm.rotated(rotation).expectation(hamiltonian), result.fun
    )
    np.testing.assert_allclose(np.linalg.det(rotation), initial_sign)


def _random_problem(
    norb: int, *, dtype: type[float] | type[complex] = float
) -> tuple[ffsim.ReducedDensityMatrix, ffsim.MolecularHamiltonian]:
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=dtype)
    nelec = (norb // 2, norb // 2)
    vec = ffsim.random.random_state_vector(
        ffsim.dim(norb, nelec), seed=RNG, dtype=dtype
    )
    one_rdm, two_rdm = ffsim.rdms(vec, norb, nelec, rank=2, spin_summed=True)
    if dtype is float:
        one_rdm, two_rdm = one_rdm.real, two_rdm.real
    return ffsim.ReducedDensityMatrix(one_rdm, two_rdm), hamiltonian


def _assert_rotation_energy(
    rdm: ffsim.ReducedDensityMatrix,
    hamiltonian: ffsim.MolecularHamiltonian,
    rotation: np.ndarray,
    result: scipy.optimize.OptimizeResult,
) -> None:
    np.testing.assert_allclose(
        rotation @ rotation.T.conj(), np.eye(hamiltonian.norb), atol=1e-12
    )
    np.testing.assert_allclose(
        rdm.rotated(rotation).expectation(hamiltonian), result.fun, atol=1e-10
    )
    np.testing.assert_allclose(
        rdm.expectation(hamiltonian.rotated(rotation.T.conj())), result.fun, atol=1e-10
    )


@pytest.mark.parametrize("norb", [2, 3, 4, 5])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("negative_eigenvalues", [False, True])
@pytest.mark.parametrize("complex_dtype", [False, True])
def test_optimize_orbitals_initial_rotation(
    monkeypatch: pytest.MonkeyPatch,
    norb: int,
    sign: int,
    negative_eigenvalues: bool,
    complex_dtype: bool,
) -> None:
    """Test that zero generator parameters preserve the initial real rotation."""
    rdm, hamiltonian = _random_problem(norb)
    if negative_eigenvalues:
        basis = ffsim.random.random_orthogonal(norb, seed=RNG)
        eigs = np.ones(norb)
        eigs[: 2 if sign == 1 else 1] = -1
        initial = basis @ np.diag(eigs) @ basis.T
    else:
        initial = ffsim.random.random_special_orthogonal(norb, seed=RNG)
        initial[:, 0] *= sign
    if complex_dtype:
        initial = initial.astype(complex)

    def inspect_initial(
        fun: Callable[[np.ndarray], tuple[float, np.ndarray]],
        x0: np.ndarray,
        **kwargs: object,
    ) -> scipy.optimize.OptimizeResult:
        # Inspect before taking a step; maxiter=0 does not ensure this in SciPy.
        value, gradient = fun(x0)
        np.testing.assert_allclose(value, rdm.rotated(initial).expectation(hamiltonian))
        assert x0.shape == gradient.shape == (norb * (norb - 1) // 2,)
        np.testing.assert_array_equal(x0, np.zeros_like(x0))
        return scipy.optimize.OptimizeResult(x=x0, fun=value, jac=gradient)

    monkeypatch.setattr(scipy.optimize, "minimize", inspect_initial)
    rotation, result = ffsim.optimize_orbitals(
        rdm,
        hamiltonian,
        initial_orbital_rotation=initial,
        real=True,
        return_optimize_result=True,
    )
    assert np.isrealobj(rotation)
    np.testing.assert_allclose(rotation, initial, atol=1e-12)
    _assert_rotation_energy(rdm, hamiltonian, rotation, result)


@pytest.mark.parametrize("norb", [3, 4])
@pytest.mark.parametrize("sign", [1, -1])
def test_optimize_orbitals_components(norb: int, sign: int) -> None:
    """Test that real optimization preserves the initial determinant component."""
    rdm, hamiltonian = _random_problem(norb)
    initial = ffsim.random.random_special_orthogonal(norb, seed=RNG)
    initial[:, 0] *= sign
    rotation, result = ffsim.optimize_orbitals(
        rdm,
        hamiltonian,
        initial_orbital_rotation=initial,
        return_optimize_result=True,
        options=dict(maxiter=5),
    )
    np.testing.assert_allclose(np.linalg.det(rotation), sign)
    np.testing.assert_allclose(
        special_orthogonal_from_parameters(result.x, norb) @ initial,
        rotation,
        atol=1e-12,
    )
    _assert_rotation_energy(rdm, hamiltonian, rotation, result)


@pytest.mark.parametrize("sign", [1, -1])
def test_optimize_orbitals_singular_initial_generator(sign: int) -> None:
    """Test convergence from rotations with singular exponential coordinates."""
    # The real logarithm of -I has two pi-rotation blocks. Its exponential's
    # derivative hides downhill directions for this occupied-orbital configuration.
    norb = 4
    rdm = ffsim.ReducedDensityMatrix(
        *ffsim.slater_determinant_rdms(norb, [0, 3], rank=2)
    )
    one_body_tensor = np.zeros((norb, norb))
    one_body_tensor[0, 2] = one_body_tensor[2, 0] = 1
    one_body_tensor[1, 3] = one_body_tensor[3, 1] = 1
    hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor, np.zeros((norb, norb, norb, norb))
    )
    initial = -np.eye(norb)
    initial[:, 0] *= sign
    rotation, result = ffsim.optimize_orbitals(
        rdm,
        hamiltonian,
        initial_orbital_rotation=initial,
        return_optimize_result=True,
    )

    assert result.success
    assert result.nit > 0
    np.testing.assert_allclose(result.fun, -2, atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.linalg.det(rotation), sign)
    _assert_rotation_energy(rdm, hamiltonian, rotation, result)


def test_optimize_orbitals_correlated_determinants() -> None:
    """Test that determinant components can give different correlated-state energies."""
    norb, nelec = 4, (2, 2)
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    linop = ffsim.linear_operator(hamiltonian, norb, nelec)
    energies, states = np.linalg.eigh(linop @ np.eye(ffsim.dim(norb, nelec)))
    one_rdm, two_rdm = ffsim.rdms(states[:, 0], norb, nelec, rank=2, spin_summed=True)
    rdm = ffsim.ReducedDensityMatrix(one_rdm.real, two_rdm.real)
    reflection = np.diag([-1, 1, 1, 1])
    rdm = rdm.rotated(reflection)
    options = dict(gtol=1e-9, ftol=1e-13, maxiter=500)
    rotation, result = ffsim.optimize_orbitals(
        rdm, hamiltonian, return_optimize_result=True, options=options
    )
    rotation_reflected, result_reflected = ffsim.optimize_orbitals(
        rdm,
        hamiltonian,
        initial_orbital_rotation=reflection,
        return_optimize_result=True,
        options=options,
    )
    _assert_rotation_energy(rdm, hamiltonian, rotation, result)
    _assert_rotation_energy(rdm, hamiltonian, rotation_reflected, result_reflected)
    np.testing.assert_allclose(np.linalg.det(rotation), 1)
    np.testing.assert_allclose(np.linalg.det(rotation_reflected), -1)
    np.testing.assert_allclose(result_reflected.fun, energies[0], atol=1e-8, rtol=0)
    assert result.fun - result_reflected.fun > 0.1


@pytest.mark.parametrize("real", [None, True])
def test_optimize_orbitals_reject_one_real_orbital(real: bool | None) -> None:
    """Test that real optimization rejects a single orbital."""
    rdm, hamiltonian = _random_problem(1)
    with pytest.raises(ValueError, match="at least two orbitals"):
        ffsim.optimize_orbitals(rdm, hamiltonian, real=real)


def test_optimize_orbitals_one_complex_orbital_constraints() -> None:
    """Test that complex optimization enforces a phase constraint for one orbital."""
    # A doubly occupied orbital has a nonzero two-RDM contribution.
    rdm = ffsim.ReducedDensityMatrix(np.array([[2.0]]), np.full((1, 1, 1, 1), 2.0))
    hamiltonian = ffsim.random.random_molecular_hamiltonian(1, seed=RNG, dtype=float)
    rotation, result = ffsim.optimize_orbitals(
        rdm,
        hamiltonian,
        real=False,
        return_optimize_result=True,
        method="SLSQP",
        constraints=scipy.optimize.LinearConstraint([[1.0]], [0.2], [0.2]),
    )
    assert result.success
    np.testing.assert_allclose(result.x, [0.2], atol=1e-10)
    np.testing.assert_allclose(rotation, [[np.exp(0.2j)]], atol=1e-10)
    np.testing.assert_allclose(result.fun, rdm.expectation(hamiltonian))
    _assert_rotation_energy(rdm, hamiltonian, rotation, result)


@pytest.mark.parametrize("norb", [1, 2])
def test_optimize_orbitals_reject_imaginary_initial(norb: int) -> None:
    """Test that real optimization rejects nonzero imaginary entries in the guess."""
    rdm, hamiltonian = _random_problem(norb)
    with pytest.raises(ValueError, match="nonzero imaginary"):
        ffsim.optimize_orbitals(
            rdm,
            hamiltonian,
            real=True,
            initial_orbital_rotation=np.eye(norb) * np.exp(1e-15j),
        )


@pytest.mark.parametrize("real", [None, False])
@pytest.mark.parametrize("norb", [1, 2])
def test_optimize_orbitals_complex(
    monkeypatch: pytest.MonkeyPatch, real: bool | None, norb: int
) -> None:
    """Test complex optimization with zero initial generator parameters."""
    rdm, hamiltonian = _random_problem(norb, dtype=complex)
    initial = ffsim.random.random_unitary(norb, seed=RNG)
    minimize = Mock(wraps=scipy.optimize.minimize)
    monkeypatch.setattr(scipy.optimize, "minimize", minimize)
    rotation, result = ffsim.optimize_orbitals(
        rdm,
        hamiltonian,
        initial_orbital_rotation=initial,
        real=real,
        return_optimize_result=True,
    )
    assert result.x.shape == (norb**2,)
    assert np.iscomplexobj(rotation)
    np.testing.assert_array_equal(minimize.call_args.args[1], np.zeros(norb**2))
    np.testing.assert_allclose(
        unitary_from_parameters(result.x, norb) @ initial, rotation, atol=1e-12
    )
    _assert_rotation_energy(rdm, hamiltonian, rotation, result)


@pytest.mark.parametrize("stop", [False, True])
def test_optimize_orbitals_callback(stop: bool) -> None:
    """Test callback energies and optional early termination."""
    rdm, hamiltonian = _random_problem(4)
    initial = ffsim.random.random_orthogonal(4, seed=RNG)
    iterates: list[np.ndarray] = []

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        assert intermediate_result.x.shape == (6,)
        iterates.append(intermediate_result.x.copy())
        rotation = (
            special_orthogonal_from_parameters(intermediate_result.x, 4) @ initial
        )
        np.testing.assert_allclose(
            intermediate_result.fun, rdm.rotated(rotation).expectation(hamiltonian)
        )
        if stop:
            raise StopIteration

    rotation, result = ffsim.optimize_orbitals(
        rdm,
        hamiltonian,
        initial_orbital_rotation=initial,
        return_optimize_result=True,
        callback=callback,
        options=dict(maxiter=2),
    )
    assert iterates
    if stop:
        assert len(iterates) == 1
        assert not result.success
    _assert_rotation_energy(rdm, hamiltonian, rotation, result)


def test_optimize_orbitals_constraints() -> None:
    """Test that constraints apply to the rotation relative to the initial guess."""
    rdm, hamiltonian = _random_problem(2)
    initial = scipy.linalg.expm(np.array([[0.0, 0.1], [-0.1, 0.0]])) @ np.diag([-1, 1])
    rotation, result = ffsim.optimize_orbitals(
        rdm,
        hamiltonian,
        initial_orbital_rotation=initial,
        return_optimize_result=True,
        method="SLSQP",
        constraints=scipy.optimize.LinearConstraint([[1.0]], [0.2], [0.2]),
    )
    assert result.success
    np.testing.assert_allclose(result.x, [0.2], atol=1e-10)
    expected = scipy.linalg.expm(np.array([[0.0, 0.3], [-0.3, 0.0]])) @ np.diag([-1, 1])
    np.testing.assert_allclose(rotation, expected, atol=1e-10)
    _assert_rotation_energy(rdm, hamiltonian, rotation, result)


def test_optimize_orbitals() -> None:
    """Test real and complex orbital optimization of a nitrogen CISD state."""
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, -1.2)], ["N", (0, 0, 1.2)]],
        basis="6-31g",
        symmetry="Dooh",
    )

    # Define active space
    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular data and Hamiltonian
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
    mol_hamiltonian = mol_data.hamiltonian
    norb = mol_data.norb

    # Run CISD
    cisd = pyscf.ci.CISD(scf, frozen=n_frozen).run()
    cisd_energy = cisd.e_tot
    np.testing.assert_allclose(cisd_energy, -108.58532210214092)

    # Get RDMs
    rdm1 = cisd.make_rdm1()[n_frozen:, n_frozen:]
    rdm2 = cisd.make_rdm2()[n_frozen:, n_frozen:, n_frozen:, n_frozen:]
    rdm = ffsim.ReducedDensityMatrix(rdm1, rdm2)

    # Optimize orbitals with real rotations
    orbital_rotation, result = ffsim.optimize_orbitals(
        rdm,
        mol_hamiltonian,
        return_optimize_result=True,
    )
    # Compute energy
    energy = rdm.rotated(orbital_rotation).expectation(mol_hamiltonian)
    energy_alt = rdm.expectation(mol_hamiltonian.rotated(orbital_rotation.T.conj()))
    # Check results
    np.testing.assert_allclose(energy, result.fun)
    np.testing.assert_allclose(energy, energy_alt)
    np.testing.assert_allclose(energy, -108.58613393502857)
    assert np.isrealobj(orbital_rotation)
    assert len(result.x) == norb * (norb - 1) // 2
    assert result.nit <= 7
    assert result.nfev <= 9
    assert result.njev <= 9

    # Optimize orbitals with complex rotations
    orbital_rotation, result = ffsim.optimize_orbitals(
        rdm,
        mol_hamiltonian,
        initial_orbital_rotation=scipy.linalg.expm(
            1e-4 * ffsim.random.random_antihermitian(norb, seed=RNG)
        ),
        return_optimize_result=True,
    )
    # Compute energy
    energy = rdm.rotated(orbital_rotation).expectation(mol_hamiltonian)
    energy_alt = rdm.expectation(mol_hamiltonian.rotated(orbital_rotation.T.conj()))
    # Check results
    np.testing.assert_allclose(energy, result.fun)
    np.testing.assert_allclose(energy, energy_alt)
    np.testing.assert_allclose(energy, -108.58613393502857)
    assert np.iscomplexobj(orbital_rotation)
    assert len(result.x) == norb**2
    assert result.nit <= 8
    assert result.nfev <= 11
    assert result.njev <= 11
