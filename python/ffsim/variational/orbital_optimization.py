# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from ffsim.hamiltonians import MolecularHamiltonian
from ffsim.linalg.util import (
    rotate_one_body_tensor,
    rotate_two_body_tensor,
    special_orthogonal_from_parameters,
    special_orthogonal_from_parameters_jax,
    unitary_from_parameters,
    unitary_from_parameters_jax,
)
from ffsim.states.rdm import ReducedDensityMatrix

jax.config.update("jax_enable_x64", True)


@functools.cache
def _make_optimize_orbitals_value_and_grad(norb: int, real: bool):
    """Build a jitted value-and-gradient function for the orbital energy objective.

    The result is cached and reused across optimizer iterations and across calls with
    the same static structure ``(norb, real)``, so the loss is traced and compiled only
    once per structure. The reduced density matrices, one- and two-body tensors, and
    constant are passed as runtime arguments (they do not affect the traced graph), and
    the gradient is taken with respect to the flat variable vector ``x`` only.
    """

    def loss(
        x: jax.Array,
        one_rdm: jax.Array,
        two_rdm: jax.Array,
        one_body_tensor: jax.Array,
        two_body_tensor: jax.Array,
        constant: jax.Array,
    ) -> jax.Array:
        from_parameters = (
            special_orthogonal_from_parameters_jax
            if real
            else unitary_from_parameters_jax
        )
        orbital_rotation = from_parameters(x, dim=norb)
        # Reduced density matrices are dual to operator tensors, so they transform under
        # the conjugate of the orbital rotation: rotating the RDMs by U gives the same
        # energy as rotating the Hamiltonian by U^dagger. The rotation helpers implement
        # the operator convention, so they receive the conjugate rotation here.
        conjugated = orbital_rotation.conj()
        one_rdm_rotated = rotate_one_body_tensor(one_rdm, conjugated)
        two_rdm_rotated = rotate_two_body_tensor(two_rdm, conjugated, conjugated)
        return (
            constant
            + jnp.einsum("ab,ab->", one_body_tensor, one_rdm_rotated)
            + 0.5 * jnp.einsum("abcd,abcd->", two_body_tensor, two_rdm_rotated)
        ).real

    return jax.jit(jax.value_and_grad(loss, argnums=0))


def optimize_orbitals(
    rdm: ReducedDensityMatrix,
    hamiltonian: MolecularHamiltonian,
    *,
    initial_orbital_rotation: np.ndarray | None = None,
    real: bool | None = None,
    return_optimize_result: bool = False,
    **optimize_kwargs,
) -> np.ndarray | tuple[np.ndarray, scipy.optimize.OptimizeResult]:
    """Find orbitals that minimize the energy of a pair of one- and two-RDMs.

    Uses :func:`scipy.optimize.minimize` to find an orbital rotation that minimizes the
    energy of a pair of one- and two-RDMs with respect to a molecular Hamiltonian.

    Real optimization stays in the determinant component of the initial rotation.
    If no initial rotation is given, it searches rotations with determinant +1.
    Optimization starts with zero generator parameters for an additional rotation
    applied after the initial rotation, which remains fixed throughout optimization.

    The minimized energy can be computed from the returned orbital rotation as

    .. code::

        rdm.rotated(orbital_rotation).expectation(mol_hamiltonian)

    or

    .. code::

        rdm.expectation(mol_hamiltonian.rotated(orbital_rotation.T.conj()))

    See :doc:`/how-to-guides/optimize-orbitals` for a how-to guide.

    Args:
        rdm: The reduced density matrices.
        hamiltonian: The Hamiltonian.
        initial_orbital_rotation: An initial guess for the orbital rotation. If not
            provided, the identity matrix will be used. Real optimization accepts
            either determinant sign, including complex-typed matrices whose imaginary
            entries are all zero.
        real: Whether to restrict the optimization to real-valued orbital rotations.
            The default behavior is to restrict to real-valued orbital rotations
            if the reduced density matrices, Hamiltonian, and initial orbital rotation
            (if one was given) are all real-valued, and to allow complex orbital
            rotations if any of these are complex-valued.
        return_optimize_result: Whether to also return the
            :class:`~scipy.optimize.OptimizeResult` returned by
            :func:`scipy.optimize.minimize`.
        optimize_kwargs: Keyword arguments to pass to
            :func:`scipy.optimize.minimize`. The default method is
            ``"L-BFGS-B"``. Bounds, constraints, and callbacks use the generator
            parameters of the additional rotation relative to the initial guess.

    Returns:
        The orbital rotation that minimizes the energy when applied to the reduced
        density matrix, or when its adjoint is applied to the Hamiltonian.
        If ``return_optimize_result`` is set to ``True``, the
        :class:`~scipy.optimize.OptimizeResult` returned by
        :func:`scipy.optimize.minimize` is also returned. The parameters ``result.x``
        describe the additional rotation exp(K), and the returned rotation is
        exp(K) U0, where U0 is the initial rotation. The generator K is real
        skew-symmetric for real optimization and anti-Hermitian for complex
        optimization.

    Raises:
        ValueError: The initial rotation has nonzero imaginary entries when
            ``real=True``, or real optimization is requested for one orbital.
    """
    norb = hamiltonian.norb
    if initial_orbital_rotation is None:
        initial_orbital_rotation = np.eye(norb)

    if real is None:
        real = not any(
            np.iscomplexobj(x)
            for x in [
                rdm.one_rdm,
                rdm.two_rdm,
                hamiltonian.one_body_tensor,
                hamiltonian.two_body_tensor,
                initial_orbital_rotation,
            ]
        )

    if real:
        if np.any(initial_orbital_rotation.imag):
            raise ValueError(
                "The initial orbital rotation has a nonzero imaginary part, but "
                "real=True requires real entries. Set real=False to allow complex "
                "orbital rotations."
            )
        initial_orbital_rotation = initial_orbital_rotation.real

        if norb == 1:
            raise ValueError(
                "Real orbital optimization requires at least two orbitals. "
                "Set real=False to allow complex orbital rotations."
            )

    # The exponential has a nonsingular derivative at zero. Using the initial
    # rotation as a fixed reference avoids singularities in its matrix logarithm's
    # exponential coordinates, such as two simultaneous rotations by pi.
    rdm = rdm.rotated(initial_orbital_rotation)
    x0 = np.zeros(norb * (norb - 1) // 2 if real else norb**2)

    one_rdm = jnp.asarray(rdm.one_rdm)
    two_rdm = jnp.asarray(rdm.two_rdm)
    one_body_tensor = jnp.asarray(hamiltonian.one_body_tensor)
    two_body_tensor = jnp.asarray(hamiltonian.two_body_tensor)
    constant = jnp.asarray(hamiltonian.constant)

    value_and_grad = _make_optimize_orbitals_value_and_grad(norb, real)

    def scipy_func(x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(
            jnp.asarray(x),
            one_rdm,
            two_rdm,
            one_body_tensor,
            two_body_tensor,
            constant,
        )
        return float(value), np.asarray(grad)

    optimize_kwargs.setdefault("method", "L-BFGS-B")
    result = scipy.optimize.minimize(scipy_func, x0, jac=True, **optimize_kwargs)

    from_parameters = (
        special_orthogonal_from_parameters if real else unitary_from_parameters
    )
    orbital_rotation = from_parameters(result.x, dim=norb) @ initial_orbital_rotation

    if return_optimize_result:
        return orbital_rotation, result
    return orbital_rotation
