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

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
from opt_einsum import contract

from ffsim.hamiltonians import MolecularHamiltonian
from ffsim.linalg.util import (
    unitary_from_parameters,
    unitary_from_parameters_jax,
    unitary_to_parameters,
)
from ffsim.states.rdm import ReducedDensityMatrix

jax.config.update("jax_enable_x64", True)


def optimize_orbitals(
    rdm: ReducedDensityMatrix,
    hamiltonian: MolecularHamiltonian,
    *,
    initial_orbital_rotation: np.ndarray | None = None,
    real: bool | None = None,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    return_optimize_result: bool = False,
) -> np.ndarray | tuple[np.ndarray, scipy.optimize.OptimizeResult]:
    """Find orbitals that minimize the energy of a pair of one- and two-RDMs.

    Uses `scipy.optimize.minimize`_ to find an orbital rotation that minimizes the
    energy of a pair of one- and two-RDMs with respect to a molecualar Hamiltonian.

    The minimized energy can be computed from the returned orbital rotation as

    .. code::

        rdm.rotated(orbital_rotation).expectation(mol_hamiltonian)

    or

    .. code::

        rdm.expectation(mol_hamiltonian.rotated(orbital_rotation.T.conj()))

    Args:
        rdm: The reduced density matrices.
        hamiltonian: The Hamiltonian.
        initial_orbital_rotation: An initial guess for the orbital rotation. If not
            provided, the identity matrix will be used.
        real: Whether to restrict the optimization to real-valued orbital rotations.
            The default behavior is to restrict to real-valued orbital rotations
            if the reduced density matrices, Hamiltonian, and initial orbital rotation
            (if one was given) are all real-valued, and to allow complex orbital
            rotations if any of these are complex-valued.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
        return_optimize_result: Whether to also return the `OptimizeResult`_ returned
            by `scipy.optimize.minimize`_.

    Returns:
        The orbital rotation, which, when applied to the reduced density matrix
        (or conjugated and applied to the Hamiltonian), minimizes its energy.
        If `return_optimize_result` is set to ``True``, the `OptimizeResult`_ returned
        by `scipy.optimize.minimize`_ is also returned.

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    .. _OptimizeResult: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    """
    if real is None:
        real = not any(
            np.iscomplexobj(x)
            for x in [
                rdm.one_rdm,
                rdm.two_rdm,
                hamiltonian.one_body_tensor,
                hamiltonian.two_body_tensor,
                # complicate the next line to satisfy mypy
                initial_orbital_rotation if initial_orbital_rotation is not None else 0,
            ]
        )

    norb = hamiltonian.norb
    one_rdm = jnp.array(rdm.one_rdm)
    two_rdm = jnp.array(rdm.two_rdm)
    one_body_tensor = jnp.array(hamiltonian.one_body_tensor)
    two_body_tensor = jnp.array(hamiltonian.two_body_tensor)

    def fun(x: np.ndarray):
        orbital_rotation = unitary_from_parameters_jax(x, dim=norb, real=real)
        one_rdm_rotated = contract(
            "ab,Aa,Bb->AB",
            one_rdm,
            orbital_rotation.conj(),
            orbital_rotation,
            optimize="greedy",
        )
        two_rdm_rotated = contract(
            "abcd,Aa,Bb,Cc,Dd->ABCD",
            two_rdm,
            orbital_rotation.conj(),
            orbital_rotation,
            orbital_rotation.conj(),
            orbital_rotation,
            optimize="greedy",
        )
        return (
            hamiltonian.constant
            + contract("ab,ab->", one_body_tensor, one_rdm_rotated)
            + 0.5 * contract("abcd,abcd->", two_body_tensor, two_rdm_rotated)
        ).real

    value_and_grad = jax.value_and_grad(fun)

    if initial_orbital_rotation is None:
        initial_orbital_rotation = np.eye(norb)

    result = scipy.optimize.minimize(
        value_and_grad,
        unitary_to_parameters(initial_orbital_rotation, real=real),
        method=method,
        jac=True,
        callback=callback,
        options=options,
    )

    orbital_rotation = unitary_from_parameters(result.x, dim=norb, real=real)

    if return_optimize_result:
        return orbital_rotation, result
    return orbital_rotation
