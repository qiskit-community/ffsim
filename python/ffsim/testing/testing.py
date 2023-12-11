# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Testing utilities."""

from __future__ import annotations

import numpy as np

from ffsim.linalg import match_global_phase


def random_nelec(norb: int, *, seed=None) -> tuple[int, int]:
    """Return a random pair of (n_alpha, n_beta) particle numbers.

    Args:
        norb: The number of spatial orbitals.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled pair of (n_alpha, n_beta) particle numbers.
    """
    rng = np.random.default_rng(seed)
    n_alpha, n_beta = rng.integers(norb + 1, size=2)
    return (n_alpha, n_beta)


def random_occupied_orbitals(
    norb: int, nelec: tuple[int, int], *, seed=None
) -> tuple[list[int], list[int]]:
    """Return a random pair of occupied orbitals lists.

    Args:
        norb: The number of spatial orbitals.
        nelec: The number of alpha and beta electrons.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled pair of (occ_a, occ_b) occupied orbitals lists.
    """
    rng = np.random.default_rng(seed)
    n_alpha, n_beta = nelec
    occ_a = list(rng.choice(norb, n_alpha, replace=False))
    occ_b = list(rng.choice(norb, n_beta, replace=False))
    return (occ_a, occ_b)


def assert_allclose_up_to_global_phase(
    actual: np.ndarray,
    desired: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 0,
    equal_nan: bool = True,
    err_msg: str = "",
    verbose: bool = True,
):
    """Check if a == b * exp(i phi) for some real number phi.

    Args:
        actual: A Numpy array.
        desired: Another Numpy array.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: If True, NaNs will compare equal.
        err_msg: The error message to be printed in case of failure.
        verbose: If True, the conflicting values are appended to the error message.

    Raises:
        AssertionError: If a and b are not equal up to global phase, up to the
            specified precision.
    """
    actual, desired = match_global_phase(actual, desired)
    np.testing.assert_allclose(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
        verbose=verbose,
    )
