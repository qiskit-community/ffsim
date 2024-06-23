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

import itertools
from collections.abc import Iterable, Iterator
from typing import cast, overload

import numpy as np

from ffsim.linalg import match_global_phase
from ffsim.spin import Spin


def generate_norb_nelec_spin(
    norb_range: Iterable[int],
) -> Iterator[tuple[int, tuple[int, int], Spin]]:
    """Generate (`norb`, `nelec`, `spin`) tuples for testing.

    Given a range of choices for `norb`, generates all possible
    (`norb`, `nelec`, `spin`) triplets.
    """
    for norb in norb_range:
        for nelec in itertools.product(range(norb + 1), repeat=2):
            for spin in Spin.__members__.values():
                yield norb, cast(tuple[int, int], nelec), spin


def generate_norb_nelec(
    norb_range: Iterable[int],
) -> Iterator[tuple[int, tuple[int, int]]]:
    """Generate (`norb`, `nelec`) tuples for testing.

    Given a range of choices for `norb`, generates all possible (`norb`, `nelec`) pairs.
    """
    for norb in norb_range:
        for nelec in itertools.product(range(norb + 1), repeat=2):
            yield norb, cast(tuple[int, int], nelec)


def generate_norb_nocc(
    norb_range: Iterable[int],
) -> Iterator[tuple[int, int]]:
    """Generate (`norb`, `nocc`) tuples for testing.

    Given a range of choices for `norb`, generates all possible (`norb`, `nocc`) pairs.
    `nocc` refers to the occupation of a single spin species, so it ranges from 0 to
    `norb`.
    """
    for norb in norb_range:
        for nocc in range(norb + 1):
            yield norb, nocc


def generate_norb_spin(norb_range: Iterable[int]) -> Iterator[tuple[int, Spin]]:
    """Generate (`norb`, `spin`) tuples for testing.

    Given a range of choices for `norb`, generates all possible (`norb`, `spin`) pairs.
    """
    for norb in norb_range:
        for spin in Spin.__members__.values():
            yield norb, spin


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


@overload
def random_occupied_orbitals(norb: int, nelec: int, *, seed=None) -> list[int]: ...
@overload
def random_occupied_orbitals(
    norb: int, nelec: tuple[int, int], *, seed=None
) -> tuple[list[int], list[int]]: ...
def random_occupied_orbitals(
    norb: int, nelec: int | tuple[int, int], *, seed=None
) -> list[int] | tuple[list[int], list[int]]:
    """Return a random pair of occupied orbitals lists.

    Args:
        norb: The number of spatial orbitals.
        nelec: Either a single integer representing the number of fermions for a
            spinless system, or a pair of integers storing the numbers of spin alpha
            and spin beta fermions.
        seed: A seed to initialize the pseudorandom number generator.
            Should be a valid input to ``np.random.default_rng``.

    Returns:
        The sampled pair of (occ_a, occ_b) occupied orbitals lists.
    """
    rng = np.random.default_rng(seed)
    if isinstance(nelec, int):
        return list(rng.choice(norb, nelec, replace=False))
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
