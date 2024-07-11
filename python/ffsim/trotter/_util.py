# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Trotter simulation utilities."""

from __future__ import annotations

from collections.abc import Iterator


def simulate_trotter_step_iterator(
    n_terms: int, time: float, order: int = 0
) -> Iterator[tuple[int, float]]:
    if order == 0:
        for i in range(n_terms):
            yield i, time
    else:
        yield from simulate_trotter_step_iterator_symmetric(n_terms, time, order)


def simulate_trotter_step_iterator_symmetric(
    n_terms: int, time: float, order: int
) -> Iterator[tuple[int, float]]:
    if order == 1:
        for i in range(n_terms - 1):
            yield i, time / 2
        yield n_terms - 1, time
        for i in reversed(range(n_terms - 1)):
            yield i, time / 2
    else:
        split_time = time / (4 - 4 ** (1 / (2 * order - 1)))
        for _ in range(2):
            yield from simulate_trotter_step_iterator_symmetric(
                n_terms, split_time, order - 1
            )
        yield from simulate_trotter_step_iterator_symmetric(
            n_terms, time - 4 * split_time, order - 1
        )
        for _ in range(2):
            yield from simulate_trotter_step_iterator_symmetric(
                n_terms, split_time, order - 1
            )
