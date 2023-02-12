# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections.abc import Sequence

import numpy as np
from pyscf.fci import cistring
from scipy.special import comb


def slater_determinant(
    n_orbitals: int,
    occupied_orbitals: tuple[Sequence[int], Sequence[int]],
    dtype: type = complex,
) -> np.ndarray:
    alpha_orbitals, beta_orbitals = occupied_orbitals
    n_alpha = len(alpha_orbitals)
    n_beta = len(beta_orbitals)
    dim1 = comb(n_orbitals, n_alpha, exact=True)
    dim2 = comb(n_orbitals, n_beta, exact=True)
    slater = np.zeros((dim1, dim2), dtype=dtype)
    alpha_bits = np.zeros(n_orbitals, dtype=bool)
    alpha_bits[list(alpha_orbitals)] = 1
    alpha_string = int("".join("1" if b else "0" for b in alpha_bits[::-1]), base=2)
    alpha_index = cistring.str2addr(n_orbitals, n_alpha, alpha_string)
    beta_bits = np.zeros(n_orbitals, dtype=bool)
    beta_bits[list(beta_orbitals)] = 1
    beta_string = int("".join("1" if b else "0" for b in beta_bits[::-1]), base=2)
    beta_index = cistring.str2addr(n_orbitals, n_beta, beta_string)
    slater[alpha_index, beta_index] = 1
    return slater.reshape(-1)


def one_hot(dim: int, index: int, *, dtype=float):
    """Return a vector of all zeros except for a one at ``index``."""
    vec = np.zeros(dim, dtype=dtype)
    vec[index] = 1
    return vec
