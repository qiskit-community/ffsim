# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear algebra utilities."""

from ffsim.linalg.double_factorized_decomposition import (
    double_factorized,
    double_factorized_t2,
    double_factorized_t2_alpha_beta,
    modified_cholesky,
)
from ffsim.linalg.givens import (
    GivensRotation,
    apply_matrix_to_slices,
    givens_decomposition,
)
from ffsim.linalg.linalg import (
    expm_multiply_taylor,
    lup,
    match_global_phase,
    one_hot,
    reduced_matrix,
)
from ffsim.linalg.predicates import (
    is_antihermitian,
    is_hermitian,
    is_orthogonal,
    is_real_symmetric,
    is_special_orthogonal,
    is_unitary,
)

__all__ = [
    "GivensRotation",
    "apply_matrix_to_slices",
    "double_factorized",
    "double_factorized_t2",
    "double_factorized_t2_alpha_beta",
    "expm_multiply_taylor",
    "givens_decomposition",
    "is_antihermitian",
    "is_hermitian",
    "is_orthogonal",
    "is_real_symmetric",
    "is_special_orthogonal",
    "is_unitary",
    "lup",
    "match_global_phase",
    "modified_cholesky",
    "one_hot",
    "reduced_matrix",
]
