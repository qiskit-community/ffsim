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

from ffsim.linalg.double_factorized import (
    double_factorized,
    double_factorized_t2,
    modified_cholesky,
)
from ffsim.linalg.givens import (
    apply_matrix_to_slices,
    givens_decomposition,
    givens_matrix,
)
from ffsim.linalg.linalg import (
    expm_multiply_taylor,
    lup,
)
from ffsim.linalg.predicates import (
    is_antihermitian,
    is_hermitian,
    is_orthogonal,
    is_real_symmetric,
    is_special_orthogonal,
    is_unitary,
)
