# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import scipy.linalg

import ffsim


def _exponentiate_t1(t1: np.ndarray, norb: int, nocc: int) -> np.ndarray:
    generator = np.zeros((norb, norb), dtype=complex)
    generator[:nocc, nocc:] = t1
    generator[nocc:, :nocc] = -t1.T
    return scipy.linalg.expm(generator)


def test_lucj_ansatz_operator_roundtrip():
    norb = 2
    n_reps = 2
    diag_coulomb_mats_alpha_alpha = np.array(
        [ffsim.random.random_real_symmetric_matrix(norb) for _ in range(n_reps)]
    )
    diag_coulomb_mats_alpha_beta = np.array(
        [ffsim.random.random_real_symmetric_matrix(norb) for _ in range(n_reps)]
    )
    orbital_rotations = np.array(
        [ffsim.random.random_unitary(norb) for _ in range(n_reps)]
    )
    final_orbital_rotation = ffsim.random.random_unitary(norb)
    operator = ffsim.LUCJOperator(
        diag_coulomb_mats_alpha_alpha=diag_coulomb_mats_alpha_alpha,
        diag_coulomb_mats_alpha_beta=diag_coulomb_mats_alpha_beta,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )
    roundtripped = ffsim.LUCJOperator.from_parameters(
        operator.to_parameters(),
        norb=norb,
        n_reps=n_reps,
        with_final_orbital_rotation=True,
    )
    np.testing.assert_allclose(
        roundtripped.diag_coulomb_mats_alpha_alpha,
        operator.diag_coulomb_mats_alpha_alpha,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        roundtripped.diag_coulomb_mats_alpha_beta,
        operator.diag_coulomb_mats_alpha_beta,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        roundtripped.orbital_rotations,
        operator.orbital_rotations,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        roundtripped.final_orbital_rotation,
        operator.final_orbital_rotation,
        atol=1e-12,
    )


def test_lucj_ansatz_operator_t_amplitudes_roundtrip():
    nocc = 5

    rng = np.random.default_rng()

    t2 = ffsim.random.random_two_body_tensor_real(nocc)
    t1 = rng.standard_normal((nocc, nocc))

    operator = ffsim.LUCJOperator.from_t_amplitudes(t2, t1=t1)
    t2_roundtripped, t1_roundtripped = operator.to_t_amplitudes(nocc=nocc)

    np.testing.assert_allclose(
        t2_roundtripped,
        t2,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _exponentiate_t1(t1_roundtripped, norb=2 * nocc, nocc=nocc),
        _exponentiate_t1(t1, norb=2 * nocc, nocc=nocc),
        atol=1e-12,
    )
