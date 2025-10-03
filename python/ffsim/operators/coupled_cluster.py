# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Coupled cluster operators."""

import itertools

import numpy as np

from ffsim._lib import FermionOperator
from ffsim.operators.fermion_action import cre_a, cre_b, des_a, des_b


def singles_excitations_restricted(t1: np.ndarray) -> FermionOperator:
    r"""Restricted singles excitations operator.

    The restricted singles excitations operator is

    .. math::

        T_1 = \sum_{ia} t_{ia}
        (a^\dagger_{\alpha, a} a_{\alpha, i} + a^\dagger_{\beta, a} a_{\beta, i})

    where :math:`i` runs over occupied orbitals, :math:`a` runs over virtual orbitals,
    and :math:`t_{ia}` are the singles amplitudes.

    Args:
        t1: The singles amplitudes tensor of shape ``(nocc, nvrt)``, where ``nocc`` is
            the number of occupied orbitals and ``nvrt`` is the number of virtual
            orbitals.

    Returns:
        The singles excitations operator.
    """
    nocc, nvrt = t1.shape
    op = FermionOperator({})
    for i, a in itertools.product(range(nocc), range(nvrt)):
        coeff = t1[i, a]
        op += FermionOperator(
            {
                (cre_a(nocc + a), des_a(i)): coeff,
                (cre_b(nocc + a), des_b(i)): coeff,
            }
        )
    return op


def singles_excitations_unrestricted(
    t1: tuple[np.ndarray, np.ndarray],
) -> FermionOperator:
    r"""Unrestricted singles excitations operator.

    The unrestricted singles excitations operator is

    .. math::

        T_1 = \sum_{ia} t^{(\alpha)}_{ia}
        a^\dagger_{\alpha, a} a_{\alpha, i}
        + \sum_{IA} t^{(\beta)}_{IA}
        a^\dagger_{\beta, A} a_{\beta, I}

    where

    - :math:`i` runs over occupied spin-up orbitals,
    - :math:`a` runs over virtual spin-up orbitals,
    - :math:`I` runs over occupied spin-down orbitals,
    - :math:`A` runs over virtual spin-down orbitals,
    - :math:`t^{(\alpha})_{ia}` are the spin-up singles amplitudes, and
    - :math:`t^{(\beta})_{IA}` are the spin-down singles amplitudes.

    Args:
        t1: The singles amplitudes. This should be a pair of Numpy arrays,
            ``(t1a, t1b)``, containing the spin-up and spin-down singles amplitudes.
            ``t1a`` should have shape (``nocc_a, nvrt_a``), where ``nocc_a`` is the
            number of occupied spin-up orbitals and ``nvrt_a`` is the number of
            virtual spin-up orbitals.
            ``t1b`` should have shape (``nocc_b, nvrt_b``), where ``nocc_b`` is the
            number of occupied spin-down orbitals and ``nvrt_b`` is the number of
            virtual spin-down orbitals.

    Returns:
        The singles excitations operator.
    """
    t1a, t1b = t1
    nocc_a, nvrt_a = t1a.shape
    nocc_b, nvrt_b = t1b.shape
    op = FermionOperator({})
    for i, a in itertools.product(range(nocc_a), range(nvrt_a)):
        coeff = t1a[i, a]
        op += FermionOperator({(cre_a(nocc_a + a), des_a(i)): coeff})
    for i, a in itertools.product(range(nocc_b), range(nvrt_b)):
        coeff = t1b[i, a]
        op += FermionOperator({(cre_b(nocc_b + a), des_b(i)): coeff})
    return op


def doubles_excitations_restricted(t2: np.ndarray) -> FermionOperator:
    r"""Restricted doubles excitations operator.

    The restricted doubles excitations operator is

    .. math::

        T_2 = \sum_{ijab} t_{ijab} \left[
        \frac12 \left(
        a^\dagger_{\alpha, a} a^\dagger_{\alpha, b} a_{\alpha, j} a_{\alpha, i} +
        a^\dagger_{\beta, a} a^\dagger_{\beta, b} a_{\beta, j} a_{\beta, i} \right)
        + a^\dagger_{\alpha, a} a^\dagger_{\beta, b} a_{\beta, j} a_{\alpha, i}\right]

    where :math:`i` and :math:`j` run over occupied orbitals, :math:`a` and :math:`b`
    run over virtual orbitals, and :math:`t_{ijab}` are the doubles amplitudes.

    Args:
        t2: The doubles amplitudes tensor of shape ``(nocc, nocc, nvrt, nvrt)``, where
            ``nocc`` is the number of occupied orbitals and ``nvrt`` is the number of
            virtual orbitals.

    Returns:
        The doubles excitations operator.
    """
    nocc, _, nvrt, _ = t2.shape
    op = FermionOperator({})
    for i, j, a, b in itertools.product(
        range(nocc), range(nocc), range(nvrt), range(nvrt)
    ):
        coeff = t2[i, j, a, b]
        op += FermionOperator(
            {
                (cre_a(nocc + a), cre_a(nocc + b), des_a(j), des_a(i)): 0.5 * coeff,
                (cre_b(nocc + a), cre_b(nocc + b), des_b(j), des_b(i)): 0.5 * coeff,
                (cre_a(nocc + a), cre_b(nocc + b), des_b(j), des_a(i)): coeff,
            }
        )
    return op


def doubles_excitations_unrestricted(
    t2: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> FermionOperator:
    r"""Unrestricted doubles excitations operator.

    The unrestricted doubles excitations operator is

    .. math::

        T_2 = \frac12 \sum_{ijab} t^{(\alpha \alpha)}_{ijab}
        a^\dagger_{\alpha, a} a^\dagger_{\alpha, b} a_{\alpha, j} a_{\alpha, i} +
        \frac12 \sum_{ijab} t^{(\beta \beta)}_{IJAB}
        a^\dagger_{\beta, a} a^\dagger_{\beta, b} a_{\beta, j} a_{\beta, i} \right) +
        \sum_{ijab} t^{(\alpha \beta)}_{iJaB}
        + a^\dagger_{\alpha, a} a^\dagger_{\beta, B} a_{\beta, J} a_{\alpha, i}

    where

    - :math:`i` and `j` run over occupied spin-up orbitals,
    - :math:`a` and `b` run over virtual spin-up orbitals,
    - :math:`I` and `J` run over occupied spin-down orbitals,
    - :math:`A` and `B` run over virtual spin-down orbitals,
    - :math:`t^{(\alpha \alpha})_{ijab}` are the doubles amplitudes within spin-up
      orbitals,
    - :math:`t^{(\alpha beta})_{iJaB}` are the doubles amplitudes between spin-up
      and spin-down orbitals, and
    - :math:`t^{(\beta \beta})_{IJAB}` are the doubles amplitudes within spin-down
      orbitals.

    Args:
        t2: The doubles amplitudes. This should be a tuple of three of Numpy arrays,
            ``(t2aa, t2ab, t2bb)``.

    Returns:
        The doubles excitations operator.
    """
    t2aa, t2ab, t2bb = t2
    nocc_a, nocc_b, nvrt_a, nvrt_b = t2ab.shape
    op = FermionOperator({})
    for i, j, a, b in itertools.product(
        range(nocc_a), range(nocc_a), range(nvrt_a), range(nvrt_a)
    ):
        coeff = t2aa[i, j, a, b]
        op += FermionOperator(
            {
                (cre_a(nocc_a + a), cre_a(nocc_a + b), des_a(j), des_a(i)): 0.5 * coeff,
            }
        )
    for i, j, a, b in itertools.product(
        range(nocc_b), range(nocc_b), range(nvrt_b), range(nvrt_b)
    ):
        coeff = t2bb[i, j, a, b]
        op += FermionOperator(
            {
                (cre_b(nocc_a + a), cre_b(nocc_a + b), des_b(j), des_b(i)): 0.5 * coeff,
            }
        )
    for i, j, a, b in itertools.product(
        range(nocc_a), range(nocc_b), range(nvrt_a), range(nvrt_b)
    ):
        coeff = t2ab[i, j, a, b]
        op += FermionOperator(
            {
                (cre_a(nocc_a + a), cre_b(nocc_a + b), des_b(j), des_a(i)): coeff,
            }
        )
    return op


def ccsd_generator_restricted(t1: np.ndarray, t2: np.ndarray) -> FermionOperator:
    r"""Restricted coupled cluster, singles and doubles (CCSD) generator.

    The restricted CCSD generator is

    .. math::

        T = T_1 + T_2

    where :math:`T_1` is the restricted singles excitations operator
    (see :func:`singles_excitations_restricted`) and :math:`T_2` is the restricted
    doubles excitations operator (see :func:`doubles_excitations_restricted`).

    Args:
        t1: The singles amplitudes tensor of shape ``(nocc, nvrt)``, where ``nocc`` is
            the number of occupied orbitals and ``nvrt`` is the number of virtual
            orbitals.
        t2: The doubles amplitudes tensor of shape ``(nocc, nocc, nvrt, nvrt)``, where
            ``nocc`` is the number of occupied orbitals and ``nvrt`` is the number of
            virtual orbitals.

    Returns:
        The CCSD generator.
    """
    return singles_excitations_restricted(t1) + doubles_excitations_restricted(t2)


def uccsd_generator_restricted(t1: np.ndarray, t2: np.ndarray) -> FermionOperator:
    r"""Restricted unitary coupled cluster, singles and doubles (UCCSD) generator.

    The restricted UCCSD generator is

    .. math::

        T - T^\dagger

    where :math:`T` is the restricted CCSD generator
    (see :func:`ccsd_generator_restricted`).

    Args:
        t1: The singles amplitudes tensor of shape ``(nocc, nvrt)``, where ``nocc`` is
            the number of occupied orbitals and ``nvrt`` is the number of virtual
            orbitals.
        t2: The doubles amplitudes tensor of shape ``(nocc, nocc, nvrt, nvrt)``, where
            ``nocc`` is the number of occupied orbitals and ``nvrt`` is the number of
            virtual orbitals.

    Returns:
        The UCCSD generator.
    """
    ccsd_gen = ccsd_generator_restricted(t1=t1, t2=t2)
    return ccsd_gen - ccsd_gen.adjoint()
