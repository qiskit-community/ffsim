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
