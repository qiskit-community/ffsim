# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The FermionAction NamedTuple and construction functions."""

from typing import NamedTuple


class FermionAction(NamedTuple):
    """A fermionic action."""

    action: bool  # False = destroy (annihilate), True = create
    spin: bool  # False = alpha, True = beta
    orb: int  # index of the orbital to act on


def cre(spin: bool, orb: int) -> FermionAction:
    """Create a fermion.

    Args:
        spin: The spin of the orbital. False = alpha, True = beta.
        orb: The index of the orbital to act on.
    """
    return FermionAction(action=True, spin=spin, orb=orb)


def des(spin: bool, orb: int) -> FermionAction:
    """Destroy a fermion.

    Args:
        spin: The spin of the orbital. False = alpha, True = beta.
        orb: The index of the orbital to act on.
    """
    return FermionAction(action=False, spin=spin, orb=orb)


def cre_a(orb: int) -> FermionAction:
    """Create a fermion with spin alpha.

    Args:
        orb: The index of the orbital to act on.
    """
    return cre(False, orb)


def des_a(orb: int) -> FermionAction:
    """Destroy a fermion with spin alpha.

    Args:
        orb: The index of the orbital to act on.
    """
    return des(False, orb)


def cre_b(orb: int) -> FermionAction:
    """Create a fermion with spin beta.

    Args:
        orb: The index of the orbital to act on.
    """
    return cre(True, orb)


def des_b(orb: int) -> FermionAction:
    """Destroy a fermion with spin beta.

    Args:
        orb: The index of the orbital to act on.
    """
    return des(True, orb)
