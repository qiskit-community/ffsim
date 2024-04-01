# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The FermionOperator class."""

from collections.abc import ItemsView, KeysView, ValuesView

from ffsim._lib import FermionOperator


def keys(self):
    return KeysView(self)


def values(self):
    return ValuesView(self)


def items(self):
    return ItemsView(self)


FermionOperator.keys = keys  # type: ignore[method-assign]
FermionOperator.values = values  # type: ignore[method-assign]
FermionOperator.items = items  # type: ignore[method-assign]
