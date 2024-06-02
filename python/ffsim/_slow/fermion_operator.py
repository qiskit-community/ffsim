# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections.abc import Iterable, Iterator, MutableMapping
from dataclasses import dataclass

import numpy as np
import pyscf.fci
from scipy.sparse.linalg import LinearOperator

from ffsim.operators.fermion_action import FermionAction
from ffsim.states import dim, dims


@dataclass
class FermionOperator(MutableMapping):
    """A fermionic operator.

    A linear combination of products of fermionic creation and annihilation operators.
    """

    coeffs: dict[tuple[FermionAction, ...], complex]

    def copy(self) -> "FermionOperator":
        return FermionOperator(self.coeffs.copy())

    def __getitem__(self, key: tuple[FermionAction, ...]) -> complex:
        return self.coeffs[key]

    def __setitem__(self, key: tuple[FermionAction, ...], val: complex) -> None:
        self.coeffs[key] = val

    def __delitem__(self, key: tuple[FermionAction, ...]) -> None:
        del self.coeffs[key]

    def __iter__(self) -> Iterator[tuple[FermionAction, ...]]:
        return iter(self.coeffs)

    def __len__(self) -> int:
        return len(self.coeffs)

    def __iadd__(self, other) -> "FermionOperator":
        if isinstance(other, FermionOperator):
            for term, coeff in other.coeffs.items():
                if term in self.coeffs:
                    self.coeffs[term] += coeff
                else:
                    self.coeffs[term] = coeff
            return self

        if isinstance(other, complex):
            if () in self.coeffs:
                self.coeffs[()] += other
            else:
                self.coeffs[()] = other
            return self

        return NotImplemented

    def __add__(self, other) -> "FermionOperator":
        result = self.copy()
        result += other
        return result

    def __isub__(self, other) -> "FermionOperator":
        if isinstance(other, FermionOperator):
            for term, coeff in other.coeffs.items():
                if term in self.coeffs:
                    self.coeffs[term] -= coeff
                else:
                    self.coeffs[term] = -coeff
            return self

        if isinstance(other, complex):
            if () in self.coeffs:
                self.coeffs[()] -= other
            else:
                self.coeffs[()] = -other
            return self

        return NotImplemented

    def __sub__(self, other) -> "FermionOperator":
        result = self.copy()
        result -= other
        return result

    def __neg__(self) -> "FermionOperator":
        result = self.copy()
        result *= -1
        return result

    def __imul__(self, other) -> "FermionOperator":
        if isinstance(other, (int, float, complex)):
            for key in self.coeffs:
                self.coeffs[key] *= other
            return self
        return NotImplemented

    def __rmul__(self, other) -> "FermionOperator":
        if isinstance(other, (int, float, complex)):
            result = self.copy()
            result *= other
            return result
        return NotImplemented

    def __mul__(self, other) -> "FermionOperator":
        if isinstance(other, FermionOperator):
            new_coeffs: dict[tuple[FermionAction, ...], complex] = {}
            for term_1, coeff_1 in self.coeffs.items():
                for term_2, coeff_2 in other.coeffs.items():
                    new_term = term_1 + term_2
                    if new_term in new_coeffs:
                        new_coeffs[new_term] += coeff_1 * coeff_2
                    else:
                        new_coeffs[new_term] = coeff_1 * coeff_2
            return FermionOperator(new_coeffs)

        return NotImplemented

    def __pow__(self, exponent, modulo=None) -> "FermionOperator":
        if isinstance(exponent, int):
            if modulo is not None:
                raise ValueError("mod argument not supported")
            result = FermionOperator({(): 1})
            for _ in range(exponent):
                result = result * self
            return result
        return NotImplemented

    def normal_ordered(self) -> "FermionOperator":
        result = FermionOperator({})
        for term, coeff in self.coeffs.items():
            result += _normal_ordered_term(term, coeff)
        return result

    def conserves_particle_number(self) -> bool:
        for term in self.coeffs:
            create_count = sum(action for action, _, _ in term)
            destroy_count = sum(not action for action, _, _ in term)
            if create_count != destroy_count:
                return False
        return True

    def conserves_spin_z(self) -> bool:
        for term in self.coeffs:
            create_count_a = sum(action for action, spin, _ in term if not spin)
            destroy_count_a = sum(not action for action, spin, _ in term if not spin)
            create_count_b = sum(action for action, spin, _ in term if spin)
            destroy_count_b = sum(not action for action, spin, _ in term if spin)
            if create_count_a - destroy_count_a != create_count_b - destroy_count_b:
                return False
        return True

    def many_body_order(self) -> int:
        return max((len(term) for term in self.coeffs), default=0)

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, FermionOperator):
            for key in self.keys() | other.keys():
                if not np.isclose(
                    self.get(key, 0), other.get(key, 0), rtol=rtol, atol=atol
                ):
                    return False
            return True
        return NotImplemented

    def _linear_operator_(self, norb: int, nelec: tuple[int, int]):
        if not (self.conserves_particle_number() and self.conserves_spin_z()):
            raise ValueError(
                "The given FermionOperator could not be converted to a LinearOperator "
                "because it does not conserve particle number and the Z component "
                "of spin. "
                f"Conserves particle number: {self.conserves_particle_number()} "
                f"Conserves spin Z: {self.conserves_spin_z()}"
            )

        dim_ = dim(norb, nelec)
        dims_ = dims(norb, nelec)

        action_funcs = {
            # key: (action, spin)
            (False, False): pyscf.fci.addons.des_a,
            (False, True): pyscf.fci.addons.des_b,
            (True, False): pyscf.fci.addons.cre_a,
            (True, True): pyscf.fci.addons.cre_b,
        }

        def matvec(vec: np.ndarray):
            result = np.zeros(dim_, dtype=complex)
            vec_real = np.real(vec)
            vec_imag = np.imag(vec)
            for term in self:
                coeff = self[term]
                transformed_real = vec_real.reshape(dims_)
                transformed_imag = vec_imag.reshape(dims_)
                this_nelec = list(nelec)
                for action, spin, orb in reversed(term):
                    action_func = action_funcs[(action, spin)]
                    transformed_real = action_func(
                        transformed_real, norb, this_nelec, orb
                    )
                    transformed_imag = action_func(
                        transformed_imag, norb, this_nelec, orb
                    )
                    this_nelec[spin] += 1 if action else -1
                result += coeff * transformed_real.reshape(-1)
                result += coeff * 1j * transformed_imag.reshape(-1)
            return result

        return LinearOperator(
            shape=(dim_, dim_), matvec=matvec, rmatvec=matvec, dtype=complex
        )


def _normal_ordered_term(
    term: Iterable[FermionAction], coeff: complex
) -> FermionOperator:
    coeffs: dict[tuple[FermionAction, ...], complex] = {}
    stack = [(list(term), coeff)]
    while stack:
        term, coeff = stack.pop()
        parity = False
        for i in range(1, len(term)):
            # shift operator at index i to the left until it's in the correct location
            for j in range(i, 0, -1):
                action_right, spin_right, index_right = term[j]
                action_left, spin_left, index_left = term[j - 1]
                if action_right == action_left:
                    # both create or both destroy
                    if (spin_right, index_right) == (spin_left, index_left):
                        # operators are the same, so product is zero
                        return FermionOperator({})
                    elif (spin_right, index_right) > (spin_left, index_left):
                        # swap operators and update sign
                        term[j - 1], term[j] = term[j], term[j - 1]
                        parity = not parity
                elif action_right and not action_left:
                    # create on right and destroy on left
                    if (spin_right, index_right) == (spin_left, index_left):
                        # add new term
                        new_term = term[: j - 1] + term[j + 1 :]
                        sign = -1 if parity else 1
                        stack.append((new_term, sign * coeff))
                    # swap operators and update sign
                    term[j - 1], term[j] = term[j], term[j - 1]
                    parity = not parity
        term = tuple(term)
        sign = -1 if parity else 1
        if term in coeffs:
            coeffs[term] += sign * coeff
        else:
            coeffs[term] = sign * coeff
    return FermionOperator(coeffs)
