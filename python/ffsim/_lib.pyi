from collections.abc import Iterator, Mapping

import numpy as np

class FermionOperator(Mapping[tuple[tuple[bool, bool, int], ...], complex]):
    def __init__(
        self, coeffs: dict[tuple[tuple[bool, bool, int], ...], complex]
    ) -> None: ...
    def normal_ordered(self) -> "FermionOperator": ...
    def conserves_particle_number(self) -> bool: ...
    def conserves_spin_z(self) -> bool: ...
    def many_body_order(self) -> int: ...
    def __getitem__(self, key: tuple[tuple[bool, bool, int], ...]) -> complex: ...
    def __contains__(self, key: object) -> bool: ...
    def __iter__(self) -> Iterator[tuple[tuple[bool, bool, int], ...]]: ...
    def __len__(self) -> int: ...
    def __iadd__(self, other: FermionOperator) -> FermionOperator: ...
    def __add__(self, other: FermionOperator) -> FermionOperator: ...
    def __isub__(self, other: FermionOperator) -> FermionOperator: ...
    def __sub__(self, other: FermionOperator) -> FermionOperator: ...
    def __neg__(self) -> FermionOperator: ...
    def __itruediv__(self, other: int | float | complex) -> FermionOperator: ...
    def __truediv__(self, other: int | float | complex) -> FermionOperator: ...
    def __rmul__(self, other: int | float | complex) -> FermionOperator: ...
    def __imul__(
        self, other: int | float | complex | FermionOperator
    ) -> FermionOperator: ...
    def __mul__(self, other: FermionOperator) -> FermionOperator: ...
    def __pow__(self, other: int) -> FermionOperator: ...

def apply_phase_shift_in_place(
    vec: np.ndarray, phase: complex, indices: np.ndarray
) -> None: ...
def apply_givens_rotation_in_place(
    vec: np.ndarray, c: float, s: complex, slice1: np.ndarray, slice2: np.ndarray
) -> None: ...
def gen_orbital_rotation_index_in_place(
    norb: int,
    nocc: int,
    linkstr_index: np.ndarray,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_strings_index: np.ndarray,
    off_diag_index: np.ndarray,
) -> None: ...
def apply_single_column_transformation_in_place(
    vec: np.ndarray,
    column: np.ndarray,
    diag_val: complex,
    diag_strings: np.ndarray,
    off_diag_strings: np.ndarray,
    off_diag_index: np.ndarray,
) -> None: ...
def apply_num_op_sum_evolution_in_place(
    vec: np.ndarray,
    phases: np.ndarray,
    occupations: np.ndarray,
) -> None: ...
def apply_diag_coulomb_evolution_in_place_num_rep(
    vec: np.ndarray,
    mat_exp: np.ndarray,
    norb: int,
    mat_alpha_beta_exp: np.ndarray,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
) -> None: ...
def apply_diag_coulomb_evolution_in_place_z_rep(
    vec: np.ndarray,
    mat_exp: np.ndarray,
    mat_exp_conj: np.ndarray,
    norb: int,
    mat_alpha_beta_exp: np.ndarray,
    mat_alpha_beta_exp_conj: np.ndarray,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
) -> None: ...
def contract_diag_coulomb_into_buffer_num_rep(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    mat_alpha_beta: np.ndarray,
    occupations_a: np.ndarray,
    occupations_b: np.ndarray,
    out: np.ndarray,
) -> None: ...
def contract_diag_coulomb_into_buffer_z_rep(
    vec: np.ndarray,
    mat: np.ndarray,
    norb: int,
    mat_alpha_beta: np.ndarray,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
    out: np.ndarray,
) -> None: ...
def contract_num_op_sum_spin_into_buffer(
    vec: np.ndarray,
    coeffs: np.ndarray,
    occupations: np.ndarray,
    out: np.ndarray,
) -> None: ...
