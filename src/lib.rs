// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;

mod contract;
mod fermion_operator;
mod gates;

/// Python module exposing Rust extensions.
#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        gates::phase_shift::apply_phase_shift_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        gates::orbital_rotation::apply_givens_rotation_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        gates::num_op_sum::apply_num_op_sum_evolution_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        gates::diag_coulomb::apply_diag_coulomb_evolution_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        gates::diag_coulomb::apply_diag_coulomb_evolution_in_place_z_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        contract::diag_coulomb::contract_diag_coulomb_into_buffer_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        contract::diag_coulomb::contract_diag_coulomb_into_buffer_z_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        contract::num_op_sum::contract_num_op_sum_spin_into_buffer,
        m
    )?)?;
    m.add_class::<fermion_operator::FermionOperator>()?;
    Ok(())
}
