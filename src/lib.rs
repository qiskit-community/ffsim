// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

extern crate blas_src;

use blas::zaxpy;
use blas::zscal;
use ndarray::s;
use ndarray::Array;
use ndarray::Axis;
use ndarray::Zip;
use numpy::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadonlyArray3;
use numpy::PyReadwriteArray2;
use numpy::PyReadwriteArray4;
use pyo3::prelude::*;

/// Generate orbital rotation index.
#[pyfunction]
fn gen_orbital_rotation_index_in_place(
    norb: usize,
    nocc: usize,
    linkstr_index: PyReadonlyArray3<i32>,
    mut diag_strings: PyReadwriteArray2<usize>,
    mut off_diag_strings: PyReadwriteArray2<usize>,
    mut off_diag_strings_index: PyReadwriteArray2<usize>,
    mut off_diag_index: PyReadwriteArray4<i32>,
) {
    let linkstr_index = linkstr_index.as_array();
    let mut diag_strings = diag_strings.as_array_mut();
    let mut off_diag_strings = off_diag_strings.as_array_mut();
    let mut off_diag_strings_index = off_diag_strings_index.as_array_mut();
    let mut off_diag_index = off_diag_index.as_array_mut();

    let mut diag_counter = Array::zeros(norb);
    let mut off_diag_counter = Array::zeros(norb);

    // TODO parallelize this
    linkstr_index
        .slice(s![.., .., 0])
        .rows()
        .into_iter()
        .enumerate()
        .for_each(|(str0, tab)| {
            for &orb in tab.slice(s![..nocc]).iter() {
                let orb = orb as usize;
                let count = diag_counter[orb];
                diag_strings[(orb, count)] = str0;
                diag_counter[orb] += 1;
            }
            for &orb in tab.slice(s![nocc..norb]).iter() {
                let orb = orb as usize;
                let count = off_diag_counter[orb];
                off_diag_strings[(orb, count)] = str0;
                off_diag_strings_index[(orb, str0)] = count;
                off_diag_counter[orb] += 1;
            }
        });

    // TODO parallelize this
    let mut index_counter = Array::zeros(off_diag_strings.raw_dim());
    linkstr_index
        .axis_iter(Axis(0))
        .enumerate()
        .for_each(|(str0, tab)| {
            for row in tab.slice(s![nocc.., ..]).rows() {
                let orb_c = row[0];
                let orb_d = row[1] as usize;
                let str1 = row[2] as usize;
                let sign = row[3];
                let index = off_diag_strings_index[(orb_d, str1)];
                let count = index_counter[(orb_d, index)];
                off_diag_index[(orb_d, index, count, 0)] = orb_c;
                off_diag_index[(orb_d, index, count, 1)] = str0 as i32;
                off_diag_index[(orb_d, index, count, 2)] = sign;
                index_counter[(orb_d, index)] += 1;
            }
        });
}

/// Apply a single-column orbital rotation.
#[pyfunction]
fn apply_single_column_transformation_in_place(
    column: PyReadonlyArray1<Complex64>,
    mut vec: PyReadwriteArray2<Complex64>,
    diag_val: Complex64,
    diag_strings: PyReadonlyArray1<usize>,
    off_diag_strings: PyReadonlyArray1<usize>,
    off_diag_index: PyReadonlyArray3<i32>,
) {
    let column = column.as_array();
    let mut vec = vec.as_array_mut();
    let diag_strings = diag_strings.as_array();
    let off_diag_strings = off_diag_strings.as_array();
    let off_diag_index = off_diag_index.as_array();

    let shape = vec.shape();
    let dim_b = shape[1] as i32;

    // TODO parallelize this
    Zip::from(&off_diag_strings)
        .and(off_diag_index.axis_iter(Axis(0)))
        .for_each(|&str0, tab| {
            for row in tab.rows() {
                let orb = row[0] as usize;
                let str1 = row[1] as usize;
                let sign = Complex64::new(row[2] as f64, 0.0);
                let (mut target, source) = vec.multi_slice_mut((s![str0, ..], s![str1, ..]));
                match target.as_slice_mut() {
                    Some(target) => match source.as_slice() {
                        Some(source) => unsafe {
                            zaxpy(dim_b, sign * column[orb], source, 1, target, 1);
                        },
                        None => panic!(
                            "Failed to convert ArrayBase to slice, possibly because the data was not contiguous and in standard order."
                        ),
                    },
                    None => panic!(
                        "Failed to convert ArrayBase to slice, possibly because the data was not contiguous and in standard order."
                    ),
                };
            }
        });

    // TODO parallelize this
    diag_strings.for_each(|&str0| {
        let mut target = vec.row_mut(str0);
        match target.as_slice_mut() {
            Some(target) => unsafe {
                zscal(dim_b, diag_val, target, 1);
            },
            None => panic!(
                "Failed to convert ArrayBase to slice, possibly because the data was not contiguous and in standard order."
            ),
        };
    })
}

/// Apply time evolution by a sum of number operators in-place.
#[pyfunction]
fn apply_num_op_sum_evolution_in_place(
    phases: PyReadonlyArray1<Complex64>,
    mut vec: PyReadwriteArray2<Complex64>,
    occupations: PyReadonlyArray2<usize>,
) {
    let phases = phases.as_array();
    let mut vec = vec.as_array_mut();
    let occupations = occupations.as_array();

    Zip::from(vec.rows_mut())
        .and(occupations.rows())
        .par_for_each(|mut row, orbs| {
            let mut phase = Complex64::new(1.0, 0.0);
            orbs.for_each(|&orb| phase *= phases[orb]);
            row *= phase;
        });
}

/// Apply time evolution by a diagonal Coulomb operator in-place.
#[pyfunction]
fn apply_diag_coulomb_evolution_in_place(
    mat_exp: PyReadonlyArray2<Complex64>,
    mut vec: PyReadwriteArray2<Complex64>,
    norb: usize,
    mat_alpha_beta_exp: PyReadonlyArray2<Complex64>,
    occupations_a: PyReadonlyArray2<usize>,
    occupations_b: PyReadonlyArray2<usize>,
) {
    let mat_exp = mat_exp.as_array();
    let mut vec = vec.as_array_mut();
    let mat_alpha_beta_exp = mat_alpha_beta_exp.as_array();
    let occupations_a = occupations_a.as_array();
    let occupations_b = occupations_b.as_array();

    let shape = vec.shape();
    let dim_a = shape[0];
    let dim_b = shape[1];
    let n_alpha = occupations_a.shape()[1];
    let n_beta = occupations_b.shape()[1];

    let mut alpha_phases = Array::zeros(dim_a);
    let mut beta_phases = Array::zeros(dim_b);
    let mut phase_map = Array::ones((dim_a, norb));

    Zip::from(&mut alpha_phases)
        .and(occupations_a.rows())
        .par_for_each(|val, orbs| {
            let mut phase = Complex64::new(1.0, 0.0);
            for j in 0..n_alpha {
                let orb_1 = orbs[j];
                for k in j..n_alpha {
                    let orb_2 = orbs[k];
                    phase *= mat_exp[(orb_1, orb_2)];
                }
            }
            *val = phase;
        });

    Zip::from(&mut beta_phases)
        .and(occupations_b.rows())
        .par_for_each(|val, orbs| {
            let mut phase = Complex64::new(1.0, 0.0);
            for j in 0..n_beta {
                let orb_1 = orbs[j];
                for k in j..n_beta {
                    let orb_2 = orbs[k];
                    phase *= mat_exp[(orb_1, orb_2)];
                }
            }
            *val = phase;
        });

    Zip::from(phase_map.rows_mut())
        .and(occupations_a.rows())
        .par_for_each(|mut row, orbs| orbs.for_each(|&orb| row *= &mat_alpha_beta_exp.row(orb)));

    Zip::from(vec.rows_mut())
        .and(&alpha_phases)
        .and(phase_map.rows())
        .par_for_each(|row, alpha_phase, phase_map| {
            Zip::from(row)
                .and(&beta_phases)
                .and(occupations_b.rows())
                .for_each(|val, beta_phase, orbs| {
                    let mut phase = *alpha_phase * *beta_phase;
                    orbs.for_each(|&orb| phase *= phase_map[orb]);
                    *val *= phase;
                })
        });
}

/// Python module exposing Rust extensions.
#[pymodule]
fn _ffsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gen_orbital_rotation_index_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(
        apply_single_column_transformation_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(apply_num_op_sum_evolution_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(apply_diag_coulomb_evolution_in_place, m)?)?;
    Ok(())
}
