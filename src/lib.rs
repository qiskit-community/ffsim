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
use blas::zdrot;
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

/// Apply a phase shift to slices of a state vector.
#[pyfunction]
fn apply_phase_shift_in_place(
    mut vec: PyReadwriteArray2<Complex64>,
    phase: Complex64,
    indices: PyReadonlyArray1<usize>,
) {
    let mut vec = vec.as_array_mut();
    let indices = indices.as_array();
    let shape = vec.shape();
    let dim_b = shape[1] as i32;

    // TODO parallelize this
    indices.for_each(|&str0| {
        let mut target = vec.row_mut(str0);
        match target.as_slice_mut() {
            Some(target) => unsafe {
                zscal(dim_b, phase, target, 1);
            },
            None => panic!(
                "Failed to convert ArrayBase to slice, possibly because the data was not contiguous and in standard order."
            ),
        };
    })
}

/// Apply a matrix to slices of a matrix.
#[pyfunction]
fn apply_givens_rotation_in_place(
    mut vec: PyReadwriteArray2<Complex64>,
    c: f64,
    s: f64,
    phase: Complex64,
    slice1: PyReadonlyArray1<usize>,
    slice2: PyReadonlyArray1<usize>,
) {
    let mut vec = vec.as_array_mut();
    let slice1 = slice1.as_array();
    let slice2 = slice2.as_array();
    let shape = vec.shape();
    let dim_b = shape[1] as i32;
    let phase_conj = phase.conj();

    // TODO parallelize this
    Zip::from(&slice1).and(&slice2).for_each(|&i, &j| {
        let (mut row_i, mut row_j) = vec.multi_slice_mut((s![i, ..], s![j, ..]));
        match row_i.as_slice_mut() {
            Some(row_i) => match row_j.as_slice_mut() {
                Some(row_j) => unsafe {
                    zscal(dim_b, phase_conj, row_i, 1);
                    zdrot(dim_b, row_i, 1, row_j, 1, c, s);
                    zscal(dim_b, phase, row_i, 1);
                },
                None => panic!(
                    "Failed to convert ArrayBase to slice, possibly because the data was not contiguous and in standard order."
                ),
            },
            None => panic!(
                "Failed to convert ArrayBase to slice, possibly because the data was not contiguous and in standard order."
            ),
        };
    });
}

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
                let orb_c = row[0] as usize;
                let orb_d = row[1];
                let str1 = row[2] as usize;
                let sign = row[3];
                let index = off_diag_strings_index[(orb_c, str0)];
                let count = index_counter[(orb_c, index)];
                off_diag_index[(orb_c, index, count, 0)] = orb_d;
                off_diag_index[(orb_c, index, count, 1)] = str1 as i32;
                off_diag_index[(orb_c, index, count, 2)] = sign;
                index_counter[(orb_c, index)] += 1;
            }
        });
}

/// Apply a single-column orbital rotation.
#[pyfunction]
fn apply_single_column_transformation_in_place(
    mut vec: PyReadwriteArray2<Complex64>,
    column: PyReadonlyArray1<Complex64>,
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
    mut vec: PyReadwriteArray2<Complex64>,
    phases: PyReadonlyArray1<Complex64>,
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
fn apply_diag_coulomb_evolution_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    mat_exp: PyReadonlyArray2<Complex64>,
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

    Zip::from(&mut alpha_phases)
        .and(occupations_a.rows())
        .and(phase_map.rows_mut())
        .par_for_each(|val, orbs, mut row| {
            let mut phase = Complex64::new(1.0, 0.0);
            for j in 0..n_alpha {
                let orb_1 = orbs[j];
                row *= &mat_alpha_beta_exp.row(orb_1);
                for k in j..n_alpha {
                    let orb_2 = orbs[k];
                    phase *= mat_exp[(orb_1, orb_2)];
                }
            }
            *val = phase;
        });

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

/// Apply time evolution by a diagonal Coulomb operator in-place, Z representation.
#[pyfunction]
fn apply_diag_coulomb_evolution_in_place_z_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    mat_exp: PyReadonlyArray2<Complex64>,
    mat_exp_conj: PyReadonlyArray2<Complex64>,
    norb: usize,
    mat_alpha_beta_exp: PyReadonlyArray2<Complex64>,
    mat_alpha_beta_exp_conj: PyReadonlyArray2<Complex64>,
    strings_a: PyReadonlyArray1<i64>,
    strings_b: PyReadonlyArray1<i64>,
) {
    let mat_exp = mat_exp.as_array();
    let mat_exp_conj = mat_exp_conj.as_array();
    let mut vec = vec.as_array_mut();
    let mat_alpha_beta_exp = mat_alpha_beta_exp.as_array();
    let mat_alpha_beta_exp_conj = mat_alpha_beta_exp_conj.as_array();
    let strings_a = strings_a.as_array();
    let strings_b = strings_b.as_array();

    let shape = vec.shape();
    let dim_a = shape[0];
    let dim_b = shape[1];

    let mut alpha_phases = Array::zeros(dim_a);
    let mut beta_phases = Array::zeros(dim_b);
    let mut phase_map = Array::ones((dim_a, norb));

    Zip::from(&mut beta_phases)
        .and(strings_b)
        .par_for_each(|val, str0| {
            let mut phase = Complex64::new(1.0, 0.0);
            for j in 0..norb {
                let sign_j = str0 >> j & 1 == 1;
                for k in j + 1..norb {
                    let sign_k = str0 >> k & 1 == 1;
                    let this_phase = if sign_j ^ sign_k {
                        mat_exp_conj[(j, k)]
                    } else {
                        mat_exp[(j, k)]
                    };
                    phase *= this_phase;
                }
            }
            *val = phase;
        });

    Zip::from(&mut alpha_phases)
        .and(strings_a)
        .and(phase_map.rows_mut())
        .par_for_each(|val, str0, mut row| {
            let mut phase = Complex64::new(1.0, 0.0);
            for j in 0..norb {
                let sign_j = str0 >> j & 1 == 1;
                let this_row = if sign_j {
                    mat_alpha_beta_exp_conj.row(j)
                } else {
                    mat_alpha_beta_exp.row(j)
                };
                row *= &this_row;
                for k in j + 1..norb {
                    let sign_k = str0 >> k & 1 == 1;
                    let this_phase = if sign_j ^ sign_k {
                        mat_exp_conj[(j, k)]
                    } else {
                        mat_exp[(j, k)]
                    };
                    phase *= this_phase;
                }
            }
            *val = phase;
        });

    Zip::from(vec.rows_mut())
        .and(&alpha_phases)
        .and(phase_map.rows())
        .par_for_each(|row, alpha_phase, phase_map| {
            Zip::from(row)
                .and(&beta_phases)
                .and(strings_b)
                .for_each(|val, beta_phase, str0| {
                    let mut phase = *alpha_phase * *beta_phase;
                    for j in 0..norb {
                        let this_phase = if str0 >> j & 1 == 1 {
                            phase_map[j].conj()
                        } else {
                            phase_map[j]
                        };
                        phase *= this_phase
                    }
                    *val *= phase;
                })
        });
}

/// Contract a diagonal Coulomb operator into a buffer.
#[pyfunction]
fn contract_diag_coulomb_into_buffer_num_rep(
    vec: PyReadonlyArray2<Complex64>,
    mat: PyReadonlyArray2<f64>,
    norb: usize,
    mat_alpha_beta: PyReadonlyArray2<f64>,
    occupations_a: PyReadonlyArray2<usize>,
    occupations_b: PyReadonlyArray2<usize>,
    mut out: PyReadwriteArray2<Complex64>,
) {
    let vec = vec.as_array();
    let mat = mat.as_array();
    let mat_alpha_beta = mat_alpha_beta.as_array();
    let occupations_a = occupations_a.as_array();
    let occupations_b = occupations_b.as_array();
    let mut out = out.as_array_mut();

    let shape = vec.shape();
    let dim_a = shape[0];
    let dim_b = shape[1];
    let n_alpha = occupations_a.shape()[1];
    let n_beta = occupations_b.shape()[1];

    let mut alpha_coeffs = Array::zeros(dim_a);
    let mut beta_coeffs = Array::zeros(dim_b);
    let mut coeff_map = Array::zeros((dim_a, norb));

    Zip::from(&mut beta_coeffs)
        .and(occupations_b.rows())
        .par_for_each(|val, orbs| {
            let mut coeff = Complex64::new(0.0, 0.0);
            for j in 0..n_beta {
                let orb_1 = orbs[j];
                for k in j..n_beta {
                    let orb_2 = orbs[k];
                    coeff += mat[(orb_1, orb_2)];
                }
            }
            *val = coeff;
        });

    Zip::from(&mut alpha_coeffs)
        .and(occupations_a.rows())
        .and(coeff_map.rows_mut())
        .par_for_each(|val, orbs, mut row| {
            let mut coeff = Complex64::new(0.0, 0.0);
            for j in 0..n_alpha {
                let orb_1 = orbs[j];
                row += &mat_alpha_beta.row(orb_1);
                for k in j..n_alpha {
                    let orb_2 = orbs[k];
                    coeff += mat[(orb_1, orb_2)];
                }
            }
            *val = coeff;
        });

    Zip::from(vec.rows())
        .and(out.rows_mut())
        .and(&alpha_coeffs)
        .and(coeff_map.rows())
        .par_for_each(|source, target, alpha_coeff, coeff_map| {
            Zip::from(source)
                .and(target)
                .and(&beta_coeffs)
                .and(occupations_b.rows())
                .for_each(|source, target, beta_coeff, orbs| {
                    let mut coeff = *alpha_coeff + *beta_coeff;
                    orbs.for_each(|&orb| coeff += coeff_map[orb]);
                    *target += coeff * source;
                })
        });
}

/// Contract a diagonal Coulomb operator into a buffer, Z representation.
#[pyfunction]
fn contract_diag_coulomb_into_buffer_z_rep(
    vec: PyReadonlyArray2<Complex64>,
    mat: PyReadonlyArray2<f64>,
    norb: usize,
    mat_alpha_beta: PyReadonlyArray2<f64>,
    strings_a: PyReadonlyArray1<i64>,
    strings_b: PyReadonlyArray1<i64>,
    mut out: PyReadwriteArray2<Complex64>,
) {
    let vec = vec.as_array();
    let mat = mat.as_array();
    let mat_alpha_beta = mat_alpha_beta.as_array();
    let strings_a = strings_a.as_array();
    let strings_b = strings_b.as_array();
    let mut out = out.as_array_mut();

    let shape = vec.shape();
    let dim_a = shape[0];
    let dim_b = shape[1];

    let mut alpha_coeffs = Array::zeros(dim_a);
    let mut beta_coeffs = Array::zeros(dim_b);
    let mut coeff_map = Array::zeros((dim_a, norb));

    Zip::from(&mut beta_coeffs)
        .and(strings_b)
        .par_for_each(|val, str0| {
            let mut coeff = Complex64::new(0.0, 0.0);
            for j in 0..norb {
                let sign_j = if str0 >> j & 1 == 1 { -1 } else { 1 } as f64;
                for k in j + 1..norb {
                    let sign_k = if str0 >> k & 1 == 1 { -1 } else { 1 } as f64;
                    coeff += sign_j * sign_k * mat[(j, k)];
                }
            }
            *val = coeff;
        });

    Zip::from(&mut alpha_coeffs)
        .and(strings_a)
        .and(coeff_map.rows_mut())
        .par_for_each(|val, str0, mut row| {
            let mut coeff = Complex64::new(0.0, 0.0);
            for j in 0..norb {
                let sign_j = if str0 >> j & 1 == 1 { -1 } else { 1 } as f64;
                row += &(sign_j * &mat_alpha_beta.row(j));
                for k in j + 1..norb {
                    let sign_k = if str0 >> k & 1 == 1 { -1 } else { 1 } as f64;
                    coeff += sign_j * sign_k * mat[(j, k)];
                }
            }
            *val = coeff;
        });

    Zip::from(vec.rows())
        .and(out.rows_mut())
        .and(&alpha_coeffs)
        .and(coeff_map.rows())
        .par_for_each(|source, target, alpha_coeff, coeff_map| {
            Zip::from(source)
                .and(target)
                .and(&beta_coeffs)
                .and(strings_b)
                .for_each(|source, target, beta_coeff, str0| {
                    let mut coeff = *alpha_coeff + *beta_coeff;
                    for j in 0..norb {
                        let sign_j = if str0 >> j & 1 == 1 { -1 } else { 1 } as f64;
                        coeff += sign_j * coeff_map[j];
                    }
                    *target += 0.25 * coeff * source;
                })
        });
}

/// Contract a sum of number operators into a buffer.
#[pyfunction]
fn contract_num_op_sum_spin_into_buffer(
    vec: PyReadonlyArray2<Complex64>,
    coeffs: PyReadonlyArray1<f64>,
    occupations: PyReadonlyArray2<usize>,
    mut out: PyReadwriteArray2<Complex64>,
) {
    let vec = vec.as_array();
    let coeffs = coeffs.as_array();
    let occupations = occupations.as_array();
    let mut out = out.as_array_mut();

    Zip::from(vec.rows())
        .and(out.rows_mut())
        .and(occupations.rows())
        .par_for_each(|source, mut target, orbs| {
            let mut coeff = Complex64::new(0.0, 0.0);
            orbs.for_each(|&orb| coeff += coeffs[orb]);
            target += &(coeff * &source);
        });
}

/// Python module exposing Rust extensions.
#[pymodule]
fn _ffsim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_phase_shift_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(apply_givens_rotation_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(gen_orbital_rotation_index_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(
        apply_single_column_transformation_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(apply_num_op_sum_evolution_in_place, m)?)?;
    m.add_function(wrap_pyfunction!(
        apply_diag_coulomb_evolution_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        apply_diag_coulomb_evolution_in_place_z_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        contract_diag_coulomb_into_buffer_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        contract_diag_coulomb_into_buffer_z_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(contract_num_op_sum_spin_into_buffer, m)?)?;
    Ok(())
}
