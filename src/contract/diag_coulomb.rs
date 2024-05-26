// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::Array;
use ndarray::Zip;
use numpy::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;

/// Contract a diagonal Coulomb operator into a buffer.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn contract_diag_coulomb_into_buffer_num_rep(
    vec: PyReadonlyArray2<Complex64>,
    mat_aa: PyReadonlyArray2<f64>,
    mat_ab: PyReadonlyArray2<f64>,
    mat_bb: PyReadonlyArray2<f64>,
    norb: usize,
    occupations_a: PyReadonlyArray2<usize>,
    occupations_b: PyReadonlyArray2<usize>,
    mut out: PyReadwriteArray2<Complex64>,
) {
    let vec = vec.as_array();
    let mat_aa = mat_aa.as_array();
    let mat_ab = mat_ab.as_array();
    let mat_bb = mat_bb.as_array();
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
                    coeff += mat_bb[(orb_1, orb_2)];
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
                row += &mat_ab.row(orb_1);
                for k in j..n_alpha {
                    let orb_2 = orbs[k];
                    coeff += mat_aa[(orb_1, orb_2)];
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
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn contract_diag_coulomb_into_buffer_z_rep(
    vec: PyReadonlyArray2<Complex64>,
    mat_aa: PyReadonlyArray2<f64>,
    mat_ab: PyReadonlyArray2<f64>,
    mat_bb: PyReadonlyArray2<f64>,
    norb: usize,
    strings_a: PyReadonlyArray1<i64>,
    strings_b: PyReadonlyArray1<i64>,
    mut out: PyReadwriteArray2<Complex64>,
) {
    let vec = vec.as_array();
    let mat_aa = mat_aa.as_array();
    let mat_ab = mat_ab.as_array();
    let mat_bb = mat_bb.as_array();
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
                    coeff += sign_j * sign_k * mat_bb[(j, k)];
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
                row += &(sign_j * &mat_ab.row(j));
                for k in j + 1..norb {
                    let sign_k = if str0 >> k & 1 == 1 { -1 } else { 1 } as f64;
                    coeff += sign_j * sign_k * mat_aa[(j, k)];
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
