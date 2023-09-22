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
use numpy::PyReadonlyArray3;
use numpy::PyReadwriteArray2;
use numpy::PyReadwriteArray4;
use pyo3::prelude::*;

/// Generate orbital rotation index.
#[pyfunction]
pub fn gen_orbital_rotation_index_in_place(
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
pub fn apply_single_column_transformation_in_place(
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

/// Apply a matrix to slices of a matrix.
#[pyfunction]
pub fn apply_givens_rotation_in_place(
    mut vec: PyReadwriteArray2<Complex64>,
    c: f64,
    s: Complex64,
    slice1: PyReadonlyArray1<usize>,
    slice2: PyReadonlyArray1<usize>,
) {
    let mut vec = vec.as_array_mut();
    let slice1 = slice1.as_array();
    let slice2 = slice2.as_array();
    let shape = vec.shape();
    let dim_b = shape[1] as i32;
    let s_abs = s.norm();
    let phase = s / s_abs;
    let phase_conj = phase.conj();

    // TODO parallelize this
    Zip::from(&slice1).and(&slice2).for_each(|&i, &j| {
        let (mut row_i, mut row_j) = vec.multi_slice_mut((s![i, ..], s![j, ..]));
        match row_i.as_slice_mut() {
            Some(row_i) => match row_j.as_slice_mut() {
                Some(row_j) => unsafe {
                    zscal(dim_b, phase_conj, row_i, 1);
                    // TODO use zrot from lapack once it's available
                    // See https://github.com/blas-lapack-rs/lapack/issues/30
                    zdrot(dim_b, row_i, 1, row_j, 1, c, s_abs);
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
