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

use blas::zdrot;
use blas::zscal;
use ndarray::s;

use ndarray::Zip;
use numpy::Complex64;
use numpy::PyReadonlyArray1;

use numpy::PyReadwriteArray2;

use pyo3::prelude::*;

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
    let angle = s.arg();
    let phase = Complex64::new(angle.cos(), angle.sin());
    let phase_conj = phase.conj();

    Zip::from(&slice1).and(&slice2).for_each(|&i, &j| {
        let (mut row_i, mut row_j) = vec.multi_slice_mut((s![i, ..], s![j, ..]));
        match row_i.as_slice_mut() {
            Some(row_i) => match row_j.as_slice_mut() {
                Some(row_j) => unsafe {
                    zscal(dim_b, phase_conj, row_i, 1);
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
