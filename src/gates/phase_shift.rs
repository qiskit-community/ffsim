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

use blas::zscal;
use numpy::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;

/// Apply a phase shift to slices of a state vector.
#[pyfunction]
pub fn apply_phase_shift_in_place(
    mut vec: PyReadwriteArray2<Complex64>,
    phase: Complex64,
    indices: PyReadonlyArray1<usize>,
) {
    let mut vec = vec.as_array_mut();
    let indices = indices.as_array();
    let shape = vec.shape();
    let dim_b = shape[1] as i32;

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
