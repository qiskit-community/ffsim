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
use numpy::Complex64;
use numpy::PyReadonlyArray1;

use numpy::PyReadwriteArray2;

use pyo3::prelude::*;

/// Apply a Givens rotation between two slices of a vector
#[pyfunction]
pub fn apply_givens_rotation_in_place(
    mut vec: PyReadwriteArray2<Complex64>,
    c: f64,
    s: Complex64,
    slice1: PyReadonlyArray1<usize>,
    slice2: PyReadonlyArray1<usize>,
) {
    if slice1.is_empty().unwrap() {
        return;
    }

    let mut vec = vec.as_array_mut();
    let slice1 = slice1.as_array();
    let slice2 = slice2.as_array();
    let dim_b = vec.shape()[1];
    let dim_b_i32 = dim_b as i32;
    let s_abs = s.norm();
    let angle = s.arg();
    let phase = Complex64::new(angle.cos(), angle.sin());
    let phase_conj = phase.conj();

    let n_pairs = slice1.len();
    let n_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .min(n_pairs);
    let chunk_size = n_pairs.div_ceil(n_threads);
    let vec_ptr = vec.as_mut_ptr() as usize;
    let slice1 = slice1.as_slice().unwrap();
    let slice2 = slice2.as_slice().unwrap();

    std::thread::scope(|scope| {
        for k in 0..n_threads {
            let start = k * chunk_size;
            let end = (start + chunk_size).min(n_pairs);
            let slice1_chunk = &slice1[start..end];
            let slice2_chunk = &slice2[start..end];
            scope.spawn(move || {
                let vec_ptr = vec_ptr as *mut Complex64;
                for (&i, &j) in slice1_chunk.iter().zip(slice2_chunk) {
                    unsafe {
                        let row_i = std::slice::from_raw_parts_mut(vec_ptr.add(i * dim_b), dim_b);
                        let row_j = std::slice::from_raw_parts_mut(vec_ptr.add(j * dim_b), dim_b);
                        // Use zdrot and zscal because zrot is not currently available
                        // See https://github.com/qiskit-community/ffsim/issues/28
                        zscal(dim_b_i32, phase_conj, row_i, 1);
                        zdrot(dim_b_i32, row_i, 1, row_j, 1, c, s_abs);
                        zscal(dim_b_i32, phase, row_i, 1);
                    }
                }
            });
        }
    });
}
