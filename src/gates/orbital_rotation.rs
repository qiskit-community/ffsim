// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

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

    let slice1_slice = slice1.as_slice().unwrap();
    let slice2_slice = slice2.as_slice().unwrap();

    // Sequential execution for single thread or small slice
    if n_threads == 1 || n_pairs < 128 {
        let vec_ptr = vec.as_mut_ptr();
        for (&i, &j) in slice1_slice.iter().zip(slice2_slice) {
            unsafe {
                _apply_givens_rotation_to_pair(vec_ptr, i, j, dim_b, c, s);
            }
        }
        return;
    }

    // Parallel execution
    let chunk_size = n_pairs.div_ceil(n_threads);
    let vec_ptr = vec.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        for k in 0..n_threads {
            let start = k * chunk_size;
            if start >= n_pairs {
                break;
            }
            let end = (start + chunk_size).min(n_pairs);
            let slice1_chunk = &slice1_slice[start..end];
            let slice2_chunk = &slice2_slice[start..end];
            scope.spawn(move || {
                let vec_ptr = vec_ptr as *mut Complex64;
                for (&i, &j) in slice1_chunk.iter().zip(slice2_chunk) {
                    unsafe {
                        _apply_givens_rotation_to_pair(vec_ptr, i, j, dim_b, c, s);
                    }
                }
            });
        }
    });
}

/// Apply Givens rotation to a pair of rows
unsafe fn _apply_givens_rotation_to_pair(
    vec_ptr: *mut Complex64,
    i: usize,
    j: usize,
    dim_b: usize,
    c: f64,
    s: Complex64,
) {
    let row_i = std::slice::from_raw_parts_mut(vec_ptr.add(i * dim_b), dim_b);
    let row_j = std::slice::from_raw_parts_mut(vec_ptr.add(j * dim_b), dim_b);
    for k in 0..dim_b {
        // This is BLAS zrot but it's not available
        // See https://github.com/qiskit-community/ffsim/issues/28
        let i_old = row_i[k];
        let j_old = row_j[k];
        row_i[k] = c * i_old + s * j_old;
        row_j[k] = c * j_old - s.conj() * i_old;
    }
}
