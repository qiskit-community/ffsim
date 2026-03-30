// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use blas::zrotg as zrotg_blas;
use ndarray::Array2;
use numpy::{Complex64, IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type GivensRotationTuple = (f64, Complex64, usize, usize);
type GivensDecompositionResult = (Vec<GivensRotationTuple>, Py<PyArray1<Complex64>>);

fn zrotg_safe(a: Complex64, b: Complex64, tol: f64) -> (f64, Complex64) {
    if a.norm() <= tol {
        return (0.0, Complex64::new(1.0, 0.0));
    }
    if b.norm() <= tol {
        return (1.0, Complex64::new(0.0, 0.0));
    }
    let mut a_mut = a;
    let mut c = 0.0;
    let mut s = Complex64::new(0.0, 0.0);
    unsafe {
        zrotg_blas(&mut a_mut, b, &mut c, &mut s);
    }
    (c, s)
}

fn rotate_columns_in_place(mat: &mut Array2<Complex64>, col_x: usize, col_y: usize, c: f64, s: Complex64) {
    let n_rows = mat.nrows();
    for row in 0..n_rows {
        let x_old = mat[[row, col_x]];
        let y_old = mat[[row, col_y]];
        mat[[row, col_x]] = c * x_old + s * y_old;
        mat[[row, col_y]] = c * y_old - s.conj() * x_old;
    }
}

fn rotate_rows_in_place(mat: &mut Array2<Complex64>, row_x: usize, row_y: usize, c: f64, s: Complex64) {
    let n_cols = mat.ncols();
    for col in 0..n_cols {
        let x_old = mat[[row_x, col]];
        let y_old = mat[[row_y, col]];
        mat[[row_x, col]] = c * x_old + s * y_old;
        mat[[row_y, col]] = c * y_old - s.conj() * x_old;
    }
}

/// Givens rotation decomposition of a unitary matrix.
#[pyfunction]
pub fn givens_decomposition(
    py: Python<'_>,
    mat: PyReadonlyArray2<Complex64>,
) -> PyResult<GivensDecompositionResult> {
    let mat_view = mat.as_array();
    let shape = mat_view.shape();
    if shape[0] != shape[1] {
        return Err(PyValueError::new_err("mat must be a square matrix"));
    }
    let n = shape[0];
    let mut current_matrix: Array2<Complex64> = mat_view.to_owned();
    let mut left_rotations: Vec<GivensRotationTuple> = Vec::new();
    let mut right_rotations: Vec<GivensRotationTuple> = Vec::new();
    let tol = 1e-12;

    for i in 0..n.saturating_sub(1) {
        if i % 2 == 0 {
            for j in 0..=i {
                let target_index = i - j;
                let row = n - j - 1;
                if current_matrix[[row, target_index]] != Complex64::new(0.0, 0.0) {
                    let (c, s) = zrotg_safe(
                        current_matrix[[row, target_index + 1]],
                        current_matrix[[row, target_index]],
                        tol,
                    );
                    right_rotations.push((c, s, target_index + 1, target_index));
                    rotate_columns_in_place(&mut current_matrix, target_index + 1, target_index, c, s);
                }
            }
        } else {
            for j in 0..=i {
                let target_index = n - i + j - 1;
                let col = j;
                if current_matrix[[target_index, col]] != Complex64::new(0.0, 0.0) {
                    let (c, s) = zrotg_safe(
                        current_matrix[[target_index - 1, col]],
                        current_matrix[[target_index, col]],
                        tol,
                    );
                    left_rotations.push((c, s, target_index - 1, target_index));
                    rotate_rows_in_place(&mut current_matrix, target_index - 1, target_index, c, s);
                }
            }
        }
    }

    for &(c_left, s_left, i, j) in left_rotations.iter().rev() {
        let (c, s) = zrotg_safe(
            Complex64::new(c_left, 0.0) * current_matrix[[j, j]],
            s_left.conj() * current_matrix[[i, i]],
            tol,
        );

        right_rotations.push((c.clamp(-1.0, 1.0), -s.conj(), i, j));

        let diag_i = current_matrix[[i, i]];
        let diag_j = current_matrix[[j, j]];
        let g00 = Complex64::new(c, 0.0) * diag_i;
        let g01 = -s * diag_j;
        let g10 = s.conj() * diag_i;
        let g11 = Complex64::new(c, 0.0) * diag_j;
        let (c_new, s_new) = zrotg_safe(g11, g10, tol);

        let phase00 = g00 * Complex64::new(c_new, 0.0) + g01 * (-s_new.conj());
        let phase11 = g10 * s_new + g11 * Complex64::new(c_new, 0.0);
        current_matrix[[i, i]] = phase00;
        current_matrix[[j, j]] = phase11;
    }

    let diagonal = current_matrix.diag().to_owned();
    Ok((right_rotations, diagonal.into_pyarray(py).unbind()))
}
