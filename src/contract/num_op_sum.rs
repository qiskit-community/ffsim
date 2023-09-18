// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::Zip;
use numpy::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;

/// Contract a sum of number operators into a buffer.
#[pyfunction]
pub fn contract_num_op_sum_spin_into_buffer(
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
