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

/// Apply time evolution by a sum of number operators in-place.
#[pyfunction]
pub fn apply_num_op_sum_evolution_in_place(
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
