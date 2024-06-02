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

/// Apply time evolution by a diagonal Coulomb operator in-place.
#[pyfunction]
pub fn apply_diag_coulomb_evolution_in_place_num_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    mat_exp_aa: PyReadonlyArray2<Complex64>,
    mat_exp_ab: PyReadonlyArray2<Complex64>,
    mat_exp_bb: PyReadonlyArray2<Complex64>,
    norb: usize,
    occupations_a: PyReadonlyArray2<usize>,
    occupations_b: PyReadonlyArray2<usize>,
) {
    let mat_exp_aa = mat_exp_aa.as_array();
    let mat_exp_ab = mat_exp_ab.as_array();
    let mat_exp_bb = mat_exp_bb.as_array();
    let mut vec = vec.as_array_mut();
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
                    phase *= mat_exp_bb[(orb_1, orb_2)];
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
                row *= &mat_exp_ab.row(orb_1);
                for k in j..n_alpha {
                    let orb_2 = orbs[k];
                    phase *= mat_exp_aa[(orb_1, orb_2)];
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
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn apply_diag_coulomb_evolution_in_place_z_rep(
    mut vec: PyReadwriteArray2<Complex64>,
    mat_exp_aa: PyReadonlyArray2<Complex64>,
    mat_exp_ab: PyReadonlyArray2<Complex64>,
    mat_exp_bb: PyReadonlyArray2<Complex64>,
    mat_exp_aa_conj: PyReadonlyArray2<Complex64>,
    mat_exp_ab_conj: PyReadonlyArray2<Complex64>,
    mat_exp_bb_conj: PyReadonlyArray2<Complex64>,
    norb: usize,
    strings_a: PyReadonlyArray1<i64>,
    strings_b: PyReadonlyArray1<i64>,
) {
    let mat_exp_aa = mat_exp_aa.as_array();
    let mat_exp_ab = mat_exp_ab.as_array();
    let mat_exp_bb = mat_exp_bb.as_array();
    let mat_exp_aa_conj = mat_exp_aa_conj.as_array();
    let mat_exp_ab_conj = mat_exp_ab_conj.as_array();
    let mat_exp_bb_conj = mat_exp_bb_conj.as_array();
    let mut vec = vec.as_array_mut();
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
                        mat_exp_bb_conj[(j, k)]
                    } else {
                        mat_exp_bb[(j, k)]
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
                    mat_exp_ab_conj.row(j)
                } else {
                    mat_exp_ab.row(j)
                };
                row *= &this_row;
                for k in j + 1..norb {
                    let sign_k = str0 >> k & 1 == 1;
                    let this_phase = if sign_j ^ sign_k {
                        mat_exp_aa_conj[(j, k)]
                    } else {
                        mat_exp_aa[(j, k)]
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
