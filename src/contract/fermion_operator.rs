// (C) Copyright IBM 2026
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
use numpy::PyReadonlyArray2;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;

#[inline]
fn src_dst(src: &[usize], dst: &[usize], index: usize, reverse: bool) -> (usize, usize) {
    if reverse {
        (dst[index], src[index])
    } else {
        (src[index], dst[index])
    }
}

struct ScaledTransitions<'a> {
    src: &'a [usize],
    dst: &'a [usize],
    scale: &'a [Complex64],
}

struct PhasedTransitions<'a> {
    indptr: &'a [usize],
    src: &'a [usize],
    dst: &'a [usize],
    phase: &'a [i8],
}

#[inline]
fn directed_scale(scale: Complex64, reverse: bool) -> Complex64 {
    if reverse {
        scale.conj()
    } else {
        scale
    }
}

fn contract_scalar_terms(
    vec: &[Complex64],
    scalar_coeffs: &[Complex64],
    out: &mut [Complex64],
    reverse: bool,
) {
    for &scale in scalar_coeffs {
        let scale = directed_scale(scale, reverse);
        for index in 0..out.len() {
            out[index] += scale * vec[index];
        }
    }
}

fn contract_alpha_only_terms(
    vec: &[Complex64],
    out: &mut [Complex64],
    dim_b: usize,
    transitions: &ScaledTransitions,
    reverse: bool,
) {
    for index in 0..transitions.scale.len() {
        let (source_alpha, target_alpha) =
            src_dst(transitions.src, transitions.dst, index, reverse);
        let scale = directed_scale(transitions.scale[index], reverse);
        let source_offset = source_alpha * dim_b;
        let target_offset = target_alpha * dim_b;
        for beta_index in 0..dim_b {
            out[target_offset + beta_index] += scale * vec[source_offset + beta_index];
        }
    }
}

fn contract_beta_only_terms(
    vec: &[Complex64],
    out: &mut [Complex64],
    dim_a: usize,
    dim_b: usize,
    transitions: &ScaledTransitions,
    reverse: bool,
) {
    for alpha_index in 0..dim_a {
        let row_offset = alpha_index * dim_b;
        for index in 0..transitions.scale.len() {
            let (source_beta, target_beta) =
                src_dst(transitions.src, transitions.dst, index, reverse);
            let scale = directed_scale(transitions.scale[index], reverse);
            out[row_offset + target_beta] += scale * vec[row_offset + source_beta];
        }
    }
}

// Mixed terms apply:
// out[t_alpha, t_beta] += coeff * p_alpha * p_beta * vec[s_alpha, s_beta].
fn contract_mixed_terms(
    vec: &[Complex64],
    out: &mut [Complex64],
    dim_b: usize,
    coeffs: &[Complex64],
    alpha: &PhasedTransitions,
    beta: &PhasedTransitions,
    reverse: bool,
) {
    for (term_index, &coeff) in coeffs.iter().enumerate() {
        let coeff = directed_scale(coeff, reverse);

        let alpha_start = alpha.indptr[term_index];
        let alpha_end = alpha.indptr[term_index + 1];
        let beta_start = beta.indptr[term_index];
        let beta_end = beta.indptr[term_index + 1];

        if alpha_end - alpha_start <= beta_end - beta_start {
            for (alpha_offset, &alpha_phase_value) in
                alpha.phase[alpha_start..alpha_end].iter().enumerate()
            {
                let alpha_index = alpha_start + alpha_offset;
                let (source_alpha, target_alpha) =
                    src_dst(alpha.src, alpha.dst, alpha_index, reverse);
                let alpha_scale = coeff * Complex64::new(alpha_phase_value as f64, 0.0);
                let source_alpha_offset = source_alpha * dim_b;
                let target_alpha_offset = target_alpha * dim_b;
                for (beta_offset, &beta_phase_value) in
                    beta.phase[beta_start..beta_end].iter().enumerate()
                {
                    let beta_index = beta_start + beta_offset;
                    let (source_beta, target_beta) =
                        src_dst(beta.src, beta.dst, beta_index, reverse);
                    let scale = alpha_scale * Complex64::new(beta_phase_value as f64, 0.0);
                    out[target_alpha_offset + target_beta] +=
                        scale * vec[source_alpha_offset + source_beta];
                }
            }
            continue;
        }

        for (beta_offset, &beta_phase_value) in beta.phase[beta_start..beta_end].iter().enumerate()
        {
            let beta_index = beta_start + beta_offset;
            let (source_beta, target_beta) = src_dst(beta.src, beta.dst, beta_index, reverse);
            let beta_scale = coeff * Complex64::new(beta_phase_value as f64, 0.0);
            for (alpha_offset, &alpha_phase_value) in
                alpha.phase[alpha_start..alpha_end].iter().enumerate()
            {
                let alpha_index = alpha_start + alpha_offset;
                let (source_alpha, target_alpha) =
                    src_dst(alpha.src, alpha.dst, alpha_index, reverse);
                let scale = beta_scale * Complex64::new(alpha_phase_value as f64, 0.0);
                out[target_alpha * dim_b + target_beta] +=
                    scale * vec[source_alpha * dim_b + source_beta];
            }
        }
    }
}

/// Contract a packed FermionOperator into a buffer.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn contract_fermion_operator_into_buffer(
    vec: PyReadonlyArray2<Complex64>,
    scalar_coeffs: PyReadonlyArray1<Complex64>,
    alpha_only_src: PyReadonlyArray1<usize>,
    alpha_only_dst: PyReadonlyArray1<usize>,
    alpha_only_scale: PyReadonlyArray1<Complex64>,
    beta_only_src: PyReadonlyArray1<usize>,
    beta_only_dst: PyReadonlyArray1<usize>,
    beta_only_scale: PyReadonlyArray1<Complex64>,
    mixed_coeffs: PyReadonlyArray1<Complex64>,
    mixed_alpha_indptr: PyReadonlyArray1<usize>,
    mixed_alpha_src: PyReadonlyArray1<usize>,
    mixed_alpha_dst: PyReadonlyArray1<usize>,
    mixed_alpha_phase: PyReadonlyArray1<i8>,
    mixed_beta_indptr: PyReadonlyArray1<usize>,
    mixed_beta_src: PyReadonlyArray1<usize>,
    mixed_beta_dst: PyReadonlyArray1<usize>,
    mixed_beta_phase: PyReadonlyArray1<i8>,
    mut out: PyReadwriteArray2<Complex64>,
    reverse: bool,
) {
    let vec = vec.as_array();
    let shape = vec.shape();
    let dim_a = shape[0];
    let dim_b = shape[1];
    let vec = vec.as_slice().unwrap();

    let mut out = out.as_array_mut();
    let out = out.as_slice_mut().unwrap();

    let scalar_coeffs = scalar_coeffs.as_slice().unwrap();
    let alpha_only = ScaledTransitions {
        src: alpha_only_src.as_slice().unwrap(),
        dst: alpha_only_dst.as_slice().unwrap(),
        scale: alpha_only_scale.as_slice().unwrap(),
    };
    let beta_only = ScaledTransitions {
        src: beta_only_src.as_slice().unwrap(),
        dst: beta_only_dst.as_slice().unwrap(),
        scale: beta_only_scale.as_slice().unwrap(),
    };
    let mixed_coeffs = mixed_coeffs.as_slice().unwrap();
    let mixed_alpha = PhasedTransitions {
        indptr: mixed_alpha_indptr.as_slice().unwrap(),
        src: mixed_alpha_src.as_slice().unwrap(),
        dst: mixed_alpha_dst.as_slice().unwrap(),
        phase: mixed_alpha_phase.as_slice().unwrap(),
    };
    let mixed_beta = PhasedTransitions {
        indptr: mixed_beta_indptr.as_slice().unwrap(),
        src: mixed_beta_src.as_slice().unwrap(),
        dst: mixed_beta_dst.as_slice().unwrap(),
        phase: mixed_beta_phase.as_slice().unwrap(),
    };

    contract_scalar_terms(vec, scalar_coeffs, out, reverse);
    contract_alpha_only_terms(vec, out, dim_b, &alpha_only, reverse);
    contract_beta_only_terms(vec, out, dim_a, dim_b, &beta_only, reverse);
    contract_mixed_terms(
        vec,
        out,
        dim_b,
        mixed_coeffs,
        &mixed_alpha,
        &mixed_beta,
        reverse,
    );
}
