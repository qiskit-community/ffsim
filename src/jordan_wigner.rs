use crate::fermion_operator::FermionOperator;
use numpy::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::sync::Once;

type SparseLabel = String;
type SparseIndices = Vec<usize>;
type SparseCoeff = Complex64;
type SparseListEntry = (SparseLabel, SparseIndices, SparseCoeff);
type SparseList = Vec<SparseListEntry>;

/// Jordan–Wigner map of a FermionOperator with Rayon multithreading support.
///
/// Returns (sparse_list, num_qubits) where sparse_list is used to construct a Sparse Pauli Operator in Qiskit.
#[pyfunction]
pub fn jordan_wigner_qiskit(
    op: &FermionOperator,
    norb: usize,
    tol: f64,
) -> PyResult<(SparseList, usize)> {
    init_thread_pool(); // Build the global Rayon pool once

    let (sparse_list, n_qubits) = Python::with_gil(|py| {
        py.allow_threads(|| {
            let n_qubits: usize = 2 * norb;

            // ---- helpers ----
            #[inline]
            fn mul_pauli(a: u8, b: u8) -> (u8, Complex64) {
                match (a, b) {
                    (b'I', p) => (p, Complex64::new(1.0, 0.0)),
                    (p, b'I') => (p, Complex64::new(1.0, 0.0)),
                    (b'X', b'X') | (b'Y', b'Y') | (b'Z', b'Z') => (b'I', Complex64::new(1.0, 0.0)),
                    (b'X', b'Y') => (b'Z', Complex64::new(0.0, 1.0)),
                    (b'X', b'Z') => (b'Y', Complex64::new(0.0, -1.0)),
                    (b'Y', b'X') => (b'Z', Complex64::new(0.0, -1.0)),
                    (b'Y', b'Z') => (b'X', Complex64::new(0.0, 1.0)),
                    (b'Z', b'X') => (b'Y', Complex64::new(0.0, 1.0)),
                    (b'Z', b'Y') => (b'X', Complex64::new(0.0, -1.0)),
                    _ => unreachable!(),
                }
            }

            #[inline]
            fn multiply_by_zs_and_main(
                s: &mut [u8],
                zs: &[usize],
                q: usize,
                main: u8,
            ) -> Complex64 {
                let mut phase = Complex64::new(1.0, 0.0);
                for &i in zs {
                    let (c, p) = mul_pauli(s[i], b'Z');
                    s[i] = c;
                    phase *= p;
                }
                let (c, p) = mul_pauli(s[q], main);
                s[q] = c;
                phase *= p;
                phase
            }

            // Identity Pauli string template reused per term.
            let identity = vec![b'I'; n_qubits];

            // Parallel outer accumulation over all fermion terms
            let acc: HashMap<Vec<u8>, Complex64> = op
                .coeffs()
                .par_iter()
                .fold(
                    HashMap::<Vec<u8>, Complex64>::new,
                    |mut acc_local, (ops, &term_coeff)| {
                        // dense Pauli bytes -> coeff for the specific term's expansion
                        let mut current: HashMap<Vec<u8>, Complex64> = HashMap::new();
                        current.insert(identity.clone(), Complex64::new(1.0, 0.0));

                        // ops: Vec<(action, spin, orb)>
                        // action: true=creation, false=annihilation
                        // spin:   false=alpha, true=beta
                        for &(action, spin, orb_i32) in ops {
                            let orb = orb_i32 as usize;
                            let q = orb + if spin { norb } else { 0 };
                            let z_positions: Vec<usize> = (0..q).collect();

                            // a^dag = (X - iY)/2, a = (X + iY)/2
                            let coeff_x = Complex64::new(0.5, 0.0);
                            let coeff_y = if action {
                                Complex64::new(0.0, -0.5)
                            } else {
                                Complex64::new(0.0, 0.5)
                            };

                            let mut next: HashMap<Vec<u8>, Complex64> = HashMap::new();
                            for (ps, c) in current.into_iter() {
                                // X branch
                                let mut s_x = ps.clone();
                                let phase_x =
                                    multiply_by_zs_and_main(&mut s_x, &z_positions, q, b'X');
                                *next.entry(s_x).or_insert(Complex64::new(0.0, 0.0)) +=
                                    c * coeff_x * phase_x;

                                // Y branch
                                let mut s_y = ps;
                                let phase_y =
                                    multiply_by_zs_and_main(&mut s_y, &z_positions, q, b'Y');
                                *next.entry(s_y).or_insert(Complex64::new(0.0, 0.0)) +=
                                    c * coeff_y * phase_y;
                            }
                            current = next;
                        }

                        // Accumulating term contribution into the thread-local accumulator.
                        for (ps, c) in current.into_iter() {
                            let w = c * term_coeff;
                            if w.re.abs() > tol || w.im.abs() > tol {
                                *acc_local.entry(ps).or_insert(Complex64::new(0.0, 0.0)) += w;
                            }
                        }
                        acc_local
                    },
                )
                .reduce(HashMap::<Vec<u8>, Complex64>::new, |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(Complex64::new(0.0, 0.0)) += v;
                    }
                    a
                });

            // Convert dense bytes to compact sparse_list triples
            let mut sparse_list: SparseList = Vec::with_capacity(acc.len());
            for (ps_bytes, w) in acc.into_iter() {
                if w.re.abs() <= tol && w.im.abs() <= tol {
                    continue;
                }
                let mut indices: Vec<usize> = Vec::new();
                let mut label = String::new();
                for (q, &ch) in ps_bytes.iter().enumerate() {
                    if ch != b'I' {
                        indices.push(q);
                        label.push(ch as char);
                    }
                }
                // Identity term allowed as ("", [], coeff)
                sparse_list.push((label, indices, w));
            }

            Ok::<_, ()>((sparse_list, n_qubits))
        })
    })
    .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("JW mapping failed"))?;

    Ok((sparse_list, n_qubits))
}

// Initialize the global Rayon thread-pool once, using env if provided.
fn init_thread_pool() {
    static START: Once = Once::new();
    START.call_once(|| {
        // 1) RAYON_NUM_THREADS (Rayon's native env var)
        // 2) OMP_NUM_THREADS (OpenMP-style)
        // 3) One of the common BLAS-style env vars
        // 4) Default (Rayon decides)
        let from_env = |k: &str| env::var(k).ok().and_then(|v| v.parse::<usize>().ok());

        let n_threads = from_env("RAYON_NUM_THREADS")
            .or_else(|| from_env("OMP_NUM_THREADS"))
            .or_else(|| from_env("OPENBLAS_NUM_THREADS"))
            .or_else(|| from_env("MKL_NUM_THREADS"))
            .or_else(|| from_env("NUMEXPR_NUM_THREADS"));

        if let Some(n) = n_threads {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n.max(1))
                .build_global();
        } else {
            // Let Rayon auto-detect (logical CPUs) if no env var is set.
            let _ = rayon::ThreadPoolBuilder::new().build_global();
        }
    });
}
