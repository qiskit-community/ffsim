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
use pyo3::class::basic::CompareOp;
use pyo3::exceptions::PyKeyError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::cmp::Ordering;
use std::collections::HashMap;

#[pyclass]
struct KeysIterator {
    keys: std::vec::IntoIter<Vec<(bool, bool, i32)>>,
}

#[pymethods]
impl KeysIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyObject> {
        Python::with_gil(|py| {
            return self
                .keys
                .next()
                .map(|vec| PyTuple::new_bound(py, &vec).to_object(py));
        })
    }
}

/// A fermionic operator.
///
/// A FermionOperator represents a linear combination of products of fermionic creation
/// and annihilation operators. Initialize a FermionOperator by passing a dictionary mapping
/// the terms in the linear combination to their associated coefficients. FermionOperators
/// Can be added, subtracted, and multiplied, and they support multiplication and division
/// by scalars. When multiplying by a scalar, the scalar should go on the left side of the
/// multiplication operator, e.g. ``scalar * op``, not ``op * scalar``.
///
/// See :doc:`/how-to-guides/fermion-operator` for an explanation of how to use this class.
///
/// Example:
///
/// .. code-block:: python
///
///     # Note: Since FermionOperator is an unordered mapping, the order of
///     # the terms in the print outputs below may vary between runs.
///
///     import ffsim
///
///     op1 = ffsim.FermionOperator(
///         {
///             (ffsim.cre_a(0), ffsim.des_a(3)): 0.5,
///             (ffsim.cre_a(3), ffsim.des_a(0)): -0.25,
///             (ffsim.cre_b(1), ffsim.des_b(5), ffsim.cre_a(4)): 1 + 1j,
///         }
///     )
///     print(2 * op1)
///     # prints    
///     # FermionOperator({
///     #     (cre_b(1), des_b(5), cre_a(4)): 2+2j,
///     #     (cre_a(3), des_a(0)): -0.5,
///     #     (cre_a(0), des_a(3)): 1
///     # })
///
///     op2 = ffsim.FermionOperator(
///         {
///             (ffsim.cre_b(2),): 1j,
///             (ffsim.des_a(3), ffsim.des_b(3)): -0.25,
///         }
///     )
///     print(op1 + op2)
///     # prints
///     # FermionOperator({
///     #     (cre_a(3), des_a(0)): -0.25,
///     #     (cre_b(2)): 0+1j,
///     #     (des_a(3), des_b(3)): -0.25,
///     #     (cre_b(1), des_b(5), cre_a(4)): 1+1j,
///     #     (cre_a(0), des_a(3)): 0.5
///     # })
///
///     print(op1 * op2)
///     # prints
///     # FermionOperator({
///     #     (cre_b(1), des_b(5), cre_a(4), cre_b(2)): -1+1j,
///     #     (cre_a(0), des_a(3), des_a(3), des_b(3)): -0.125,
///     #     (cre_a(3), des_a(0), des_a(3), des_b(3)): 0.0625,
///     #     (cre_b(1), des_b(5), cre_a(4), des_a(3), des_b(3)): -0.25-0.25j,
///     #     (cre_a(0), des_a(3), cre_b(2)): 0+0.5j,
///     #     (cre_a(3), des_a(0), cre_b(2)): 0-0.25j
///     # })
///
/// Args:
///     coeffs (dict[tuple[tuple[bool, bool, int], ...], complex]): The coefficients of the
///         operator.
#[pyclass(module = "ffsim", mapping)]
#[derive(Clone)]
pub struct FermionOperator {
    coeffs: HashMap<Vec<(bool, bool, i32)>, Complex64>,
}

#[pymethods]
impl FermionOperator {
    #[new]
    fn new(coeffs: HashMap<Vec<(bool, bool, i32)>, Complex64>) -> Self {
        Self { coeffs }
    }

    #[classattr]
    const __hash__: Option<PyObject> = None;

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> PyResult<String> {
        let mut items_str = Vec::new();
        for (key, &val) in &self.coeffs {
            let key_parts: Vec<String> = key
                .iter()
                .map(|(action, spin, orb)| {
                    format!(
                        "({}, {}, {})",
                        if *action { "True" } else { "False" },
                        if *spin { "True" } else { "False" },
                        orb
                    )
                })
                .collect();
            let key_str = format!("({})", key_parts.join(", "));
            let val_str = if val.im < 0.0 {
                format!("{}{}j", val.re, val.im)
            } else {
                format!("{}+{}j", val.re, val.im)
            };
            items_str.push(format!("{}: {}", key_str, val_str));
        }
        Ok(format!("FermionOperator({{{}}})", items_str.join(", ")))
    }

    fn _repr_pretty_str(&self) -> String {
        let mut items_str = Vec::new();
        for (key, &val) in &self.coeffs {
            let key_parts: Vec<String> = key
                .iter()
                .map(|(action, spin, orb)| {
                    let action_str;
                    if !action && !spin {
                        action_str = "des_a";
                    } else if !action && *spin {
                        action_str = "des_b"
                    } else if *action && !spin {
                        action_str = "cre_a"
                    } else {
                        action_str = "cre_b"
                    }
                    format!("{}({})", action_str, orb)
                })
                .collect();
            let key_str = format!("({})", key_parts.join(", "));
            let val_str = if val.im == 0.0 {
                format!("{}", val.re)
            } else if val.im < 0.0 {
                format!("{}{}j", val.re, val.im)
            } else {
                format!("{}+{}j", val.re, val.im)
            };
            items_str.push(format!("    {}: {}", key_str, val_str));
        }
        format!("FermionOperator({{\n{}\n}})", items_str.join(",\n"))
    }

    fn _repr_pretty_(&self, p: PyObject, cycle: bool) -> PyResult<()> {
        Python::with_gil(|py| {
            if cycle {
                p.call_method1(py, "text", ("FermionOperator(...)",))?;
            } else {
                p.call_method1(py, "text", (self._repr_pretty_str(),))?;
            }
            Ok(())
        })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self._repr_pretty_str())
    }

    fn __getitem__(&self, key: Vec<(bool, bool, i32)>) -> PyResult<Complex64> {
        self.coeffs
            .get(&key)
            .ok_or_else(|| PyKeyError::new_err("Term not present in operator."))
            .copied()
    }

    fn __setitem__(&mut self, key: Vec<(bool, bool, i32)>, value: Complex64) {
        self.coeffs.insert(key, value);
    }

    fn __delitem__(&mut self, key: Vec<(bool, bool, i32)>) {
        self.coeffs.remove(&key);
    }

    fn __contains__(&self, key: Vec<(bool, bool, i32)>) -> bool {
        self.coeffs.contains_key(&key)
    }

    fn __len__(&self) -> usize {
        self.coeffs.len()
    }

    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<KeysIterator>> {
        let keys = slf.coeffs.keys().cloned().collect::<Vec<_>>().into_iter();
        Py::new(slf.py(), KeysIterator { keys })
    }

    fn __iadd__(&mut self, other: &Self) {
        for (term, coeff) in &other.coeffs {
            let val = self.coeffs.entry(term.to_vec()).or_default();
            *val += coeff;
        }
    }

    fn __add__(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.__iadd__(other);
        result
    }

    fn __isub__(&mut self, other: &Self) {
        for (term, coeff) in &other.coeffs {
            let val = self.coeffs.entry(term.to_vec()).or_default();
            *val -= coeff;
        }
    }

    fn __sub__(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.__isub__(other);
        result
    }

    fn __neg__(&self) -> Self {
        let mut result = self.clone();
        result.__imul__(Complex64::new(-1.0, 0.0));
        result
    }

    fn __itruediv__(&mut self, other: Complex64) {
        for coeff in self.coeffs.values_mut() {
            *coeff /= other
        }
    }

    fn __truediv__(&self, other: Complex64) -> Self {
        let mut coeffs = HashMap::new();
        for (term_1, coeff_1) in &self.coeffs {
            coeffs.insert(term_1.to_vec(), coeff_1 / other);
        }
        Self { coeffs }
    }

    fn __imul__(&mut self, other: Complex64) {
        for coeff in self.coeffs.values_mut() {
            *coeff *= other
        }
    }

    fn __rmul__(&self, other: Complex64) -> Self {
        let mut coeffs = HashMap::new();
        for (term_1, coeff_1) in &self.coeffs {
            coeffs.insert(term_1.to_vec(), other * coeff_1);
        }
        Self { coeffs }
    }

    fn __mul__(&self, other: &Self) -> Self {
        let mut coeffs = HashMap::new();
        for (term_1, coeff_1) in &self.coeffs {
            for (term_2, coeff_2) in &other.coeffs {
                let mut new_term = term_1.to_vec();
                new_term.extend(term_2);
                let val = coeffs.entry(new_term).or_insert(Complex64::default());
                *val += coeff_1 * coeff_2;
            }
        }
        Self { coeffs }
    }

    fn __pow__(&self, exponent: u32, modulo: Option<u32>) -> PyResult<Self> {
        match modulo {
            Some(_) => Err(PyValueError::new_err("mod argument not supported")),
            None => {
                let mut coeffs = HashMap::new();
                coeffs.insert(vec![], Complex64::new(1.0, 0.0));
                let mut result = Self { coeffs };
                for _ in 0..exponent {
                    result = result.__mul__(self);
                }
                Ok(result)
            }
        }
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp, py: Python<'_>) -> PyObject {
        match op {
            CompareOp::Eq => (self.coeffs == other.coeffs).into_py(py),
            CompareOp::Ne => (self.coeffs != other.coeffs).into_py(py),
            _ => py.NotImplemented(),
        }
    }

    /// Return the normal ordered form of the operator.
    ///
    /// The normal ordered form of an operator is an equivalent operator in which
    /// each term has been reordered into a canonical ordering.
    /// In each term of a normal-ordered fermion operator, the operators comprising
    /// the term appear from left to right in descending lexicographic order by
    /// (action, spin, orb). That is, all creation operators appear before all
    /// annihilation operators; within creation/annihilation operators, spin beta
    /// operators appear before spin alpha operators, and larger orbital indices
    /// appear before smaller orbital indices.
    ///
    /// Returns:
    ///     FermionOperator: The normal-ordered fermion operator.
    fn normal_ordered(&self) -> Self {
        let mut result = Self {
            coeffs: HashMap::new(),
        };
        for (term, coeff) in &self.coeffs {
            result.__iadd__(&_normal_ordered_term(term, coeff))
        }
        result
    }

    /// Return whether the operator conserves particle number.
    ///
    /// Returns:
    ///     bool: True if the operator conserves particle number, False otherwise.
    fn conserves_particle_number(&self) -> bool {
        for term in self.coeffs.keys() {
            let (create_count, destroy_count) =
                term.iter()
                    .fold((0, 0), |(create_acc, destroy_acc), &(action, _, _)| {
                        if action {
                            (create_acc + 1, destroy_acc)
                        } else {
                            (create_acc, destroy_acc + 1)
                        }
                    });
            if create_count != destroy_count {
                return false;
            }
        }
        true
    }

    /// Return whether the operator conserves the Z component of spin.
    ///
    /// Returns:
    ///     bool: True if the operator conserves the Z component of spin, False otherwise.
    fn conserves_spin_z(&self) -> bool {
        for term in self.coeffs.keys() {
            let (create_count_a, destroy_count_a, create_count_b, destroy_count_b) =
                term.iter().fold(
                    (0, 0, 0, 0),
                    |(create_acc_a, destroy_acc_a, create_acc_b, destroy_acc_b),
                     &(action, spin, _)| {
                        if spin {
                            // spin beta
                            if action {
                                // create
                                (create_acc_a, destroy_acc_a, create_acc_b + 1, destroy_acc_b)
                            } else {
                                //destroy
                                (create_acc_a, destroy_acc_a, create_acc_b, destroy_acc_b + 1)
                            }
                        } else {
                            // spin alpha
                            if action {
                                // create
                                (create_acc_a + 1, destroy_acc_a, create_acc_b, destroy_acc_b)
                            } else {
                                //destroy
                                (create_acc_a, destroy_acc_a + 1, create_acc_b, destroy_acc_b)
                            }
                        }
                    },
                );
            if create_count_a - destroy_count_a != create_count_b - destroy_count_b {
                return false;
            }
        }
        true
    }

    /// Return the many-body order of the operator.
    ///
    /// The many-body order is defined as the length of the longest term contained
    /// in the operator.
    ///
    /// Returns:
    ///     int: The many-body order of the operator.
    fn many_body_order(&self) -> usize {
        self.coeffs.keys().map(|term| term.len()).max().unwrap_or(0)
    }

    fn _approx_eq_(&self, other: &Self, rtol: f64, atol: f64) -> bool {
        for key in self.coeffs.keys().chain(other.coeffs.keys()) {
            let val_self = *self.coeffs.get(key).unwrap_or(&Complex64::default());
            let val_other = *other.coeffs.get(key).unwrap_or(&Complex64::default());
            if (val_self - val_other).norm() > atol + rtol * val_other.norm() {
                return false;
            }
        }
        true
    }
}

fn _normal_ordered_term(term: &[(bool, bool, i32)], coeff: &Complex64) -> FermionOperator {
    let mut coeffs = HashMap::new();
    let mut stack = vec![(term.to_vec(), *coeff)];
    while let Some((mut term, coeff)) = stack.pop() {
        let mut parity = false;
        for i in 1..term.len() {
            // shift the operator at index i to the left until it's in the correct location
            for j in (1..=i).rev() {
                let (action_right, spin_right, index_right) = term[j];
                let (action_left, spin_left, index_left) = term[j - 1];
                if action_right == action_left {
                    // both create or both destroy
                    match (spin_right, index_right).cmp(&(spin_left, index_left)) {
                        Ordering::Equal => {
                            // operators are the same, so product is zero
                            return FermionOperator {
                                coeffs: HashMap::new(),
                            };
                        }
                        Ordering::Greater => {
                            // swap operators and update sign
                            term.swap(j - 1, j);
                            parity = !parity;
                        }
                        Ordering::Less => {}
                    }
                } else if action_right && !action_left {
                    // create on right and destroy on left
                    if index_right == index_left {
                        // add new term
                        let mut new_term: Vec<(bool, bool, i32)> = Vec::new();
                        new_term.extend(&term[..j - 1]);
                        new_term.extend(&term[j + 1..]);
                        let signed_coeff = if parity { -coeff } else { coeff };
                        stack.push((new_term, signed_coeff))
                    }
                    // swap operators and update sign
                    term.swap(j - 1, j);
                    parity = !parity;
                }
            }
        }
        let signed_coeff = if parity { -coeff } else { coeff };
        let val = coeffs.entry(term).or_insert(Complex64::default());
        *val += signed_coeff;
    }
    FermionOperator { coeffs }
}
