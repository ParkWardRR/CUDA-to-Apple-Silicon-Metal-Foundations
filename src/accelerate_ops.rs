use pyo3::prelude::*;

// C shim for Accelerate CBLAS (AMX offload).
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    pub fn cblas_sgemm(
        Order: i32,
        TransA: i32,
        TransB: i32,
        M: i32,
        N: i32,
        K: i32,
        alpha: f32,
        A: *const f32,
        lda: i32,
        B: *const f32,
        ldb: i32,
        beta: f32,
        C: *mut f32,
        ldc: i32,
    );
}

// CBLAS constants
const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

/// Demonstrates using the Apple Accelerate framework for dense operations.
/// By calling into Accelerate's BLAS functions (like `sgemm`), we implicitly offload 
/// to the Apple Matrix Coprocessor (AMX) when the matrix dimensions are large enough.
#[pyclass]
pub struct AccelerateRunner {}

#[pymethods]
impl AccelerateRunner {
    #[new]
    pub fn new() -> Self {
        AccelerateRunner {}
    }

    /// Performs single-precision general matrix multiplication (SGEMM) using Accelerate.
    /// `C = alpha * A * B + beta * C`
    /// A is M x K, B is K x N, C is M x N.
    pub fn sgemm(
        &self, 
        m: usize, 
        n: usize, 
        k: usize, 
        alpha: f32, 
        a: Vec<f32>, 
        b: Vec<f32>, 
        beta: f32, 
        mut c: Vec<f32>
    ) -> PyResult<Vec<f32>> {
        // Validation (rudimentary)
        if a.len() < m * k || b.len() < k * n || c.len() < m * n {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid matrix dimensions"));
        }

        // Wire up to the Apple Accelerate framework for AMX offload.
        // `cblas_sgemm` is hardware-accelerated on Apple Silicon AMX for large dimensions.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                n as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
            );
        }

        Ok(c)
    }
}
