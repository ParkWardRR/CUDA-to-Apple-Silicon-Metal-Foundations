use pyo3::prelude::*;
use pyo3::exceptions::PyNotImplementedError;

pub mod metal_runner;
pub mod graph;
pub mod pagerank;
pub mod delta_stepping;
pub mod connected_components;

use pagerank::MetalPageRank;
use delta_stepping::MetalDeltaStepping;
use connected_components::MetalConnectedComponents;
use metal_runner::MetalRunner;
use graph::Graph;

// ---------------------------------------------------------------------------
// Stub functions for upcoming paradigms.
// These give developers a clear roadmap of what is coming without cluttering
// the library with incomplete implementations.
// ---------------------------------------------------------------------------

#[pyfunction]
fn gemm() -> PyResult<()> {
    Err(PyNotImplementedError::new_err(
        "c2m_core.linalg.gemm() is an upcoming implementation that maps NVIDIA PTX \
         `mma.sync` Tensor Core algorithms to Apple AMX `simdgroup_matrix` primitives. \
         See the CUDA-to-Apple-Silicon-Metal-Foundations roadmap for details."
    ))
}

#[pyfunction]
fn convolve2d() -> PyResult<()> {
    Err(PyNotImplementedError::new_err(
        "c2m_core.stencil.convolve2d() is an upcoming implementation that dispatches \
         `MTLSize(x,y,1)` grids to leverage Apple Silicon's massive L2 cache for \
         2D spatial locality (e.g., Rodinia Hotspot). Check back soon!"
    ))
}

#[pyfunction]
fn scan() -> PyResult<()> {
    Err(PyNotImplementedError::new_err(
        "c2m_core.reduce.scan() is an upcoming implementation mapping CUDA warp \
         primitives (`__shfl_up_sync`) to Metal's `simd_shuffle_up` for Blelloch \
         prefix sums and stream compaction. Check back soon!"
    ))
}

#[pyfunction]
fn nbody() -> PyResult<()> {
    Err(PyNotImplementedError::new_err(
        "c2m_core.physics.nbody() is an upcoming implementation of O(N^2) \
         gravitational N-Body simulation on Metal compute shaders. Check back soon!"
    ))
}

#[pymodule]
fn c2m_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // -- Graph Analytics (fully implemented) --
    m.add_class::<MetalPageRank>()?;
    m.add_class::<MetalDeltaStepping>()?;
    m.add_class::<MetalConnectedComponents>()?;
    m.add_class::<MetalRunner>()?;
    m.add_class::<Graph>()?;

    // -- Stub functions for upcoming paradigms --
    // linalg
    m.add_function(wrap_pyfunction!(gemm, m)?)?;
    // stencil
    m.add_function(wrap_pyfunction!(convolve2d, m)?)?;
    // reduce
    m.add_function(wrap_pyfunction!(scan, m)?)?;
    // physics
    m.add_function(wrap_pyfunction!(nbody, m)?)?;

    Ok(())
}
