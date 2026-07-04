use pyo3::prelude::*;

pub mod metal_runner;
pub mod graph;
pub mod pagerank;
pub mod delta_stepping;
pub mod connected_components;
pub mod spmv;
pub mod primitives;
pub mod nbody;
pub mod accelerate_ops;
pub mod gnn;

use pagerank::MetalPageRank;
use delta_stepping::MetalDeltaStepping;
use connected_components::MetalConnectedComponents;
use metal_runner::MetalRunner;
use graph::Graph;
use spmv::MetalSpMV;
use primitives::MetalScanner;
use nbody::MetalNBody;
use accelerate_ops::AccelerateRunner;
use gnn::MetalGNN;

#[pymodule]
fn c2m_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // -- Graph Analytics --
    m.add_class::<MetalPageRank>()?;
    m.add_class::<MetalDeltaStepping>()?;
    m.add_class::<MetalConnectedComponents>()?;
    m.add_class::<Graph>()?;

    // -- Sparse Linear Algebra --
    m.add_class::<MetalSpMV>()?;

    // -- Parallel Primitives (Scan, Stream Compaction) --
    m.add_class::<MetalScanner>()?;

    // -- Physics Simulations --
    m.add_class::<MetalNBody>()?;

    // -- Graph Neural Networks --
    m.add_class::<MetalGNN>()?;

    // -- Apple Accelerate (AMX/BLAS) --
    m.add_class::<AccelerateRunner>()?;

    // -- Infrastructure --
    m.add_class::<MetalRunner>()?;

    Ok(())
}
