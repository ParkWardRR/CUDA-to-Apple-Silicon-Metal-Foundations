use pyo3::prelude::*;
use metal::*;

/// Maps a String error (often backed by NSError in metal-rs) into a Python Exception
pub fn map_metal_err(err: String) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Metal Error: {}", err))
}

/// A struct that manages the Metal device and command queue.
#[pyclass]
#[allow(dead_code)]
pub struct MetalRunner {
    device: Device,
    command_queue: CommandQueue,
}

#[pymethods]
impl MetalRunner {
    #[new]
    pub fn new() -> PyResult<Self> {
        // Grab the default Metal device
        let device = Device::system_default().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No Metal device found on this system")
        })?;
        
        let command_queue = device.new_command_queue();
        
        Ok(MetalRunner {
            device,
            command_queue,
        })
    }
}
