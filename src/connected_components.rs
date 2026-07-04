use metal::*;
use pyo3::prelude::*;
use crate::metal_runner::map_metal_err;

const KERNEL_SRC: &str = include_str!("kernels/connected_components.metal");

#[pyclass]
pub struct MetalConnectedComponents {
    device: Device,
    command_queue: CommandQueue,
    hooking_pipeline: ComputePipelineState,
    jumping_pipeline: ComputePipelineState,
}

#[pymethods]
impl MetalConnectedComponents {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = Device::system_default()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device found!"))?;
        let command_queue = device.new_command_queue();

        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(KERNEL_SRC, &compile_options)
            .map_err(map_metal_err)?;

        let hooking_func = library.get_function("sv_hooking", None).map_err(map_metal_err)?;
        let hooking_pipeline = device.new_compute_pipeline_state_with_function(&hooking_func).map_err(map_metal_err)?;

        let jumping_func = library.get_function("sv_pointer_jumping", None).map_err(map_metal_err)?;
        let jumping_pipeline = device.new_compute_pipeline_state_with_function(&jumping_func).map_err(map_metal_err)?;

        Ok(MetalConnectedComponents {
            device,
            command_queue,
            hooking_pipeline,
            jumping_pipeline,
        })
    }

    pub fn compute(
        &self,
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        num_nodes: u32,
    ) -> PyResult<Vec<u32>> {
        let options = MTLResourceOptions::StorageModeShared;
        
        let row_ptr_buf = self.device.new_buffer_with_data(
            row_ptr.as_ptr() as *const _, (row_ptr.len() * 4) as u64, options);
        let col_idx_buf = self.device.new_buffer_with_data(
            col_idx.as_ptr() as *const _, (col_idx.len() * 4) as u64, options);
        
        let mut parent_array: Vec<u32> = (0..num_nodes).collect();
        let parent_buf = self.device.new_buffer_with_data(
            parent_array.as_ptr() as *const _, (num_nodes as usize * 4) as u64, options);

        let changed_buf = self.device.new_buffer(4u64, options);

        let mut iteration = 0;
        let max_iterations = 1000;

        loop {
            unsafe { *(changed_buf.contents() as *mut u32) = 0; }

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            
            // Hooking
            encoder.set_compute_pipeline_state(&self.hooking_pipeline);
            encoder.set_buffer(0, Some(&row_ptr_buf), 0);
            encoder.set_buffer(1, Some(&col_idx_buf), 0);
            encoder.set_buffer(2, Some(&parent_buf), 0);
            encoder.set_buffer(3, Some(&changed_buf), 0);
            
            let grid_size = MTLSize::new(num_nodes as u64, 1, 1);
            let threadgroup_size = MTLSize::new(256.min(num_nodes as u64).max(1), 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);

            // Pointer Jumping
            encoder.set_compute_pipeline_state(&self.jumping_pipeline);
            encoder.set_buffer(0, Some(&parent_buf), 0);
            encoder.set_buffer(1, Some(&changed_buf), 0);
            encoder.dispatch_threads(grid_size, threadgroup_size);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let changed = unsafe { *(changed_buf.contents() as *const u32) };
            if changed == 0 || iteration >= max_iterations {
                break;
            }
            iteration += 1;
        }

        unsafe {
            let ptr = parent_buf.contents() as *const u32;
            std::ptr::copy_nonoverlapping(ptr, parent_array.as_mut_ptr(), num_nodes as usize);
        }

        Ok(parent_array)
    }
}
