use pyo3::prelude::*;
use metal::*;
use std::mem;
use crate::metal_runner::map_metal_err;

const PAGERANK_KERNEL: &str = include_str!("kernels/pagerank.metal");

#[pyclass]
pub struct MetalPageRank {
    device: Device,
    command_queue: CommandQueue,
    scalar_pipeline: ComputePipelineState,
    vector_pipeline: ComputePipelineState,
}

#[pymethods]
impl MetalPageRank {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = Device::system_default().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No Metal device found")
        })?;
        
        let command_queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device.new_library_with_source(PAGERANK_KERNEL, &options).map_err(map_metal_err)?;
        
        let scalar_func = library.get_function("pagerank_scalar", None).map_err(map_metal_err)?;
        let scalar_pipeline = device.new_compute_pipeline_state_with_function(&scalar_func).map_err(map_metal_err)?;
        
        let vector_func = library.get_function("pagerank_vector", None).map_err(map_metal_err)?;
        let vector_pipeline = device.new_compute_pipeline_state_with_function(&vector_func).map_err(map_metal_err)?;
        
        Ok(Self {
            device,
            command_queue,
            scalar_pipeline,
            vector_pipeline,
        })
    }

    pub fn compute(
        &self,
        values: Vec<f32>,
        col_indices: Vec<u32>,
        row_ptr: Vec<u32>,
        alpha: f32,
        max_iters: u32,
        use_vector_kernel: bool,
    ) -> PyResult<Vec<f32>> {
        let num_nodes = (row_ptr.len() - 1) as u32;
        let options = MTLResourceOptions::StorageModeShared;
        
        let val_bytes = (values.len() * mem::size_of::<f32>()) as u64;
        let col_bytes = (col_indices.len() * mem::size_of::<u32>()) as u64;
        let ptr_bytes = (row_ptr.len() * mem::size_of::<u32>()) as u64;
        let r_bytes = (num_nodes as usize * mem::size_of::<f32>()) as u64;
        
        let val_buffer = self.device.new_buffer_with_data(values.as_ptr() as *const _, val_bytes, options);
        let col_buffer = self.device.new_buffer_with_data(col_indices.as_ptr() as *const _, col_bytes, options);
        let ptr_buffer = self.device.new_buffer_with_data(row_ptr.as_ptr() as *const _, ptr_bytes, options);
        
        // Initialize r_old with 1/N
        let initial_r = vec![1.0f32 / num_nodes as f32; num_nodes as usize];
        let mut r_old = self.device.new_buffer_with_data(initial_r.as_ptr() as *const _, r_bytes, options);
        let mut r_new = self.device.new_buffer(r_bytes, options);
        
        let num_nodes_buffer = self.device.new_buffer_with_data(&num_nodes as *const _ as *const _, 4, options);
        let alpha_buffer = self.device.new_buffer_with_data(&alpha as *const _ as *const _, 4, options);

        let threadgroup_size = 512;
        
        for _ in 0..max_iters {
            let command_buffer = self.command_queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();
            
            if use_vector_kernel {
                compute_encoder.set_compute_pipeline_state(&self.vector_pipeline);
            } else {
                compute_encoder.set_compute_pipeline_state(&self.scalar_pipeline);
            }
            
            compute_encoder.set_buffer(0, Some(&ptr_buffer), 0);
            compute_encoder.set_buffer(1, Some(&col_buffer), 0);
            compute_encoder.set_buffer(2, Some(&val_buffer), 0);
            compute_encoder.set_buffer(3, Some(&r_old), 0);
            compute_encoder.set_buffer(4, Some(&r_new), 0);
            compute_encoder.set_buffer(5, Some(&num_nodes_buffer), 0);
            compute_encoder.set_buffer(6, Some(&alpha_buffer), 0);
            
            if use_vector_kernel {
                let total_threads = num_nodes * 32;
                let mut grid_size = MTLSize::new(total_threads as u64, 1, 1);
                let mut tg_size = MTLSize::new(threadgroup_size, 1, 1);
                if total_threads < threadgroup_size as u32 {
                    tg_size = MTLSize::new(total_threads.max(32) as u64, 1, 1);
                    grid_size = MTLSize::new(total_threads.max(32) as u64, 1, 1);
                }
                compute_encoder.dispatch_threads(grid_size, tg_size);
            } else {
                let mut grid_size = MTLSize::new(num_nodes as u64, 1, 1);
                let mut tg_size = MTLSize::new(threadgroup_size, 1, 1);
                if num_nodes < threadgroup_size as u32 {
                    tg_size = MTLSize::new(num_nodes.max(1) as u64, 1, 1);
                    grid_size = MTLSize::new(num_nodes.max(1) as u64, 1, 1);
                }
                compute_encoder.dispatch_threads(grid_size, tg_size);
            }
            
            compute_encoder.end_encoding();
            command_buffer.commit();
            // We can wait after each pass, or let the command queue pipeline it.
            // For benchmarking we will just wait at the end, but we need to swap buffers!
            // Actually, we must wait or use multiple command buffers if we swap pointers.
            // Let's just wait for simplicity and correctness, although double-buffering without waiting is faster.
            command_buffer.wait_until_completed();
            
            // Swap buffers
            let temp = r_old;
            r_old = r_new;
            r_new = temp;
        }
        
        let mut result = vec![0.0f32; num_nodes as usize];
        unsafe {
            // The final result is in r_old due to the swap at the end of the loop
            let ptr = r_old.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), num_nodes as usize);
        }
        
        Ok(result)
    }
}
