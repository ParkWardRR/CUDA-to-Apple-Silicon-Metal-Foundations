use pyo3::prelude::*;
use metal::*;
use std::mem;
use crate::metal_runner::map_metal_err;

const SPMV_KERNEL: &str = include_str!("kernels/spmv.metal");

#[pyclass]
pub struct MetalSpMV {
    device: Device,
    command_queue: CommandQueue,
    scalar_pipeline: ComputePipelineState,
    vector_pipeline: ComputePipelineState,
}

#[pymethods]
impl MetalSpMV {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = Device::system_default().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No Metal device found")
        })?;
        
        let command_queue = device.new_command_queue();
        
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SPMV_KERNEL, &options).map_err(map_metal_err)?;
        
        let scalar_func = library.get_function("spmv_csr_scalar", None).map_err(map_metal_err)?;
        let scalar_pipeline = device.new_compute_pipeline_state_with_function(&scalar_func).map_err(map_metal_err)?;
        
        let vector_func = library.get_function("spmv_csr_vector", None).map_err(map_metal_err)?;
        let vector_pipeline = device.new_compute_pipeline_state_with_function(&vector_func).map_err(map_metal_err)?;
        
        Ok(Self {
            device,
            command_queue,
            scalar_pipeline,
            vector_pipeline,
        })
    }

    /// Performs SpMV: y = A * x
    /// Uses scalar kernel (1 thread per row) or vector kernel (1 SIMD group per row)
    pub fn compute(
        &self,
        values: Vec<f32>,
        col_indices: Vec<u32>,
        row_ptr: Vec<u32>,
        x: Vec<f32>,
        use_vector_kernel: bool,
    ) -> PyResult<Vec<f32>> {
        let num_rows = (row_ptr.len() - 1) as u32;
        let options = MTLResourceOptions::StorageModeShared;
        
        let val_bytes = (values.len() * mem::size_of::<f32>()) as u64;
        let col_bytes = (col_indices.len() * mem::size_of::<u32>()) as u64;
        let ptr_bytes = (row_ptr.len() * mem::size_of::<u32>()) as u64;
        let x_bytes = (x.len() * mem::size_of::<f32>()) as u64;
        let y_bytes = (num_rows as usize * mem::size_of::<f32>()) as u64;
        
        let val_buffer = self.device.new_buffer_with_data(values.as_ptr() as *const _, val_bytes, options);
        let col_buffer = self.device.new_buffer_with_data(col_indices.as_ptr() as *const _, col_bytes, options);
        let ptr_buffer = self.device.new_buffer_with_data(row_ptr.as_ptr() as *const _, ptr_bytes, options);
        let x_buffer = self.device.new_buffer_with_data(x.as_ptr() as *const _, x_bytes, options);
        let y_buffer = self.device.new_buffer(y_bytes, options);
        let num_rows_buffer = self.device.new_buffer_with_data(&num_rows as *const _ as *const _, 4, options);

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        if use_vector_kernel {
            compute_encoder.set_compute_pipeline_state(&self.vector_pipeline);
        } else {
            compute_encoder.set_compute_pipeline_state(&self.scalar_pipeline);
        }
        
        compute_encoder.set_buffer(0, Some(&val_buffer), 0);
        compute_encoder.set_buffer(1, Some(&col_buffer), 0);
        compute_encoder.set_buffer(2, Some(&ptr_buffer), 0);
        compute_encoder.set_buffer(3, Some(&x_buffer), 0);
        compute_encoder.set_buffer(4, Some(&y_buffer), 0);
        compute_encoder.set_buffer(5, Some(&num_rows_buffer), 0);
        
        let threadgroup_size = 512;
        
        if use_vector_kernel {
            // 32 threads per row
            let total_threads = num_rows * 32;
            let mut grid_size = MTLSize::new(total_threads as u64, 1, 1);
            let mut tg_size = MTLSize::new(threadgroup_size, 1, 1);
            if total_threads < threadgroup_size as u32 {
                tg_size = MTLSize::new(total_threads.max(32) as u64, 1, 1);
                grid_size = MTLSize::new(total_threads.max(32) as u64, 1, 1);
            }
            compute_encoder.dispatch_threads(grid_size, tg_size);
        } else {
            // 1 thread per row
            let mut grid_size = MTLSize::new(num_rows as u64, 1, 1);
            let mut tg_size = MTLSize::new(threadgroup_size, 1, 1);
            if num_rows < threadgroup_size as u32 {
                tg_size = MTLSize::new(num_rows.max(1) as u64, 1, 1);
                grid_size = MTLSize::new(num_rows.max(1) as u64, 1, 1);
            }
            compute_encoder.dispatch_threads(grid_size, tg_size);
        }
        
        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        let mut result = vec![0.0f32; num_rows as usize];
        unsafe {
            let ptr = y_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), num_rows as usize);
        }
        
        drop(values);
        drop(col_indices);
        drop(row_ptr);
        drop(x);
        
        Ok(result)
    }
}
