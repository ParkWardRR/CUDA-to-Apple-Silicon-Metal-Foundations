use pyo3::prelude::*;
use metal::*;
use std::mem;
use crate::metal_runner::map_metal_err;

const SCAN_KERNEL: &str = include_str!("kernels/scan.metal");

#[pyclass]
pub struct MetalScanner {
    device: Device,
    command_queue: CommandQueue,
    single_pass_pipeline_f32: ComputePipelineState,
    single_pass_pipeline_u32: ComputePipelineState,
    eval_predicate_pipeline_f32: ComputePipelineState,
    stream_compact_pipeline: ComputePipelineState,
}

#[pymethods]
impl MetalScanner {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = Device::system_default().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No Metal device found")
        })?;
        
        let command_queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SCAN_KERNEL, &options).map_err(map_metal_err)?;
        
        let func_scan_f32 = library.get_function("single_pass_scan_f32", None).map_err(map_metal_err)?;
        let single_pass_pipeline_f32 = device.new_compute_pipeline_state_with_function(&func_scan_f32).map_err(map_metal_err)?;
        
        let func_scan_u32 = library.get_function("single_pass_scan_u32", None).map_err(map_metal_err)?;
        let single_pass_pipeline_u32 = device.new_compute_pipeline_state_with_function(&func_scan_u32).map_err(map_metal_err)?;
        
        let func_eval = library.get_function("evaluate_predicate_f32", None).map_err(map_metal_err)?;
        let eval_predicate_pipeline_f32 = device.new_compute_pipeline_state_with_function(&func_eval).map_err(map_metal_err)?;
        
        let func_compact = library.get_function("stream_compact_scatter", None).map_err(map_metal_err)?;
        let stream_compact_pipeline = device.new_compute_pipeline_state_with_function(&func_compact).map_err(map_metal_err)?;
        
        Ok(Self {
            device,
            command_queue,
            single_pass_pipeline_f32,
            single_pass_pipeline_u32,
            eval_predicate_pipeline_f32,
            stream_compact_pipeline,
        })
    }

    /// Performs an exclusive prefix scan on an array of f32 using decoupled look-back.
    pub fn scan_f32_vec(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let n = input.len() as u64;
        let threadgroup_size = 512;
        let num_blocks = (n + threadgroup_size - 1) / threadgroup_size;
        
        let input_bytes = n * mem::size_of::<f32>() as u64;
        let options = MTLResourceOptions::StorageModeShared;
        
        let in_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const _,
            input_bytes,
            options,
        );
        let out_buffer = self.device.new_buffer(input_bytes, options);
        
        let state_vec = vec![0u32; num_blocks as usize];
        let status_buffer = self.device.new_buffer_with_data(
            state_vec.as_ptr() as *const _,
            num_blocks * 4,
            options,
        );
        let agg_buffer = self.device.new_buffer_with_data(
            state_vec.as_ptr() as *const _,
            num_blocks * 4,
            options,
        );
        let prefix_buffer = self.device.new_buffer_with_data(
            state_vec.as_ptr() as *const _,
            num_blocks * 4,
            options,
        );
        
        let global_counter_val = 0u32;
        let global_counter_buffer = self.device.new_buffer_with_data(
            &global_counter_val as *const _ as *const _,
            4,
            options,
        );
        
        let n_val = n as u32;
        let n_buffer = self.device.new_buffer_with_data(
            &n_val as *const _ as *const _,
            4,
            options,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        compute_encoder.set_compute_pipeline_state(&self.single_pass_pipeline_f32);
        compute_encoder.set_buffer(0, Some(&in_buffer), 0);
        compute_encoder.set_buffer(1, Some(&out_buffer), 0);
        compute_encoder.set_buffer(2, Some(&status_buffer), 0);
        compute_encoder.set_buffer(3, Some(&agg_buffer), 0);
        compute_encoder.set_buffer(4, Some(&prefix_buffer), 0);
        compute_encoder.set_buffer(5, Some(&global_counter_buffer), 0);
        compute_encoder.set_buffer(6, Some(&n_buffer), 0);
        
        let grid_size = MTLSize::new(num_blocks, 1, 1);
        let tg_size = MTLSize::new(threadgroup_size, 1, 1);
        compute_encoder.dispatch_thread_groups(grid_size, tg_size);
        
        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        let mut result = vec![0.0f32; n as usize];
        unsafe {
            let ptr = out_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), n as usize);
        }
        
        drop(input);
        Ok(result)
    }

    /// Stream compacts an array of f32, keeping only elements > 0.0
    pub fn compact_f32_vec(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let n = input.len() as u64;
        let threadgroup_size = 512;
        let num_blocks = (n + threadgroup_size - 1) / threadgroup_size;
        
        let input_bytes = n * mem::size_of::<f32>() as u64;
        let uint_bytes = n * mem::size_of::<u32>() as u64;
        let options = MTLResourceOptions::StorageModeShared;
        
        let in_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const _,
            input_bytes,
            options,
        );
        let predicate_buffer = self.device.new_buffer(uint_bytes, options);
        let scan_buffer = self.device.new_buffer(uint_bytes, options);
        let out_buffer = self.device.new_buffer(input_bytes, options);
        
        let state_vec = vec![0u32; num_blocks as usize];
        let status_buffer = self.device.new_buffer_with_data(state_vec.as_ptr() as *const _, num_blocks * 4, options);
        let agg_buffer = self.device.new_buffer_with_data(state_vec.as_ptr() as *const _, num_blocks * 4, options);
        let prefix_buffer = self.device.new_buffer_with_data(state_vec.as_ptr() as *const _, num_blocks * 4, options);
        
        let zero_val: u32 = 0;
        let global_counter_buffer = self.device.new_buffer_with_data(&zero_val as *const _ as *const _, 4, options);
        let zero_val2: u32 = 0;
        let total_count_buffer = self.device.new_buffer_with_data(&zero_val2 as *const _ as *const _, 4, options);
        let n_val = n as u32;
        let n_buffer = self.device.new_buffer_with_data(&n_val as *const _ as *const _, 4, options);

        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        // Pass 1: Evaluate predicate
        compute_encoder.set_compute_pipeline_state(&self.eval_predicate_pipeline_f32);
        compute_encoder.set_buffer(0, Some(&in_buffer), 0);
        compute_encoder.set_buffer(1, Some(&predicate_buffer), 0);
        compute_encoder.set_buffer(2, Some(&n_buffer), 0);
        
        let mut eval_grid_size = MTLSize::new(n, 1, 1);
        let mut eval_tg_size = MTLSize::new(threadgroup_size, 1, 1);
        if n < threadgroup_size {
            eval_tg_size = MTLSize::new(n.max(1), 1, 1);
            eval_grid_size = MTLSize::new(n.max(1), 1, 1);
        }
        compute_encoder.dispatch_threads(eval_grid_size, eval_tg_size);
        
        // Pass 2: Prefix scan (u32)
        compute_encoder.set_compute_pipeline_state(&self.single_pass_pipeline_u32);
        compute_encoder.set_buffer(0, Some(&predicate_buffer), 0);
        compute_encoder.set_buffer(1, Some(&scan_buffer), 0);
        compute_encoder.set_buffer(2, Some(&status_buffer), 0);
        compute_encoder.set_buffer(3, Some(&agg_buffer), 0);
        compute_encoder.set_buffer(4, Some(&prefix_buffer), 0);
        compute_encoder.set_buffer(5, Some(&global_counter_buffer), 0);
        compute_encoder.set_buffer(6, Some(&n_buffer), 0);
        
        let grid_size = MTLSize::new(num_blocks, 1, 1);
        let tg_size = MTLSize::new(threadgroup_size, 1, 1);
        compute_encoder.dispatch_thread_groups(grid_size, tg_size);
        
        // Pass 3: Scatter
        compute_encoder.set_compute_pipeline_state(&self.stream_compact_pipeline);
        compute_encoder.set_buffer(0, Some(&in_buffer), 0);
        compute_encoder.set_buffer(1, Some(&predicate_buffer), 0);
        compute_encoder.set_buffer(2, Some(&scan_buffer), 0);
        compute_encoder.set_buffer(3, Some(&out_buffer), 0);
        compute_encoder.set_buffer(4, Some(&total_count_buffer), 0);
        compute_encoder.set_buffer(5, Some(&n_buffer), 0);
        
        compute_encoder.dispatch_threads(eval_grid_size, eval_tg_size);
        
        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        let total_count: u32 = unsafe { *(total_count_buffer.contents() as *const u32) };
        
        let mut result = vec![0.0f32; total_count as usize];
        unsafe {
            let ptr = out_buffer.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), total_count as usize);
        }
        
        drop(input);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_f32_small() {
        let scanner = MetalScanner::new().expect("Failed to initialize MetalScanner");
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = scanner.scan_f32_vec(input).unwrap();
        assert_eq!(result[0..5], [0.0, 1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_compact_f32() {
        let scanner = MetalScanner::new().unwrap();
        let input = vec![-1.0, 2.0, 0.0, 4.0, -5.0, 6.0];
        let result = scanner.compact_f32_vec(input).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }
}
