use pyo3::prelude::*;
use metal::*;
use std::mem;
use crate::metal_runner::map_metal_err;

const NBODY_KERNEL: &str = include_str!("kernels/nbody.metal");

#[pyclass]
pub struct MetalNBody {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,
}

#[pymethods]
impl MetalNBody {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = Device::system_default().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No Metal device found")
        })?;
        
        let command_queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device.new_library_with_source(NBODY_KERNEL, &options).map_err(map_metal_err)?;
        
        let func = library.get_function("nbody_integrate_tiled", None).map_err(map_metal_err)?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func).map_err(map_metal_err)?;
        
        Ok(Self {
            device,
            command_queue,
            pipeline,
        })
    }

    pub fn compute(
        &self,
        positions: Vec<f32>, // x, y, z, mass
        velocities: Vec<f32>, // vx, vy, vz, 0
        dt: f32,
        eps: f32,
        max_iters: u32,
    ) -> PyResult<Vec<f32>> {
        let num_bodies = (positions.len() / 4) as u32;
        let options = MTLResourceOptions::StorageModeShared;
        
        let bytes = (positions.len() * mem::size_of::<f32>()) as u64;
        
        let mut p_old = self.device.new_buffer_with_data(positions.as_ptr() as *const _, bytes, options);
        let mut p_new = self.device.new_buffer(bytes, options);
        let v_buffer = self.device.new_buffer_with_data(velocities.as_ptr() as *const _, bytes, options);
        
        let num_bodies_buffer = self.device.new_buffer_with_data(&num_bodies as *const _ as *const _, 4, options);
        let dt_buffer = self.device.new_buffer_with_data(&dt as *const _ as *const _, 4, options);
        let eps_buffer = self.device.new_buffer_with_data(&eps as *const _ as *const _, 4, options);

        let threadgroup_size = 256;
        let mut grid_size = MTLSize::new(num_bodies as u64, 1, 1);
        let mut tg_size = MTLSize::new(threadgroup_size, 1, 1);
        if num_bodies < threadgroup_size as u32 {
            tg_size = MTLSize::new(num_bodies.max(1) as u64, 1, 1);
            grid_size = MTLSize::new(num_bodies.max(1) as u64, 1, 1);
        }
        
        for _ in 0..max_iters {
            let command_buffer = self.command_queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();
            
            compute_encoder.set_compute_pipeline_state(&self.pipeline);
            
            compute_encoder.set_buffer(0, Some(&p_old), 0);
            compute_encoder.set_buffer(1, Some(&p_new), 0);
            compute_encoder.set_buffer(2, Some(&v_buffer), 0);
            compute_encoder.set_buffer(3, Some(&num_bodies_buffer), 0);
            compute_encoder.set_buffer(4, Some(&dt_buffer), 0);
            compute_encoder.set_buffer(5, Some(&eps_buffer), 0);
            
            compute_encoder.dispatch_threads(grid_size, tg_size);
            
            compute_encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            
            let temp = p_old;
            p_old = p_new;
            p_new = temp;
        }
        
        let mut result = vec![0.0f32; positions.len()];
        unsafe {
            let ptr = p_old.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), positions.len());
        }
        
        Ok(result)
    }
}
