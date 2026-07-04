use metal::*;
use pyo3::prelude::*;
use crate::metal_runner::map_metal_err;

const KERNEL_SRC: &str = include_str!("kernels/delta_stepping.metal");

/// Metal-accelerated Delta-Stepping SSSP solver.
///
/// Implements the Meyer-Sanders Delta-Stepping algorithm on GPU.
/// Nodes are partitioned into buckets of width `delta`. Each iteration
/// processes all nodes in the current bucket, relaxing their outgoing edges.
/// Newly reached nodes are pushed to subsequent buckets.
///
/// This is the framework-level primitive; the orthoroute_mac example
/// uses a more advanced persistent-thread variant integrated into the
/// SPFA solver.
#[pyclass]
pub struct MetalDeltaStepping {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,
}

#[pymethods]
impl MetalDeltaStepping {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = Device::system_default()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device found!"))?;
        let command_queue = device.new_command_queue();

        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(KERNEL_SRC, &compile_options)
            .map_err(map_metal_err)?;

        let function = library.get_function("delta_stepping_relax", None).map_err(map_metal_err)?;
        let pipeline = device.new_compute_pipeline_state_with_function(&function).map_err(map_metal_err)?;

        Ok(MetalDeltaStepping {
            device,
            command_queue,
            pipeline,
        })
    }

    /// Run delta-stepping SSSP from a source node.
    ///
    /// Arguments:
    ///   row_ptr: CSR row pointer array (N+1 elements)
    ///   col_idx: CSR column index array (E elements)
    ///   weights: CSR edge weight array (E elements, must be non-negative)
    ///   source: source node index
    ///   delta: bucket width (controls granularity; smaller = more iterations, larger = more work per iteration)
    ///
    /// Returns: distance array (N elements, f32::INFINITY for unreachable nodes)
    pub fn compute(
        &self,
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        weights: Vec<f32>,
        source: u32,
        delta: f32,
    ) -> PyResult<Vec<f32>> {
        let num_nodes = (row_ptr.len() - 1) as u32;
        let options = MTLResourceOptions::StorageModeShared;

        // Graph structure buffers
        let rp_buf = self.device.new_buffer_with_data(
            row_ptr.as_ptr() as *const _, (row_ptr.len() * 4) as u64, options);
        let ci_buf = self.device.new_buffer_with_data(
            col_idx.as_ptr() as *const _, (col_idx.len() * 4) as u64, options);
        let wt_buf = self.device.new_buffer_with_data(
            weights.as_ptr() as *const _, (weights.len() * 4) as u64, options);

        // Distance buffer (atomics stored as uint bit patterns)
        let mut dist_bits = vec![0x7f800000u32; num_nodes as usize]; // f32::INFINITY as bits
        dist_bits[source as usize] = 0u32; // 0.0f32 as bits
        let dist_buf = self.device.new_buffer_with_data(
            dist_bits.as_ptr() as *const _, (num_nodes as usize * 4) as u64, options);

        // Double-buffered bucket node arrays
        let max_bucket_size = num_nodes as usize;
        let bucket_a_buf = self.device.new_buffer((max_bucket_size * 4) as u64, options);
        let bucket_b_buf = self.device.new_buffer((max_bucket_size * 4) as u64, options);
        let bucket_size_buf = self.device.new_buffer(4u64, options);

        // Delta constant buffer
        let delta_buf = self.device.new_buffer_with_data(
            &delta as *const _ as *const _, 4, options);

        // Initialize first bucket with source node
        unsafe {
            let ptr = bucket_a_buf.contents() as *mut u32;
            *ptr = source;
        }
        let mut current_bucket_size: u32 = 1;

        let mut current_buf = &bucket_a_buf;
        let mut next_buf = &bucket_b_buf;

        let tg_size = 256u64;
        let max_iters = num_nodes * 2; // Safety limit

        for _ in 0..max_iters {
            if current_bucket_size == 0 {
                break;
            }

            // Reset next bucket size to 0
            unsafe {
                let ptr = bucket_size_buf.contents() as *mut u32;
                *ptr = 0;
            }

            let threads = current_bucket_size as u64;

            let cb = self.command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&rp_buf), 0);
            enc.set_buffer(1, Some(&ci_buf), 0);
            enc.set_buffer(2, Some(&wt_buf), 0);
            enc.set_buffer(3, Some(&dist_buf), 0);
            enc.set_buffer(4, Some(current_buf), 0);
            enc.set_buffer(5, Some(next_buf), 0);
            enc.set_buffer(6, Some(&bucket_size_buf), 0);
            enc.set_buffer(7, Some(&delta_buf), 0);

            enc.dispatch_threads(
                MTLSize::new(threads, 1, 1),
                MTLSize::new(tg_size.min(threads).max(1), 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // Read next bucket size
            unsafe {
                let ptr = bucket_size_buf.contents() as *const u32;
                current_bucket_size = *ptr;
            }

            // Swap buffers
            std::mem::swap(&mut current_buf, &mut next_buf);
        }

        // Read back distances (convert from uint bit patterns to float)
        let mut result = vec![0.0f32; num_nodes as usize];
        unsafe {
            let ptr = dist_buf.contents() as *const u32;
            for i in 0..num_nodes as usize {
                result[i] = f32::from_bits(*ptr.add(i));
            }
        }

        Ok(result)
    }
}
