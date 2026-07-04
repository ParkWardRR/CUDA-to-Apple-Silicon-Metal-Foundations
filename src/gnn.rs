use pyo3::prelude::*;
use metal::*;
use crate::metal_runner::map_metal_err;

const GNN_KERNEL: &str = include_str!("kernels/gnn.metal");

/// Metal-accelerated Graph Neural Network (GNN) engine.
///
/// Implements GCN (Graph Convolutional Network) message passing entirely on GPU
/// using a single command buffer submission per forward pass — no CPU round-trips
/// between layers.
///
/// Architecture:
///   Layer i: H' = ReLU( D^{-1/2} A D^{-1/2} H W_i + b_i )
///   Final:   Z  = softmax( D^{-1/2} A D^{-1/2} H' W_out + b_out )
#[pyclass]
#[allow(dead_code)]
pub struct MetalGNN {
    device: Device,
    command_queue: CommandQueue,
    message_pipeline: ComputePipelineState,
    message_simd_pipeline: ComputePipelineState,
    linear_relu_pipeline: ComputePipelineState,
    linear_pipeline: ComputePipelineState,
    softmax_pipeline: ComputePipelineState,
    norm_pipeline: ComputePipelineState,
}

#[pymethods]
impl MetalGNN {
    #[new]
    pub fn new() -> PyResult<Self> {
        let device = Device::system_default().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No Metal device found")
        })?;

        let command_queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device.new_library_with_source(GNN_KERNEL, &options).map_err(map_metal_err)?;

        let msg_func = library.get_function("gnn_message_pass", None).map_err(map_metal_err)?;
        let message_pipeline = device.new_compute_pipeline_state_with_function(&msg_func).map_err(map_metal_err)?;

        let msg_simd_func = library.get_function("gnn_message_pass_simd", None).map_err(map_metal_err)?;
        let message_simd_pipeline = device.new_compute_pipeline_state_with_function(&msg_simd_func).map_err(map_metal_err)?;

        let lr_func = library.get_function("gnn_linear_relu", None).map_err(map_metal_err)?;
        let linear_relu_pipeline = device.new_compute_pipeline_state_with_function(&lr_func).map_err(map_metal_err)?;

        let l_func = library.get_function("gnn_linear", None).map_err(map_metal_err)?;
        let linear_pipeline = device.new_compute_pipeline_state_with_function(&l_func).map_err(map_metal_err)?;

        let sm_func = library.get_function("gnn_softmax", None).map_err(map_metal_err)?;
        let softmax_pipeline = device.new_compute_pipeline_state_with_function(&sm_func).map_err(map_metal_err)?;

        let norm_func = library.get_function("gnn_compute_norm_coeffs", None).map_err(map_metal_err)?;
        let norm_pipeline = device.new_compute_pipeline_state_with_function(&norm_func).map_err(map_metal_err)?;

        Ok(Self {
            device,
            command_queue,
            message_pipeline,
            message_simd_pipeline,
            linear_relu_pipeline,
            linear_pipeline,
            softmax_pipeline,
            norm_pipeline,
        })
    }

    /// Compute GCN-style symmetric normalization coefficients for the adjacency matrix.
    /// Returns edge_vals where edge_vals[e] = 1/sqrt(deg_src) * 1/sqrt(deg_dst).
    pub fn compute_norm_coeffs(
        &self,
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
    ) -> PyResult<Vec<f32>> {
        let num_nodes = (row_ptr.len() - 1) as u32;
        let num_edges = col_idx.len();
        let options = MTLResourceOptions::StorageModeShared;

        let rp_buf = self.device.new_buffer_with_data(
            row_ptr.as_ptr() as *const _, (row_ptr.len() * 4) as u64, options);
        let ci_buf = self.device.new_buffer_with_data(
            col_idx.as_ptr() as *const _, (num_edges * 4) as u64, options);
        let ev_buf = self.device.new_buffer((num_edges * 4) as u64, options);
        let nn_buf = self.device.new_buffer_with_data(
            &num_nodes as *const _ as *const _, 4, options);

        let cb = self.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.norm_pipeline);
        enc.set_buffer(0, Some(&rp_buf), 0);
        enc.set_buffer(1, Some(&ci_buf), 0);
        enc.set_buffer(2, Some(&ev_buf), 0);
        enc.set_buffer(3, Some(&nn_buf), 0);

        let tg = 256u64.min(num_nodes as u64).max(1);
        enc.dispatch_threads(
            MTLSize::new(num_nodes as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let mut result = vec![0.0f32; num_edges];
        unsafe {
            let ptr = ev_buf.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), num_edges);
        }
        Ok(result)
    }

    /// Run a multi-layer GCN forward pass in a SINGLE command buffer submission.
    ///
    /// This is the key architectural win: all layers are encoded sequentially into
    /// one command buffer. Metal guarantees sequential execution of compute encoders
    /// within a single command buffer, so no CPU synchronization is needed between layers.
    ///
    /// Arguments:
    ///   row_ptr, col_idx: CSR adjacency structure
    ///   edge_vals: normalized adjacency values (from compute_norm_coeffs)
    ///   features: input node features [N x F_in], flattened row-major
    ///   layer_weights: list of weight matrices, each [F_in_i x F_out_i] flattened
    ///   layer_biases: list of bias vectors, each [F_out_i]
    ///   feat_dims: list of feature dimensions [F_in, F_hidden1, ..., F_out]
    ///   num_nodes: number of nodes
    ///
    /// Returns: output logits [N x F_out], flattened row-major
    pub fn forward(
        &self,
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        edge_vals: Vec<f32>,
        features: Vec<f32>,
        layer_weights: Vec<Vec<f32>>,
        layer_biases: Vec<Vec<f32>>,
        feat_dims: Vec<u32>,
        num_nodes: u32,
    ) -> PyResult<Vec<f32>> {
        let num_layers = layer_weights.len();
        if num_layers == 0 || num_layers != layer_biases.len() || feat_dims.len() != num_layers + 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "layer_weights, layer_biases, and feat_dims must be consistent"
            ));
        }

        let options = MTLResourceOptions::StorageModeShared;

        // Graph structure buffers (shared across all layers)
        let rp_buf = self.device.new_buffer_with_data(
            row_ptr.as_ptr() as *const _, (row_ptr.len() * 4) as u64, options);
        let ci_buf = self.device.new_buffer_with_data(
            col_idx.as_ptr() as *const _, (col_idx.len() * 4) as u64, options);
        let ev_buf = self.device.new_buffer_with_data(
            edge_vals.as_ptr() as *const _, (edge_vals.len() * 4) as u64, options);
        let nn_buf = self.device.new_buffer_with_data(
            &num_nodes as *const _ as *const _, 4, options);

        // Input features buffer
        let mut h_current = self.device.new_buffer_with_data(
            features.as_ptr() as *const _, (features.len() * 4) as u64, options);

        // =====================================================================
        // SINGLE command buffer for the ENTIRE forward pass
        // =====================================================================
        let command_buffer = self.command_queue.new_command_buffer();

        for layer in 0..num_layers {
            let f_in = feat_dims[layer];
            let f_out = feat_dims[layer + 1];
            let is_last = layer == num_layers - 1;

            // Intermediate buffer for message aggregation result [N x F_in]
            let h_agg = self.device.new_buffer(
                (num_nodes as usize * f_in as usize * 4) as u64, options);

            // Output buffer for this layer [N x F_out]
            let h_next = self.device.new_buffer(
                (num_nodes as usize * f_out as usize * 4) as u64, options);

            // Weight and bias buffers for this layer
            let w_buf = self.device.new_buffer_with_data(
                layer_weights[layer].as_ptr() as *const _,
                (layer_weights[layer].len() * 4) as u64, options);
            let b_buf = self.device.new_buffer_with_data(
                layer_biases[layer].as_ptr() as *const _,
                (layer_biases[layer].len() * 4) as u64, options);

            let f_in_buf = self.device.new_buffer_with_data(
                &f_in as *const _ as *const _, 4, options);
            let f_out_buf = self.device.new_buffer_with_data(
                &f_out as *const _ as *const _, 4, options);

            // --- Pass 1: Message aggregation (SpMV: h_agg = A * h_current) ---
            {
                let enc = command_buffer.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&self.message_pipeline);
                enc.set_buffer(0, Some(&rp_buf), 0);
                enc.set_buffer(1, Some(&ci_buf), 0);
                enc.set_buffer(2, Some(&ev_buf), 0);
                enc.set_buffer(3, Some(&h_current), 0);
                enc.set_buffer(4, Some(&h_agg), 0);
                enc.set_buffer(5, Some(&nn_buf), 0);
                enc.set_buffer(6, Some(&f_in_buf), 0);

                let tg_x = 16u64.min(num_nodes as u64).max(1);
                let tg_y = 16u64.min(f_in as u64).max(1);
                enc.dispatch_threads(
                    MTLSize::new(num_nodes as u64, f_in as u64, 1),
                    MTLSize::new(tg_x, tg_y, 1),
                );
                enc.end_encoding();
            }

            // --- Pass 2: Linear transform (+ ReLU for hidden layers) ---
            {
                let enc = command_buffer.new_compute_command_encoder();
                if is_last {
                    enc.set_compute_pipeline_state(&self.linear_pipeline);
                } else {
                    enc.set_compute_pipeline_state(&self.linear_relu_pipeline);
                }
                enc.set_buffer(0, Some(&h_agg), 0);
                enc.set_buffer(1, Some(&w_buf), 0);
                enc.set_buffer(2, Some(&b_buf), 0);
                enc.set_buffer(3, Some(&h_next), 0);
                enc.set_buffer(4, Some(&nn_buf), 0);
                enc.set_buffer(5, Some(&f_in_buf), 0);
                enc.set_buffer(6, Some(&f_out_buf), 0);

                let tg_x = 16u64.min(num_nodes as u64).max(1);
                let tg_y = 16u64.min(f_out as u64).max(1);
                enc.dispatch_threads(
                    MTLSize::new(num_nodes as u64, f_out as u64, 1),
                    MTLSize::new(tg_x, tg_y, 1),
                );
                enc.end_encoding();
            }

            h_current = h_next;
        }

        // --- Pass 3: Softmax on the final layer output ---
        {
            let f_out_final = feat_dims[num_layers];
            let nc_buf = self.device.new_buffer_with_data(
                &f_out_final as *const _ as *const _, 4, options);

            let enc = command_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.softmax_pipeline);
            enc.set_buffer(0, Some(&h_current), 0);
            enc.set_buffer(1, Some(&nn_buf), 0);
            enc.set_buffer(2, Some(&nc_buf), 0);

            // One threadgroup per row, SIMD-width threads per group
            let tg = 32u64;
            enc.dispatch_thread_groups(
                MTLSize::new(num_nodes as u64, 1, 1),
                MTLSize::new(tg, 1, 1),
            );
            enc.end_encoding();
        }

        // =====================================================================
        // SINGLE commit + wait for the ENTIRE forward pass
        // =====================================================================
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back results
        let final_dim = feat_dims[num_layers] as usize;
        let result_len = num_nodes as usize * final_dim;
        let mut result = vec![0.0f32; result_len];
        unsafe {
            let ptr = h_current.contents() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), result_len);
        }

        Ok(result)
    }
}
