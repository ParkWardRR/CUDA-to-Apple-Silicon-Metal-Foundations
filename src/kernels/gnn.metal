#include <metal_stdlib>
using namespace metal;

// =============================================================================
// GNN Message Passing Kernels for Graph Neural Networks
// =============================================================================
// These kernels implement the core GNN computation pattern:
//   1. Message:   aggregate neighbor features via SpMV (A * H)
//   2. Transform: apply learned weights (H_agg * W + b)
//   3. Activate:  apply non-linearity (ReLU, softmax)
//
// The graph is stored in CSR format. Feature matrices are dense (N x F).
// All kernels are designed for single-command-buffer execution (no CPU sync).
// =============================================================================

// -----------------------------------------------------------------------------
// GNN Message Aggregation (SpMV on feature matrix columns)
// Computes: H_out[row, f] = sum_j{ A[row,j] * H_in[j, f] }
// One thread handles one (row, feature) pair.
// -----------------------------------------------------------------------------
kernel void gnn_message_pass(
    device const uint*  row_ptr    [[buffer(0)]],
    device const uint*  col_idx    [[buffer(1)]],
    device const float* edge_vals  [[buffer(2)]],   // normalized adjacency values (D^-1 * A)
    device const float* h_in       [[buffer(3)]],   // input features  [N x F_in]
    device float*       h_out      [[buffer(4)]],   // output features [N x F_in]
    constant uint&      num_nodes  [[buffer(5)]],
    constant uint&      feat_dim   [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row = tid.x;
    uint f   = tid.y;

    if (row >= num_nodes || f >= feat_dim) return;

    uint start = row_ptr[row];
    uint end   = row_ptr[row + 1];

    float sum = 0.0f;
    for (uint e = start; e < end; ++e) {
        uint neighbor = col_idx[e];
        float a_val   = edge_vals[e];
        sum += a_val * h_in[neighbor * feat_dim + f];
    }

    h_out[row * feat_dim + f] = sum;
}

// -----------------------------------------------------------------------------
// GNN Message Aggregation — SIMD-vectorized variant
// One SIMD group (32 threads) collaboratively aggregates one row for one feature.
// Better for high-degree nodes.
// -----------------------------------------------------------------------------
kernel void gnn_message_pass_simd(
    device const uint*  row_ptr    [[buffer(0)]],
    device const uint*  col_idx    [[buffer(1)]],
    device const float* edge_vals  [[buffer(2)]],
    device const float* h_in       [[buffer(3)]],
    device float*       h_out      [[buffer(4)]],
    constant uint&      num_nodes  [[buffer(5)]],
    constant uint&      feat_dim   [[buffer(6)]],
    uint  grid_tid                 [[thread_position_in_grid]],
    uint  lid                      [[thread_position_in_threadgroup]]
) {
    // Each SIMD group handles one (row, feature) pair
    uint simd_id = grid_tid / 32;
    uint lane    = lid % 32;
    uint row     = simd_id / feat_dim;
    uint f       = simd_id % feat_dim;

    if (row >= num_nodes || f >= feat_dim) return;

    uint start = row_ptr[row];
    uint end   = row_ptr[row + 1];

    float sum = 0.0f;
    for (uint e = start + lane; e < end; e += 32) {
        uint neighbor = col_idx[e];
        float a_val   = edge_vals[e];
        sum += a_val * h_in[neighbor * feat_dim + f];
    }

    float row_sum = simd_sum(sum);
    if (lane == 0) {
        h_out[row * feat_dim + f] = row_sum;
    }
}

// -----------------------------------------------------------------------------
// Linear Transform + ReLU Activation (fused)
// Computes: H_out[i, f_out] = max(0, sum_k{ H_in[i, k] * W[k, f_out] } + bias[f_out])
// One thread computes one output element.
// -----------------------------------------------------------------------------
kernel void gnn_linear_relu(
    device const float* h_in       [[buffer(0)]],   // [N x F_in]
    device const float* weights    [[buffer(1)]],   // [F_in x F_out]
    device const float* bias       [[buffer(2)]],   // [F_out]
    device float*       h_out      [[buffer(3)]],   // [N x F_out]
    constant uint&      num_nodes  [[buffer(4)]],
    constant uint&      f_in       [[buffer(5)]],
    constant uint&      f_out      [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint node = tid.x;
    uint j    = tid.y;

    if (node >= num_nodes || j >= f_out) return;

    float sum = bias[j];
    for (uint k = 0; k < f_in; ++k) {
        sum += h_in[node * f_in + k] * weights[k * f_out + j];
    }

    h_out[node * f_out + j] = max(0.0f, sum);  // ReLU
}

// -----------------------------------------------------------------------------
// Linear Transform (no activation) — for the final classification layer
// Computes: H_out[i, f_out] = sum_k{ H_in[i, k] * W[k, f_out] } + bias[f_out]
// -----------------------------------------------------------------------------
kernel void gnn_linear(
    device const float* h_in       [[buffer(0)]],
    device const float* weights    [[buffer(1)]],
    device const float* bias       [[buffer(2)]],
    device float*       h_out      [[buffer(3)]],
    constant uint&      num_nodes  [[buffer(4)]],
    constant uint&      f_in       [[buffer(5)]],
    constant uint&      f_out      [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint node = tid.x;
    uint j    = tid.y;

    if (node >= num_nodes || j >= f_out) return;

    float sum = bias[j];
    for (uint k = 0; k < f_in; ++k) {
        sum += h_in[node * f_in + k] * weights[k * f_out + j];
    }

    h_out[node * f_out + j] = sum;
}

// -----------------------------------------------------------------------------
// Row-wise Softmax for classification output
// Pass 1: compute max per row (for numerical stability)
// Pass 2: compute exp(x - max) and sum
// Pass 3: normalize by sum
// Fused into a single kernel with SIMD reductions.
// One threadgroup per row.
// -----------------------------------------------------------------------------
kernel void gnn_softmax(
    device float* data             [[buffer(0)]],   // [N x C] — in-place
    constant uint& num_nodes       [[buffer(1)]],
    constant uint& num_classes     [[buffer(2)]],
    uint row                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (row >= num_nodes) return;

    device float* row_data = data + row * num_classes;

    // Pass 1: find max (for numerical stability)
    float local_max = -INFINITY;
    for (uint j = lid; j < num_classes; j += tg_size) {
        local_max = max(local_max, row_data[j]);
    }
    float row_max = simd_max(local_max);

    // Pass 2: compute exp(x - max) and local sum
    float local_sum = 0.0f;
    for (uint j = lid; j < num_classes; j += tg_size) {
        float val = exp(row_data[j] - row_max);
        row_data[j] = val;
        local_sum += val;
    }
    float row_sum = simd_sum(local_sum);

    // Pass 3: normalize
    for (uint j = lid; j < num_classes; j += tg_size) {
        row_data[j] /= row_sum;
    }
}

// -----------------------------------------------------------------------------
// Degree-normalized adjacency computation
// Computes: edge_vals_out[e] = 1.0 / sqrt(degree[src]) / sqrt(degree[dst])
// For GCN-style symmetric normalization: D^{-1/2} A D^{-1/2}
// -----------------------------------------------------------------------------
kernel void gnn_compute_norm_coeffs(
    device const uint*  row_ptr       [[buffer(0)]],
    device const uint*  col_idx       [[buffer(1)]],
    device float*       edge_vals_out [[buffer(2)]],
    constant uint&      num_nodes     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_nodes) return;

    uint start = row_ptr[tid];
    uint end   = row_ptr[tid + 1];
    float deg_src_inv_sqrt = rsqrt((float)(end - start));

    for (uint e = start; e < end; ++e) {
        uint neighbor = col_idx[e];
        uint n_start = row_ptr[neighbor];
        uint n_end   = row_ptr[neighbor + 1];
        float deg_dst_inv_sqrt = rsqrt((float)(n_end - n_start));
        edge_vals_out[e] = deg_src_inv_sqrt * deg_dst_inv_sqrt;
    }
}
