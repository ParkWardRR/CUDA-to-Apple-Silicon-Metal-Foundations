#include <metal_stdlib>
using namespace metal;

// Scalar kernel: One thread per row
// Good for matrices with low average degree (e.g. PCB routing graphs)
kernel void spmv_csr_scalar(
    device const float* values [[buffer(0)]],
    device const uint* col_indices [[buffer(1)]],
    device const uint* row_ptr [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    uint row [[thread_position_in_grid]],
    constant uint& num_rows [[buffer(5)]]
) {
    if (row >= num_rows) return;
    
    uint start = row_ptr[row];
    uint end = row_ptr[row + 1];
    
    float sum = 0.0f;
    for (uint i = start; i < end; ++i) {
        sum += values[i] * x[col_indices[i]];
    }
    
    y[row] = sum;
}

// Vector kernel: One SIMD group (32 threads) per row
// Good for matrices with high average degree
kernel void spmv_csr_vector(
    device const float* values [[buffer(0)]],
    device const uint* col_indices [[buffer(1)]],
    device const uint* row_ptr [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_tid [[thread_position_in_grid]],
    constant uint& num_rows [[buffer(5)]]
) {
    uint row = grid_tid / 32;
    uint lane_id = lid % 32;
    
    if (row >= num_rows) return;
    
    uint start = row_ptr[row];
    uint end = row_ptr[row + 1];
    
    float sum = 0.0f;
    
    for (uint i = start + lane_id; i < end; i += 32) {
        sum += values[i] * x[col_indices[i]];
    }
    
    // Reduce within the SIMD group
    float row_sum = simd_sum(sum);
    
    // First lane writes the result
    if (lane_id == 0) {
        y[row] = row_sum;
    }
}
