#include <metal_stdlib>
using namespace metal;

kernel void pagerank_scalar(
    device const uint* row_ptr [[buffer(0)]],
    device const uint* col_idx [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* r_old [[buffer(3)]],
    device float* r_new [[buffer(4)]],
    constant uint& num_nodes [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_nodes) return;
    
    uint start = row_ptr[tid];
    uint end = row_ptr[tid + 1];
    
    float sum = 0.0f;
    for (uint i = start; i < end; ++i) {
        sum += values[i] * r_old[col_idx[i]];
    }
    
    r_new[tid] = alpha * sum + (1.0f - alpha) / (float)num_nodes;
}

kernel void pagerank_vector(
    device const uint* row_ptr [[buffer(0)]],
    device const uint* col_idx [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const float* r_old [[buffer(3)]],
    device float* r_new [[buffer(4)]],
    constant uint& num_nodes [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_tid [[thread_position_in_grid]]
) {
    uint row = grid_tid / 32;
    uint lane_id = lid % 32;
    
    if (row >= num_nodes) return;
    
    uint start = row_ptr[row];
    uint end = row_ptr[row + 1];
    
    float sum = 0.0f;
    for (uint i = start + lane_id; i < end; i += 32) {
        sum += values[i] * r_old[col_idx[i]];
    }
    
    float row_sum = simd_sum(sum);
    
    if (lane_id == 0) {
        r_new[row] = alpha * row_sum + (1.0f - alpha) / (float)num_nodes;
    }
}
