#include <metal_stdlib>
using namespace metal;

// Delta-Stepping SSSP Kernel
kernel void delta_stepping_relax(
    device const uint* indptr [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device atomic_uint* distances [[buffer(3)]],
    device const uint* current_bucket_nodes [[buffer(4)]],
    device atomic_uint* next_bucket_nodes [[buffer(5)]],
    device atomic_uint* next_bucket_size [[buffer(6)]],
    constant float& delta [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint num_nodes_in_bucket [[threads_per_grid]]
) {
    if (tid >= num_nodes_in_bucket) return;
    
    uint u = current_bucket_nodes[tid];
    uint d_u_bits = atomic_load_explicit(&distances[u], memory_order_relaxed);
    if (d_u_bits == 0x7f800000) return;
    
    float d_u = as_type<float>(d_u_bits);
    
    uint start = indptr[u];
    uint end = indptr[u + 1];
    
    for (uint e = start; e < end; ++e) {
        uint v = indices[e];
        float w = weights[e];
        float new_dist = d_u + w;
        
        uint old_val_uint = atomic_load_explicit(&distances[v], memory_order_relaxed);
        float old_val = as_type<float>(old_val_uint);
        
        while (new_dist < old_val) {
            uint desired = as_type<uint>(new_dist);
            if (atomic_compare_exchange_weak_explicit(&distances[v], &old_val_uint, desired, memory_order_relaxed, memory_order_relaxed)) {
                // If we successfully relaxed the edge, we need to categorize this node into a bucket.
                // In a full implementation, we'd have an array of buckets, but for simplicity we 
                // push it to a global 'next' queue which the CPU/driver will sort into buckets.
                uint idx = atomic_fetch_add_explicit(next_bucket_size, 1, memory_order_relaxed);
                atomic_store_explicit(&next_bucket_nodes[idx], v, memory_order_relaxed);
                break;
            }
            old_val = as_type<float>(old_val_uint);
        }
    }
}
