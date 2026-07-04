#include <metal_stdlib>
using namespace metal;

kernel void sv_hooking(
    device const uint* row_ptr [[buffer(0)]],
    device const uint* col_idx [[buffer(1)]],
    device atomic_uint* parent [[buffer(2)]],
    device atomic_uint* changed [[buffer(3)]],
    uint u [[thread_position_in_grid]],
    uint num_nodes [[threads_per_grid]]
) {
    if (u >= num_nodes) return;

    uint p_u = atomic_load_explicit(&parent[u], memory_order_relaxed);
    
    uint start = row_ptr[u];
    uint end = row_ptr[u + 1];

    for (uint i = start; i < end; i++) {
        uint v = col_idx[i];
        uint p_v = atomic_load_explicit(&parent[v], memory_order_relaxed);

        if (p_u < p_v) {
            uint old_p_v = atomic_fetch_min_explicit(&parent[p_v], p_u, memory_order_relaxed);
            if (old_p_v > p_u) {
                atomic_store_explicit(changed, 1, memory_order_relaxed);
            }
        } else if (p_v < p_u) {
            uint old_p_u = atomic_fetch_min_explicit(&parent[p_u], p_v, memory_order_relaxed);
            if (old_p_u > p_v) {
                atomic_store_explicit(changed, 1, memory_order_relaxed);
            }
        }
    }
}

kernel void sv_pointer_jumping(
    device atomic_uint* parent [[buffer(0)]],
    device atomic_uint* changed [[buffer(1)]],
    uint u [[thread_position_in_grid]],
    uint num_nodes [[threads_per_grid]]
) {
    if (u >= num_nodes) return;

    uint p_u = atomic_load_explicit(&parent[u], memory_order_relaxed);
    uint p_p_u = atomic_load_explicit(&parent[p_u], memory_order_relaxed);

    if (p_u != p_p_u) {
        atomic_store_explicit(&parent[u], p_p_u, memory_order_relaxed);
        atomic_store_explicit(changed, 1, memory_order_relaxed);
    }
}
