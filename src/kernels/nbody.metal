#include <metal_stdlib>
using namespace metal;

kernel void nbody_integrate_tiled(
    device const float4* p_old [[buffer(0)]],
    device float4* p_new [[buffer(1)]],
    device float4* v [[buffer(2)]],
    constant uint& num_bodies [[buffer(3)]],
    constant float& dt [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    if (tid >= num_bodies) return;
    
    float4 my_pos = p_old[tid];
    float3 accel = float3(0.0f);
    
    // Tiled shared memory for positions
    threadgroup float4 shared_pos[256];
    
    for (uint i = 0; i < num_bodies; i += block_size) {
        if (i + lid < num_bodies) {
            shared_pos[lid] = p_old[i + lid];
        } else {
            shared_pos[lid] = float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Loop over the tile in shared memory
        uint num_elements = min(block_size, num_bodies - i);
        for (uint j = 0; j < num_elements; ++j) {
            float4 other_pos = shared_pos[j];
            float3 r = other_pos.xyz - my_pos.xyz;
            float dist_sq = dot(r, r) + eps;
            float inv_dist = rsqrt(dist_sq);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            
            // accumulate acceleration (G is absorbed into other_pos.w)
            accel += r * other_pos.w * inv_dist3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Semi-implicit Euler integration
    float4 my_v = v[tid];
    my_v.xyz += accel * dt;
    v[tid] = my_v;
    
    float4 out_pos = my_pos;
    out_pos.xyz += my_v.xyz * dt;
    p_new[tid] = out_pos;
}
