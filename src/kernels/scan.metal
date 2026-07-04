#include <metal_stdlib>
using namespace metal;

struct BlockState {
    atomic_uint status; // 0: Not ready, 1: Aggregate ready, 2: Prefix ready
    atomic_uint value;  // Store float as uint
};

template <typename T>
T from_uint(uint v);

template <>
inline float from_uint<float>(uint v) { return as_type<float>(v); }

template <>
inline uint from_uint<uint>(uint v) { return v; }

template <typename T>
uint to_uint(T v);

template <>
inline uint to_uint<float>(float v) { return as_type<uint>(v); }

template <>
inline uint to_uint<uint>(uint v) { return v; }

template <typename T>
inline void single_pass_scan_logic(
    device const T* input,
    device T* output,
    device atomic_uint* block_status,
    device atomic_uint* block_aggregate,
    device atomic_uint* block_prefix_val,
    device atomic_uint* global_counter,
    uint lid,
    uint block_size,
    uint n,
    threadgroup uint& shared_block_id,
    threadgroup T* shared_sums,
    threadgroup T& block_agg,
    threadgroup T& shared_exclusive_prefix,
    threadgroup uint& shared_status
) {
    if (lid == 0) {
        shared_block_id = atomic_fetch_add_explicit(global_counter, 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint block_id = shared_block_id;
    uint dynamic_tid = block_id * block_size + lid;
    
    T val = (dynamic_tid < n) ? input[dynamic_tid] : (T)0;
    
    T simd_sum = simd_prefix_exclusive_sum(val);
    T simd_total = simd_sum + val;
    
    uint simd_id = lid / 32;
    uint lane_id = lid % 32;
    
    if (lane_id == 31) {
        shared_sums[simd_id] = simd_total;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0) {
        T s_val = (lane_id < (block_size / 32)) ? shared_sums[lane_id] : (T)0;
        T s_scan = simd_prefix_exclusive_sum(s_val);
        shared_sums[lane_id] = s_scan;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    T local_prefix = simd_sum + shared_sums[simd_id];
    
    if (lid == block_size - 1) {
        block_agg = local_prefix + val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    T aggregate = block_agg;
    
    // 3. Decoupled Look-back phase (Convergent)
    if (block_id == 0) {
        if (lid == 0) {
            shared_exclusive_prefix = (T)0;
            atomic_store_explicit(&block_aggregate[block_id], to_uint<T>(aggregate), memory_order_relaxed);
            atomic_store_explicit(&block_prefix_val[block_id], to_uint<T>(aggregate), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lid == 0) {
            atomic_store_explicit(&block_status[block_id], 2, memory_order_relaxed);
        }
    } else {
        if (lid == 0) {
            shared_exclusive_prefix = (T)0;
            atomic_store_explicit(&block_aggregate[block_id], to_uint<T>(aggregate), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lid == 0) {
            atomic_store_explicit(&block_status[block_id], 1, memory_order_relaxed);
        }
        
        int look_back_id = block_id - 1;
        
        while (look_back_id >= 0) {
            if (lid == 0) {
                shared_status = atomic_load_explicit(&block_status[look_back_id], memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_device);
            uint status = shared_status;
            
            if (status == 2) {
                if (lid == 0) {
                    uint val_uint = atomic_load_explicit(&block_prefix_val[look_back_id], memory_order_relaxed);
                    shared_exclusive_prefix += from_uint<T>(val_uint);
                }
                break;
            } else if (status == 1) {
                if (lid == 0) {
                    uint val_uint = atomic_load_explicit(&block_aggregate[look_back_id], memory_order_relaxed);
                    shared_exclusive_prefix += from_uint<T>(val_uint);
                }
                look_back_id--;
            } else {
                // Spin-wait (all threads loop)
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (lid == 0) {
            T inclusive_prefix = shared_exclusive_prefix + aggregate;
            atomic_store_explicit(&block_prefix_val[block_id], to_uint<T>(inclusive_prefix), memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lid == 0) {
            atomic_store_explicit(&block_status[block_id], 2, memory_order_relaxed);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    T block_prefix = shared_exclusive_prefix;
    
    if (dynamic_tid < n) {
        output[dynamic_tid] = local_prefix + block_prefix;
    }
}

kernel void single_pass_scan_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device atomic_uint* block_status [[buffer(2)]],
    device atomic_uint* block_aggregate [[buffer(3)]],
    device atomic_uint* block_prefix_val [[buffer(4)]],
    device atomic_uint* global_counter [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]],
    constant uint& n [[buffer(6)]]
) {
    threadgroup uint shared_block_id;
    threadgroup float shared_sums[32];
    threadgroup float block_agg;
    threadgroup float shared_exclusive_prefix;
    threadgroup uint shared_status;
    
    single_pass_scan_logic<float>(input, output, block_status, block_aggregate, block_prefix_val, global_counter, lid, block_size, n,
                                  shared_block_id, shared_sums, block_agg, shared_exclusive_prefix, shared_status);
}

kernel void single_pass_scan_u32(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device atomic_uint* block_status [[buffer(2)]],
    device atomic_uint* block_aggregate [[buffer(3)]],
    device atomic_uint* block_prefix_val [[buffer(4)]],
    device atomic_uint* global_counter [[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]],
    constant uint& n [[buffer(6)]]
) {
    threadgroup uint shared_block_id;
    threadgroup uint shared_sums[32];
    threadgroup uint block_agg;
    threadgroup uint shared_exclusive_prefix;
    threadgroup uint shared_status;
    
    single_pass_scan_logic<uint>(input, output, block_status, block_aggregate, block_prefix_val, global_counter, lid, block_size, n,
                                 shared_block_id, shared_sums, block_agg, shared_exclusive_prefix, shared_status);
}

kernel void evaluate_predicate_f32(
    device const float* input [[buffer(0)]],
    device uint* predicate [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    constant uint& n [[buffer(2)]]
) {
    if (tid < n) {
        predicate[tid] = (input[tid] > 0.0f) ? 1 : 0;
    }
}

kernel void stream_compact_scatter(
    device const float* input [[buffer(0)]],
    device const uint* predicate [[buffer(1)]],
    device const uint* scan_indices [[buffer(2)]],
    device float* output [[buffer(3)]],
    device atomic_uint* global_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    constant uint& n [[buffer(5)]]
) {
    if (tid >= n) return;
    
    if (predicate[tid] > 0) {
        uint dest = scan_indices[tid];
        output[dest] = input[tid];
    }
    
    if (tid == n - 1) {
        atomic_store_explicit(global_count, scan_indices[tid] + predicate[tid], memory_order_relaxed);
    }
}
