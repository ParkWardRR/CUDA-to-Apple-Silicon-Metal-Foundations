# CUDA-to-Metal Translation Guide

This document is a practical reference for developers porting CUDA kernels to Metal Shading Language (MSL) on Apple Silicon. Every mapping listed here has been validated through real benchmark workloads.

## Thread Model

| CUDA | Metal | Notes |
|---|---|---|
| Thread | Thread | 1:1 mapping |
| Warp (32 threads) | SIMD-group (32 threads) | Identical width on Apple Silicon |
| Thread Block | Threadgroup | Same concept, different name |
| Grid | Grid | Dispatched via `dispatchThreads` or `dispatchThreadgroups` |

## Built-in Variables

| CUDA | Metal |
|---|---|
| `threadIdx.x` | `thread_position_in_threadgroup` |
| `blockIdx.x` | `threadgroup_position_in_grid` |
| `blockDim.x` | `threads_per_threadgroup` |
| `gridDim.x` | `threadgroups_per_grid` |
| `threadIdx.x + blockIdx.x * blockDim.x` | `thread_position_in_grid` (built-in, no math needed) |

## Memory Spaces

| CUDA | Metal | Notes |
|---|---|---|
| Global memory (`__device__`) | `device` address space | On Apple Silicon, UMA means no PCIe copy |
| Shared memory (`__shared__`) | `threadgroup` address space | Explicit allocation per threadgroup |
| Constant memory (`__constant__`) | `constant` address space | Read-only, cached |
| Local/Register | `thread` address space | Per-thread private |

## Synchronization

| CUDA | Metal |
|---|---|
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `__threadfence()` | `threadgroup_barrier(mem_flags::mem_device)` |
| `__syncwarp()` | `simdgroup_barrier(mem_flags::mem_none)` |

## Atomics

| CUDA | Metal |
|---|---|
| `atomicAdd(ptr, val)` | `atomic_fetch_add_explicit(ptr, val, memory_order_relaxed)` |
| `atomicMin(ptr, val)` | `atomic_fetch_min_explicit(ptr, val, memory_order_relaxed)` |
| `atomicMax(ptr, val)` | `atomic_fetch_max_explicit(ptr, val, memory_order_relaxed)` |
| `atomicCAS(ptr, cmp, val)` | `atomic_compare_exchange_weak_explicit(ptr, &cmp, val, ...)` |
| `atomicExch(ptr, val)` | `atomic_exchange_explicit(ptr, val, memory_order_relaxed)` |

**Important:** Metal requires explicit `atomic<T>` types for atomic operations. You cannot perform atomics on a bare `float*` -- you need `device atomic_float* ptr`.

## Warp Shuffle Primitives

| CUDA | Metal |
|---|---|
| `__shfl_sync(mask, val, lane)` | `simd_shuffle(val, lane)` |
| `__shfl_up_sync(mask, val, d)` | `simd_shuffle_up(val, d)` |
| `__shfl_down_sync(mask, val, d)` | `simd_shuffle_down(val, d)` |
| `__shfl_xor_sync(mask, val, m)` | `simd_shuffle_xor(val, m)` |
| `__ballot_sync(mask, pred)` | `simd_ballot(pred)` |
| `__activemask()` | `simd_active_threads_mask()` |

**Key difference:** Metal shuffle intrinsics do not take an explicit mask parameter. All 32 threads in the SIMD-group participate implicitly.

## Tensor Cores / Matrix Coprocessor

| CUDA (PTX) | Metal (MSL) | Notes |
|---|---|---|
| `wmma::load_matrix_sync` | `simdgroup_load(matrix, ptr, stride)` | Load tile from memory |
| `wmma::mma_sync` | `simdgroup_multiply_accumulate(acc, a, b, acc)` | Fused multiply-accumulate |
| `wmma::store_matrix_sync` | `simdgroup_store(matrix, ptr, stride)` | Store tile to memory |
| 16x16 tile (Volta/Turing) | 8x8 tile (Apple Silicon) | Apple AMX forces 8x8 for float |

## Kernel Launch

CUDA:
```cuda
myKernel<<<gridDim, blockDim, sharedMem, stream>>>(args...);
```

Metal (via Rust/metal-rs):
```rust
encoder.set_compute_pipeline_state(&pipeline);
encoder.set_buffer(0, Some(&buffer), 0);
encoder.dispatch_threads(
    MTLSize::new(total_threads_x, total_threads_y, 1),
    MTLSize::new(threadgroup_x, threadgroup_y, 1),
);
encoder.end_encoding();
command_buffer.commit();
command_buffer.wait_until_completed();
```

## Memory Allocation

CUDA:
```cuda
float *d_data;
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
// ... compute ...
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

Metal (Apple Silicon UMA -- no copies needed):
```rust
let buffer = device.new_buffer_with_data(
    data.as_ptr() as *const c_void,
    size as u64,
    MTLResourceOptions::StorageModeShared,  // CPU + GPU share same physical memory
);
// ... compute ...
// Read results directly from buffer.contents() -- no memcpy back
```

## Common Gotchas

1. **No `printf` in Metal shaders.** Use Metal GPU debugger or write debug values to a buffer.
2. **No dynamic parallelism.** Metal does not support launching kernels from within kernels. Use Indirect Command Buffers (ICB) for GPU-driven dispatch.
3. **No global `__syncthreads()` across threadgroups.** Use multiple dispatch passes or atomic signaling.
4. **Float atomics.** Metal supports `atomic_float` natively since Metal 3.0 (macOS 13). Earlier versions require CAS loops.
5. **Threadgroup memory size.** Apple Silicon has 32KB per threadgroup (vs 48-96KB on NVIDIA). Plan tile sizes accordingly.
