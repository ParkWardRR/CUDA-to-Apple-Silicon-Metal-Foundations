# CUDA-to-Apple-Silicon-Metal-Foundations

> Run your CUDA workloads natively on Apple Silicon. No NVIDIA GPU required.

A production-ready Python/Rust framework for hardware-accelerating massively parallel compute workloads on Apple Silicon (M1/M2/M3/M4/M5) Macs via Metal compute shaders. This library is organized around **Compute Patterns** -- each pattern represents a canonical CUDA programming paradigm, ported to Metal MSL and exposed through a unified `c2m_core` Python API.

---

## Compute Patterns

CUDA programming isn't one thing -- it's a collection of distinct **compute patterns**, each with its own data structures, memory access strategies, and parallelism model. A graph traversal looks nothing like a dense matrix multiply, which looks nothing like a prefix scan. NVIDIA ships separate libraries for each pattern (`cuGraph` for graphs, `cuBLAS` for dense math, `cuSPARSE` for sparse math, `CUB` for primitives), and porting CUDA code to another platform means understanding which pattern your code uses and translating the right set of idioms.

This framework is organized around these patterns. Each one maps a canonical CUDA programming paradigm onto Metal MSL, handles the hardware differences (warp width, shared memory size, atomic semantics), and exposes the result through a single `import c2m_core` Python API. Patterns marked **Done** are fully implemented and benchmark-validated. Patterns marked **Planned** have a clear architectural path and will be added in future releases.

The **CUDA Ecosystem** column shows which NVIDIA library or framework each pattern replaces, so you can quickly find the Metal equivalent of whatever CUDA tool you are currently using.

| # | Pattern | CUDA Ecosystem | Status | Classes |
|---|---|---|---|---|
| 1 | **Graph Analytics** | cuGraph, Gunrock | Done | `MetalPageRank`, `MetalDeltaStepping`, `MetalConnectedComponents` |
| 2 | **Sparse Linear Algebra** | cuSPARSE | Done | `MetalSpMV` |
| 3 | **Parallel Primitives** | CUB / Thrust | Done | `MetalScanner` (prefix scan, stream compaction) |
| 4 | **N-Body Physics** | CUDA Samples | Done | `MetalNBody` |
| 5 | **Graph Neural Networks** | DGL / PyG | Done | `MetalGNN` (GCN message passing, softmax) |
| 6 | **Dense Linear Algebra (CPU)** | cuBLAS | Done | `AccelerateRunner` (AMX SGEMM via Accelerate) |
| 7 | **Dense Linear Algebra (GPU)** | CUTLASS | Planned | Tiled GEMM via `simdgroup_matrix` 8x8 AMX tiles |
| 8 | **Stencil / Grid Compute** | Rodinia, Parboil | Planned | 2D/3D stencils, heat equation, convolution |
| 9 | **Sparse Solvers** | cuSPARSE, AmgX | Planned | SpMM, Conjugate Gradient, ILU preconditioner |
| 10 | **Sort & Histogram** | CUB / Thrust | Planned | Radix sort, parallel histogram |
| 11 | **FFT** | cuFFT | Planned | 1D/2D FFT via Metal or Accelerate vDSP |
| 12 | **Source-to-Source Transpiler** | -- | Planned | Automatic CUDA kernel to MSL translation |

---

### Why Rust + PyO3?

- **Metal bindings without Objective-C overhead.** The `metal` crate (`0.27+`) provides safe Rust wrappers around all `MTLDevice`, `MTLCommandBuffer`, and `MTLComputeCommandEncoder` APIs.
- **Zero-copy Python interop.** PyO3 + Maturin produces native `.so` extension modules. NumPy array pointers are passed directly into `MTLBuffer`s via UMA shared storage -- no Python GIL, no intermediate allocations.
- **Safety at the boundary.** Rust ownership semantics prevent Metal resource lifetime bugs (dangling command buffers, double-free textures) that plague raw Objective-C/Swift GPU code.

---

## Quick Start

### Requirements
- macOS 13+ (Ventura) on Apple Silicon (M1 or later)
- Python 3.9+
- Rust toolchain ([rustup.rs](https://rustup.rs))

### Installation

```bash
git clone https://github.com/ParkWardRR/CUDA-to-Apple-Silicon-Metal-Foundations.git
cd CUDA-to-Apple-Silicon-Metal-Foundations
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy scipy networkx
maturin develop --release
```

Verify:

```python
import c2m_core
help(c2m_core)
```

### Hello World: GPU-Accelerated Graph Analytics

```python
import c2m_core
import numpy as np

# Build a CSR-encoded graph
row_ptr = np.array([0, 2, 4, 6, 8, 10], dtype=np.uint32)
col_idx = np.array([1, 3, 0, 2, 1, 4, 0, 4, 2, 3], dtype=np.uint32)
weights = np.array([0.5, 1.2, 0.5, 0.8, 0.3, 0.8, 1.1, 0.3, 1.0, 1.1], dtype=np.float32)

# Single-Source Shortest Path (Delta-Stepping)
sssp = c2m_core.MetalDeltaStepping()
distances = sssp.compute(
    num_nodes=5,
    row_ptr=row_ptr.tolist(),
    col_idx=col_idx.tolist(),
    weights=weights.tolist(),
    source_node=0,
    delta=2.5
)
print(distances)

# PageRank centrality
pr = c2m_core.MetalPageRank()
scores = pr.compute(
    num_nodes=5,
    row_ptr=row_ptr.tolist(),
    col_idx=col_idx.tolist(),
    weights=weights.tolist(),
    damping=0.85,
    iterations=50
)

# Connected components (Shiloach-Vishkin)
cc = c2m_core.MetalConnectedComponents()
labels = cc.compute(
    num_nodes=5,
    row_ptr=row_ptr.tolist(),
    col_idx=col_idx.tolist()
)
```

Or even simpler -- use the **NetworkX drop-in**:

```python
import sys; sys.path.insert(0, 'python')
import c2m_nx as nx

G = nx.erdos_renyi_graph(10000, 0.01)
ranks = nx.pagerank(G)              # Transparently runs on Metal GPU
dists = nx.shortest_path(G, 0)      # Transparently runs on Metal GPU
comps = list(nx.connected_components(G))  # Transparently runs on Metal GPU
```

`c2m_nx` re-exports the entire NetworkX API. Functions without GPU implementations fall through to the standard CPU path.

---

## Full API Reference

```python
import c2m_core
```

### Graph Analytics

| Class | CUDA Equivalent | Paradigm | Description |
|---|---|---|---|
| `MetalPageRank` | cuGraph PageRank | Scatter-gather, power iteration | Iterative PageRank centrality via persistent-thread atomics. Scalar and SIMD-vectorized kernels. |
| `MetalDeltaStepping` | Gunrock SSSP | Meyer-Sanders delta-stepping | Single-source shortest paths via frontier-centric wavefront expansion. |
| `MetalConnectedComponents` | Gunrock CC | Parallel label propagation | Shiloach-Vishkin connected components with pointer-jumping convergence. |

### Sparse Linear Algebra

| Class | CUDA Equivalent | Description |
|---|---|---|
| `MetalSpMV` | cuSPARSE csrmv | CSR Sparse Matrix-Vector multiplication. Includes both scalar (1 thread/row) and vector (1 SIMD-group/row) kernels. |

### Parallel Primitives

| Class | CUDA Equivalent | Description |
|---|---|---|
| `MetalScanner` | CUB DeviceScan | Decoupled look-back prefix scan for f32 and u32. Includes predicate evaluation and stream compaction. Maps `__shfl_up_sync` to `simd_shuffle_up`. |

### Physics & Simulation

| Class | CUDA Equivalent | Description |
|---|---|---|
| `MetalNBody` | CUDA N-Body Sample | O(N^2) tiled gravitational N-Body simulation. Uses threadgroup shared memory for tile caching. Achieves 1+ TFLOPS on M4. |

### Graph Neural Networks

| Class | CUDA Equivalent | Description |
|---|---|---|
| `MetalGNN` | DGL / PyG CUDA backend | Full GCN forward pass on GPU: sparse message passing (with SIMD-optimized aggregation), linear transform + ReLU, and softmax. Single command buffer submission per forward pass. |

### Dense Linear Algebra (Apple Accelerate / AMX)

| Class | CUDA Equivalent | Description |
|---|---|---|
| `AccelerateRunner` | cuBLAS sgemm | Dense SGEMM via Apple's AMX matrix coprocessor through the Accelerate CBLAS FFI. Hardware-accelerated for large matrices. |

### Infrastructure

| Class | Purpose |
|---|---|
| `MetalRunner` | Metal device selection, command queue management, and `NSError` propagation to Python exceptions. |
| `Graph` | Zero-copy UMA graph structure with `Node` and `Edge` buffers ready for Metal dispatch. |

---

## Architecture

```
Python (c2m_nx / c2m_core)
    |
    v
PyO3 (Rust <-> Python FFI)
    |
    +---> metal-rs ---> Metal Compute Shaders (MSL)
    |                      |
    |                      +---> Apple GPU (M1/M2/M3/M4/M5)
    |
    +---> Apple Accelerate Framework (CBLAS)
                               |
                               +---> Apple AMX Coprocessor
```

### Key Design Decisions

- **Zero-copy UMA.** All Metal buffers use `MTLResourceStorageModeShared`. CPU and GPU share physical memory. No PCIe bus transfers. This is the single biggest architectural advantage over discrete NVIDIA GPUs for small-to-medium workloads.
- **Warp-to-SIMD mapping.** CUDA's 32-thread warps map 1:1 to Metal's 32-thread SIMD-groups. `__shfl_up_sync` becomes `simd_shuffle_up`. `__syncthreads()` becomes `threadgroup_barrier(mem_flags::mem_threadgroup)`.
- **Persistent threads.** Graph frontier algorithms (BFS, SSSP) use persistent thread scheduling with atomic work-stealing to avoid the overhead of relaunching kernels per frontier level.
- **No PyTorch dependency.** The entire GPU stack is: Python -> PyO3 -> Rust -> metal-rs -> MSL. Lightweight, auditable, zero bloat.

---

## CUDA-to-Metal Translation Quick Reference

| CUDA Concept | Metal Equivalent | Notes |
|---|---|---|
| `__global__ void kernel(...)` | `kernel void kernel(...)` | |
| `threadIdx.x`, `blockIdx.x` | `thread_position_in_threadgroup`, `threadgroup_position_in_grid` | Metal also provides `thread_position_in_grid` directly |
| `__shared__` memory | `threadgroup` address space | Compiler can auto-promote on Apple Silicon |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | |
| `atomicAdd(ptr, val)` | `atomic_fetch_add_explicit(ptr, val, memory_order_relaxed)` | UMA collapses device/host address spaces |
| `__shfl_up_sync(mask, val, d)` | `simd_shuffle_up(val, d)` | No explicit mask needed in Metal |
| Warp (32 threads) | SIMD-group (32 threads) | Identical width on Apple Silicon |
| `cudaMalloc` / `cudaMemcpy` | `device.new_buffer(StorageModeShared)` | Zero-copy by default on UMA |
| cuBLAS `sgemm` | Accelerate `cblas_sgemm` | AMX hardware offload |
| PTX `mma.sync` (Tensor Cores) | `simdgroup_multiply_accumulate` | 8x8 tiles on Apple Silicon (vs 16x16 on NVIDIA) |
| CUDA Streams | `MTLCommandBuffer` + `MTLCommandQueue` | Multiple queues for concurrent pipelines |
| `cuGraph` | `c2m_core` graph classes | This library |

For the full translation guide with memory spaces, atomics, and gotchas, see [docs/TRANSLATION_GUIDE.md](docs/TRANSLATION_GUIDE.md).

---

## Hardware Compatibility

`c2m-core` targets the full M-series lineup. All modules run on any Apple Silicon Mac with macOS 13+.

| Chip | GPU Cores | Unified Memory BW | Key Capability |
|---|---|---|---|
| M1 / M1 Pro/Max/Ultra | 7-64 | 68-800 GB/s | Baseline UMA, simdgroup_matrix |
| M2 / M2 Pro/Max/Ultra | 10-76 | 100-800 GB/s | Improved SIMD throughput |
| M3 / M3 Pro/Max | 10-40 | 100-400 GB/s | Dynamic caching, 3rd-gen ray tracing |
| M4 / M4 Pro/Max | 10-40 | 120-546 GB/s | HW mesh shaders, improved simdgroup_matrix |
| M5 / M5 Pro/Max | 10-40+ | 153-800+ GB/s | Neural Accelerator per core, MPP tensor_ops, Metal 4 |

---

## Validated Benchmark Lineage

All implementations in this framework were validated against CPU reference implementations (NumPy, SciPy, NetworkX) and established CUDA benchmark suites during an extensive internal research phase covering 8 major suites:

| Suite | Status | What was validated |
|---|---|---|
| Gunrock | Validated | Frontier-centric BFS, SSSP, PageRank |
| GARDENIA | Validated | Scale-free, high-diameter, and dense topologies |
| cuGraph | Validated | Zero-copy Polars DataFrame ingestion to CSR |
| CUTLASS | Validated | PTX `mma.sync` to `simdgroup_matrix` AMX tile mapping |
| CUDA Samples | Validated | vectorAdd, reduction, shfl_scan primitives |
| gpu_bench | Validated | SpMV regression gauntlet |
| Rodinia | Validated | 2D Hotspot stencil compute |
| Parboil | Validated | MRI-Q trigonometric compute (38x speedup over NumPy) |

---

## Contributing

Pull requests are welcome. To add a new kernel:

1. Write the MSL shader in `src/kernels/<kernel>.metal`
2. Add the Rust host dispatch in `src/<kernel>.rs`, using `map_metal_err` from `metal_runner.rs`
3. Expose the Python class via PyO3 in `src/lib.rs`
4. Add a usage example in this README and a test in `tests/`

```bash
# Rebuild after changes
maturin develop --release
python -c "import c2m_core; print('OK')"
```

---

## License

[Blue Oak Model License 1.0.0](https://blueoakcouncil.org/license/1.0.0)
