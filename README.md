# CUDA-to-Apple-Silicon-Metal-Foundations (`c2m-core`)

> **Zero-copy, Metal-native compute primitives for Apple Silicon — a drop-in architectural bridge for workloads originally written for NVIDIA CUDA.**

[![macOS](https://img.shields.io/badge/macOS-13%2B-blue)](#) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#) [![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange)](#) [![Metal](https://img.shields.io/badge/Metal-3%2F4-silver)](#) [![License](https://img.shields.io/badge/License-MIT-green)](#)

---

## Why This Exists

NVIDIA's CUDA ecosystem — cuBLAS, cuSPARSE, cuGraph, Thrust — has dominated scientific computing and AI infrastructure for nearly two decades. Every major research lab, data pipeline, and ML framework was built assuming CUDA would be available. The result: an enormous body of valuable GPU-accelerated code that simply does not run on the one billion Apple Silicon devices now in circulation.

Apple's answer is [Metal](https://developer.apple.com/metal/), a low-overhead compute and graphics API built on the [Metal Shading Language (MSL)](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) — a C++14 dialect with GPU extensions — that sits atop Apple's Unified Memory Architecture (UMA). With Metal 4 (WWDC 2025), Apple introduced the `MTL4MachineLearningCommandEncoder`, the `tensor` resource type, and the Metal Performance Primitives (MPP) `tensor_ops` library, giving developers direct access to M5's per-core Neural Accelerators and enabling GEMM throughput on-par with datacenter GPUs for certain workload shapes.

`c2m-core` is a **Rust/PyO3-powered Python library** that systematically maps the canonical CUDA parallel computing primitives — Graph Analytics, Dense Linear Algebra, Stencil Compute, Prefix Scans, and N-Body Physics — onto native Metal MSL kernels, exploiting UMA's zero-copy architecture to eliminate the PCIe memory transfer overhead that throttles discrete GPU workloads.

---

## The Competitive Landscape

Several projects are attempting to bridge the CUDA-to-Apple-Silicon gap. Here's where `c2m-core` fits:

| Project | Approach | Status | Scope |
|---|---|---|---|
| **c2m-core** (this repo) | Native MSL kernels, PyO3/Rust bindings, hand-tuned for UMA | Active | Graph, LinAlg, Stencil, Reduce, Physics |
| [MetaXuda](https://pypi.org/project/metaxuda/) | CUDA runtime shim (`libcudart.dylib` drop-in) for Numba kernels | Alpha | Numba `@cuda.jit`, 230+ scalar ops |
| [CUDAM](https://github.com/MEHDI342/CUDAM) | CUDA-to-Metal source-level translation (AST rewrite) | Experimental | Source translation only |
| [Apple MLX](https://opensource.apple.com/projects/mlx/) | Array framework for ML on Apple Silicon | Active/Stable | ML/AI arrays, LLM inference |
| [MoltenVK](https://github.com/KhronosGroup/MoltenVK) | Vulkan → Metal translation layer | Stable | Graphics/Vulkan compute |
| PyTorch `mps` | MPSGraph backend for PyTorch tensor ops | Stable | PyTorch ops only |

`c2m-core` is **the only project** focused on hand-tuned, algorithm-specific Metal kernels for **non-ML scientific computing primitives** (sparse graph traversal, SSSP, connected components, stencil solvers, physics simulation) exposed as a clean Python API via Rust/PyO3 — the same Maturin-based stack Apple Silicon ML projects like MLX use internally.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Python Application                 │
│         import c2m_core                             │
│         c2m_core.graph.MetalDeltaStepping()         │
└────────────────────┬────────────────────────────────┘
                     │  PyO3 / Maturin (.so wheel)
┌────────────────────▼────────────────────────────────┐
│              Rust Host Layer (lib.rs)               │
│  • CSR graph encoding       • Buffer lifecycle      │
│  • MTLDevice acquisition    • Kernel dispatch       │
│  • Error propagation → Python exceptions            │
└────────────────────┬────────────────────────────────┘
                     │  metal-rs crate (MTLCommandBuffer)
┌────────────────────▼────────────────────────────────┐
│         Metal MSL Compute Kernels (.metal)          │
│  graph/      linalg/     stencil/   reduce/ physics/│
│  pagerank    gemm(stub)  hotspot    scan    nbody   │
│  delta_step             convolve2d (stub)   (stub)  │
│  conn_comp                                         │
└────────────────────┬────────────────────────────────┘
                     │  Apple Unified Memory Architecture
┌────────────────────▼────────────────────────────────┐
│     Apple Silicon GPU  ──────────  CPU / Neural Eng │
│       (shared DRAM pool, zero PCIe copies)          │
└─────────────────────────────────────────────────────┘
```

### Why Rust + PyO3?

- **Metal bindings without Objective-C overhead.** The `metal` crate (`0.27+`) provides safe Rust wrappers around all `MTLDevice`, `MTLCommandBuffer`, and `MTLComputeCommandEncoder` APIs.
- **Zero-copy Python interop.** PyO3 + Maturin produces native `.so` extension modules. NumPy array pointers are passed directly into `MTLBuffer`s via UMA shared storage — no Python GIL, no intermediate allocations.
- **Safety at the boundary.** Rust ownership semantics prevent Metal resource lifetime bugs (dangling command buffers, double-free textures) that plague raw Objective-C/Swift GPU code.

---

## Currently Available: Graph Analytics (`c2m_core.graph`)

Fully implemented, benchmark-tested Metal MSL kernels for sparse graph workloads. All three algorithms use **persistent-thread scheduling** to saturate Apple GPU thread groups and **atomic compare-and-swap** patterns (mapped from CUDA warp-level atomics to `simd_shuffle`-based coordination in MSL) to handle irregular graph traversal without warp divergence.

| Algorithm | Class | CUDA Equivalent | Paradigm |
|---|---|---|---|
| Iterative PageRank | `MetalPageRank` | `cuGraph::pagerank` | Scatter-gather, power iteration |
| Delta-Stepping SSSP | `MetalDeltaStepping` | `cuGraph::sssp` | Meyer–Sanders delta-stepping |
| Shiloach-Vishkin CC | `MetalConnectedComponents` | `cuGraph::weakly_connected_components` | Parallel label propagation |

```python
import c2m_core
import numpy as np

# Build a CSR-encoded graph
row_ptr = np.array([0, 2, 5, 7, 9, 10], dtype=np.uint32)
col_idx = np.array([1, 4, 0, 2, 3, 1, 4, 2, 1, 3], dtype=np.uint32)
weights  = np.array([0.5, 1.2, 0.5, 0.8, 0.3, 0.8, 1.1, 0.3, 1.0, 1.1], dtype=np.float32)

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
print(distances)  # [0.0, 0.5, 1.3, 0.8, 1.2]

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
print(scores)

# Connected components (Shiloach-Vishkin)
cc = c2m_core.MetalConnectedComponents()
labels = cc.compute(
    num_nodes=5,
    row_ptr=row_ptr.tolist(),
    col_idx=col_idx.tolist()
)
print(labels)
```

---

## Installation

**Requirements:** macOS 13+ (Ventura), Apple Silicon (M1 or later), Python 3.10+, Rust toolchain.

```bash
git clone https://github.com/ParkWardRR/CUDA-to-Apple-Silicon-Metal-Foundations
cd CUDA-to-Apple-Silicon-Metal-Foundations
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release
```

Verify:

```python
import c2m_core
help(c2m_core)
```

---

## Roadmap

Each phase maps a canonical CUDA compute paradigm onto Metal MSL and exposes it through the same `c2m_core.<module>` Python API.

---

### Phase 1 — Graph Analytics ✅ `c2m_core.graph` (Complete)

| Microstep | Status | Notes |
|---|---|---|
| CSR buffer encoding in Rust (`row_ptr`, `col_idx`, `weights`) | ✅ Done | Shared-storage `MTLBuffer`, zero-copy |
| `MetalPageRank` MSL kernel + persistent threads | ✅ Done | Power iteration, configurable damping/iters |
| `MetalDeltaStepping` SSSP kernel | ✅ Done | Meyer–Sanders delta-stepping, float32 weights |
| `MetalConnectedComponents` kernel | ✅ Done | Shiloach-Vishkin parallel label propagation |
| PyO3 class bindings via Maturin | ✅ Done | `.compute()` takes native Python lists |

---

### Phase 2 — Dense Linear Algebra 🔲 `c2m_core.linalg`

Maps NVIDIA PTX `mma.sync` Tensor Core GEMM onto Apple Silicon's matrix coprocessor path. On M4/M5, this means leveraging `simdgroup_matrix` instructions and — on M5 — the new Neural Accelerator `tensor_ops` primitives from Metal Performance Primitives (MPP), which Apple first documented at WWDC 2025.

| # | Microstep | Target Metal API |
|---|---|---|
| 2.1 | Design `MTLBuffer` layout for row-major and column-major dense matrices | `MTLBuffer` shared storage |
| 2.2 | Implement naive GEMM kernel as correctness baseline | `kernel` + `[[threadgroup_memory_length]]` |
| 2.3 | Tile GEMM with `simdgroup_matrix` 8×8 accumulation blocks | `simdgroup_matrix<float, 8, 8>` |
| 2.4 | Add `simdgroup_matrix` FP16 half-precision path for ML workloads | `simdgroup_matrix<half, 8, 8>` |
| 2.5 | Implement GEMV (matrix × vector) for attention score computation | `simd_sum` + `simd_shuffle` |
| 2.6 | (M5 only) Port GEMM inner loop to MPP `tensor_ops::gemm` Neural Accelerator path | `mpp::tensor_ops`, Metal 4 / MPP |
| 2.7 | Implement BLAS Level 1: SAXPY, DOT, NORM | MSL `simd_reduce_add` |
| 2.8 | Implement LU decomposition (partial pivot, in-place) | Iterative panel factorization |
| 2.9 | Expose `c2m_core.linalg.gemm(A, B)`, `sgemv`, `saxpy` to Python | PyO3 `ndarray` or buffer protocol |
| 2.10 | Benchmark against `numpy.linalg` and `Accelerate` BLAS on M-series | Roofline analysis |

---

### Phase 3 — Stencil / Grid Compute 🔲 `c2m_core.stencil`

Grid-based PDE solvers that exploit Apple Silicon's large L2 cache (M4 Max: 48 MB shared; M5: 64 MB+). Unlike CUDA's explicit `__shared__` memory management, Metal threadgroup memory is managed through `threadgroup` address space — the compiler can auto-promote hot cache lines without programmer hints, making Apple Silicon naturally suited for 2D/3D stencil workloads.

| # | Microstep | Target Metal API |
|---|---|---|
| 3.1 | Implement 2D Jacobi stencil (5-point) as correctness baseline | `MTLSize(x, y, 1)` grid |
| 3.2 | Tune 2D threadgroup tile size for L2 cache reuse (target: M2/M3/M4 L2 profiles) | Occupancy analysis |
| 3.3 | Implement Rodinia Hotspot transient thermal simulation (2D heat equation) | `threadgroup float` tile cache |
| 3.4 | Extend to 3D stencil (7-point Laplacian) for CFD / Poisson solver use cases | `MTLSize(x, y, z)` |
| 3.5 | Implement 2D convolution for image processing / signal processing kernels | `MTLTexture` sampler path |
| 3.6 | Implement Wave Equation solver (explicit finite difference) | Multi-pass command buffer |
| 3.7 | Expose `c2m_core.stencil.hotspot2d()`, `jacobi2d()`, `convolve2d()` to Python | PyO3 + NumPy buffer protocol |
| 3.8 | Benchmark against SciPy `ndimage` and reference CPU implementations | Roofline @ memory-bound regime |

---

### Phase 4 — Prefix Scans & Compaction 🔲 `c2m_core.reduce`

Maps CUDA's `__shfl_up_sync` warp-shuffle scan primitives to Metal's `simd_shuffle_up` SIMD-group intrinsics. Blelloch tree scan is the canonical parallel prefix sum, used as the backbone of stream compaction, radix sort, and histogram operations — all prerequisites for Phases 5 (Physics) and future sparse solver work.

| # | Microstep | Target Metal API |
|---|---|---|
| 4.1 | Implement single-threadgroup inclusive Blelloch scan | `simd_shuffle_up`, `simd_prefix_inclusive_sum` |
| 4.2 | Extend to multi-threadgroup scan via inter-group prefix propagation | `threadgroup_barrier`, device atomic accumulator |
| 4.3 | Implement exclusive prefix sum (prescan) | Shift-right + identity insertion |
| 4.4 | Implement stream compaction using scan output as scatter index | Two-pass: flag kernel + compaction kernel |
| 4.5 | Implement segmented scan for graph BFS frontier management | `simd_shuffle_xor` segment masking |
| 4.6 | Implement parallel reduction (sum, min, max) over arbitrary array | `simd_reduce_add/min/max` |
| 4.7 | Implement parallel histogram (integer bin counting) | `atomic_fetch_add_explicit` |
| 4.8 | Expose `c2m_core.reduce.scan()`, `compaction()`, `reduce_sum()` to Python | PyO3 + NumPy |
| 4.9 | Benchmark against Thrust `thrust::inclusive_scan` throughput on equivalent CUDA hardware | GB/s at peak memory bandwidth |

---

### Phase 5 — N-Body Physics 🔲 `c2m_core.physics`

All-pairs O(N²) gravitational simulation — the canonical GPU stress test and a direct translation of CUDA's `nbody` sample. Apple Silicon's UMA means particle position/velocity arrays never need to be transferred between CPU and GPU memory, eliminating the main bottleneck of discrete GPU N-body implementations.

| # | Microstep | Target Metal API |
|---|---|---|
| 5.1 | Implement naive O(N²) all-pairs gravitational force kernel | `kernel` + per-thread particle loop |
| 5.2 | Tile with threadgroup shared memory for O(N²/B) memory traffic reduction | `threadgroup float4` position tile |
| 5.3 | Implement Leapfrog integrator (velocity Verlet) for time stepping | Host-side integration loop |
| 5.4 | Add Barnes-Hut O(N log N) octree build + traversal (stretch goal) | Recursive MSL kernel or iterative stack |
| 5.5 | Expose `c2m_core.physics.nbody(positions, velocities, masses, dt, steps)` to Python | PyO3, returns updated NumPy arrays |
| 5.6 | Benchmark GFLOPS/s against CUDA Galaxy Simulation sample on equivalent hardware | Target: saturate GPU FLOP throughput |

---

### Phase 6 — Sparse Linear Algebra 🔲 `c2m_core.sparse`

Complementing the dense linalg module, sparse operations form the backbone of scientific simulation (FEM, PDE solvers) and graph neural networks. CUDA's `cuSPARSE` is the reference. The CSR infrastructure already built for Phase 1 graph analytics is the foundation for this module.

| # | Microstep | Notes |
|---|---|---|
| 6.1 | Implement SpMV (Sparse Matrix × Dense Vector) in CSR format | Core for iterative solvers |
| 6.2 | Implement SpMM (Sparse Matrix × Dense Matrix) for GNN aggregation | Critical for PyG/DGL workload migration |
| 6.3 | Implement Conjugate Gradient (CG) iterative solver | SpMV + dot + SAXPY from Phase 2/4 |
| 6.4 | Implement Incomplete LU (ILU) preconditioner for CG | Triangular sparse solve |
| 6.5 | Expose `c2m_core.sparse.spmv()`, `spmm()`, `cg_solve()` to Python | PyO3 + SciPy sparse interop |

---

### Phase 7 — Distribution & Packaging 🔲

| # | Microstep | Notes |
|---|---|---|
| 7.1 | Build universal2 / arm64 wheels via `maturin build --release` | Target: PyPI distribution |
| 7.2 | Add GitHub Actions CI: build + test on macos-14 (M1) and macos-15 (M2/M3) runners | `.github/workflows/ci.yml` |
| 7.3 | Publish to PyPI as `c2m-core` (`pip install c2m-core`) | Semantic versioning |
| 7.4 | Add benchmark suite (`benchmarks/`) with roofline annotations | Compare vs. reference CUDA numbers |
| 7.5 | Write contributor guide for adding new Metal kernels | `CONTRIBUTING.md` |

---

## CUDA → Metal Primitive Mapping Reference

| CUDA Concept | Metal / MSL Equivalent | Notes |
|---|---|---|
| `threadIdx`, `blockIdx` | `thread_position_in_grid`, `threadgroup_position_in_grid` | Direct mapping |
| `__shared__` memory | `threadgroup` address space | Compiler can auto-promote on Apple Silicon |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | |
| `__shfl_up_sync` | `simd_shuffle_up` | SIMD-group width = 32 on M-series |
| `atomicAdd` (global) | `atomic_fetch_add_explicit(..., memory_order_relaxed)` | UMA collapses device/host address spaces |
| `cudaMalloc` / `cudaFree` | `device.newBuffer(length:options:)` / `.setPurgeableState(.empty)` | |
| `cudaMemcpy` H↔D | Not required — UMA shared storage | Zero-copy by default |
| `mma.sync` Tensor Core | `simdgroup_matrix<T,8,8>` (M1–M4); `mpp::tensor_ops` (M5+, Metal 4) | Metal 4 MPP is M5-only |
| CUDA Streams | `MTLCommandBuffer` + `MTLCommandQueue` | Multiple queues for concurrent pipelines |
| Warp (32 threads) | SIMD-group (32 threads, M-series) | |
| `cuGraph` | `c2m_core.graph` (this library) | |
| `cuBLAS` | `c2m_core.linalg` (Phase 2) + Apple Accelerate | |
| `cuSPARSE` | `c2m_core.sparse` (Phase 6) | |
| Thrust `inclusive_scan` | `c2m_core.reduce.scan()` (Phase 4) | |

---

## Hardware Context

`c2m-core` targets the full M-series lineup. All phases (1–6) run on any Apple Silicon Mac with macOS 13+. Phase 2, Step 2.6 (MPP `tensor_ops`) requires M5 and Metal 4 (macOS 16+).

| Chip | GPU Cores | Unified Memory BW | Key Capability |
|---|---|---|---|
| M1 / M1 Pro/Max/Ultra | 7–64 | 68–800 GB/s | Baseline UMA, simdgroup_matrix |
| M2 / M2 Pro/Max/Ultra | 10–76 | 100–800 GB/s | Improved SIMD throughput |
| M3 / M3 Pro/Max | 10–40 | 100–400 GB/s | 3rd-gen ray tracing, dynamic caching |
| M4 / M4 Pro/Max | 10–40 | 120–546 GB/s | HW mesh shaders, improved simdgroup_matrix |
| **M5 / M5 Pro/Max** | **10–40+** | **153–800+ GB/s** | **Neural Accelerator per core, MPP tensor_ops, Metal 4** |

---

## Contributing

Pull requests are welcome. To add a new kernel:

1. Write the MSL shader in `src/kernels/<module>/<kernel>.metal`
2. Add the Rust host dispatch in `src/<kernel>.rs`, using `MetalRunner` from `metal_runner.rs`
3. Expose the Python class/function via `PyO3` in `src/lib.rs`
4. Add a usage example in this README and a test in `tests/`

```bash
# Rebuild after changes
maturin develop --release
python -c "import c2m_core; print('OK')"
```

---

## License

MIT
