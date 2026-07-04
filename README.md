# CUDA-to-Apple-Silicon-Metal-Foundations

A production-ready framework for hardware-accelerating massively parallel compute workloads on Apple Silicon (M-series) Macs natively via Metal.

If you have algorithms traditionally written for NVIDIA GPUs using CUDA -- graph traversals, sparse linear algebra, parallel scans, N-Body simulations, or GNNs -- this framework provides clean, zero-copy architecture to run those workflows on Apple Unified Memory at peak performance.

## Installation

```bash
git clone https://github.com/ParkWardRR/CUDA-to-Apple-Silicon-Metal-Foundations
cd CUDA-to-Apple-Silicon-Metal-Foundations
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy scipy networkx
maturin develop --release
```

## API Reference

```python
import c2m_core
```

### Graph Analytics
| Class | CUDA Equivalent | What it does |
|---|---|---|
| `MetalPageRank` | cuGraph PageRank | Iterative PageRank centrality via persistent-thread atomics |
| `MetalDeltaStepping` | Gunrock SSSP | Meyer & Sanders Delta-Stepping shortest paths |
| `MetalConnectedComponents` | Gunrock CC | Shiloach-Vishkin connected components |

### Sparse Linear Algebra
| Class | CUDA Equivalent | What it does |
|---|---|---|
| `MetalSpMV` | cuSPARSE SpMV | CSR Sparse Matrix-Vector multiplication (scalar + vector kernels) |

### Parallel Primitives
| Class | CUDA Equivalent | What it does |
|---|---|---|
| `MetalScanner` | CUB DeviceScan | Decoupled look-back prefix scan (f32/u32) + stream compaction |

### Physics & Simulation
| Class | CUDA Equivalent | What it does |
|---|---|---|
| `MetalNBody` | CUDA N-Body sample | O(N^2) tiled gravitational N-Body simulation |

### Graph Neural Networks
| Class | CUDA Equivalent | What it does |
|---|---|---|
| `MetalGNN` | DGL/PyG CUDA backend | GCN message passing with SIMD-optimized aggregation, ReLU, softmax |

### Apple Accelerate (AMX Coprocessor)
| Class | CUDA Equivalent | What it does |
|---|---|---|
| `AccelerateRunner` | cuBLAS SGEMM | Dense GEMM via Apple's AMX matrix coprocessor (Accelerate CBLAS) |

### Infrastructure
| Class | Purpose |
|---|---|
| `MetalRunner` | Metal device selection and command queue management |
| `Graph` | UMA zero-copy graph structure (`Node`, `Edge` buffers) |

## NetworkX Drop-In Replacement (`c2m_nx`)

For the fastest path from CPU to GPU, use `c2m_nx` as a transparent drop-in for NetworkX:

```python
import sys
sys.path.insert(0, 'python')
import c2m_nx as nx  # Drop-in replacement

G = nx.erdos_renyi_graph(10000, 0.01)
ranks = nx.pagerank(G)           # Runs on Metal GPU
dists = nx.shortest_path(G, 0)   # Runs on Metal GPU
comps = nx.connected_components(G)  # Runs on Metal GPU
```

## Architecture

```
Python  -->  PyO3/Rust  -->  metal-rs  -->  Metal Shading Language (MSL)
                |
                +-->  Apple Accelerate (AMX BLAS)
```

- **Zero-copy UMA:** All buffers use `MTLResourceStorageModeShared`. No PCIe transfers. CPU and GPU share physical memory.
- **Warp-to-SIMD mapping:** CUDA's 32-thread warps map directly to Metal's 32-thread SIMD-groups. `__shfl_up_sync` becomes `simd_shuffle_up`.
- **Persistent threads:** Graph frontier algorithms use persistent thread scheduling to avoid kernel relaunch overhead.

## Project Lineage

This framework is the production-ready extraction of [CUDA2Metal-Graph-Research](https://github.com/ParkWardRR/CUDA2Metal-Graph-Research), which validated the CUDA-to-Metal translation across 8 major benchmark suites (Gunrock, GARDENIA, cuGraph, CUTLASS, Rodinia, Parboil, gpu_bench, CUDA Samples).
