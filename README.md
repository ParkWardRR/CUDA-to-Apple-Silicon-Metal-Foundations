# CUDA-to-Apple-Silicon-Metal-Foundations (`c2m-core`)

Welcome to `c2m-core`! This is the foundational library for hardware-accelerating massively parallel compute workloads on Apple Silicon (M-series) Macs natively via Metal.

If you are developing algorithms that were traditionally written for NVIDIA GPUs using CUDA (Graphs, Dense Tensors, N-Body simulations, Stencils), this framework provides the clean, zero-copy architecture required to run those workflows on Apple Unified Memory Architecture (UMA) at peak performance.

## 🚀 Currently Available Architectures

### Graph Analytics (`c2m_core.graph`)
The core module currently houses fully optimized, benchmark-tested algorithms for graph analytics. We use highly concurrent persistent-thread scheduling and warp-level atomics to traverse graphs containing millions of edges in milliseconds.
* `MetalPageRank`: Offload iterative PageRank centrality.
* `MetalDeltaStepping`: Offload Single-Source Shortest Path (SSSP) via Meyer & Sanders Delta-Stepping algorithm.
* `MetalConnectedComponents`: Offload Shiloach-Vishkin connected components.

```python
import c2m_core
import numpy as np

# Example: Compute SSSP
sssp_solver = c2m_core.graph.MetalDeltaStepping()
distances = sssp_solver.compute(
    num_nodes=5000,
    row_ptr=np.array([...], dtype=np.uint32).tolist(),
    col_idx=np.array([...], dtype=np.uint32).tolist(),
    weights=np.array([...], dtype=np.float32).tolist(),
    source_node=0,
    delta=2.5
)
print(distances)
```

## 🚧 Upcoming Paradigms (Architectural Stubs)

We are actively expanding this framework to abstract other CUDA workloads into Metal MSL. We have left architectural stubs in the code to demonstrate where these are heading:

- **`c2m_core.linalg`**: Dense Linear Algebra. Coming soon! Will map NVIDIA PTX `mma.sync` Tensor Core algorithms directly to Apple AMX `simdgroup_matrix` coprocessor primitives.
- **`c2m_core.stencil`**: Grid compute. Coming soon! Will leverage Apple's huge L2 caches to solve 2D spatial locality problems (like Transient Thermal simulation) without explicit Shared Memory management.
- **`c2m_core.reduce`**: Scans & Compactions. Coming soon! Will map CUDA's `__shfl_up_sync` primitives into Metal's `simd_shuffle_up` to achieve blistering prefix sum rates.

## Installation
Ensure you have the Rust toolchain installed, then:
```bash
git clone https://github.com/ParkWardRR/CUDA-to-Apple-Silicon-Metal-Foundations
cd CUDA-to-Apple-Silicon-Metal-Foundations
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```
