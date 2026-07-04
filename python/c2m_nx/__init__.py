"""
c2m_nx: A drop-in replacement for NetworkX that transparently hardware-accelerates
graph algorithms using Apple Silicon (Metal) via cuda2metal_graph.
"""

# Export NetworkX API by default so c2m_nx can act as a drop-in
from networkx import *

# Override specific algorithms with our hardware-accelerated versions
from .algorithms import pagerank, shortest_path, connected_components
