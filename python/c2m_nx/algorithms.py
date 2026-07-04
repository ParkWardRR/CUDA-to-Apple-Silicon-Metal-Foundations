import networkx as nx
import numpy as np
from .conversion import get_csr
from c2m_core import MetalPageRank, MetalDeltaStepping, MetalConnectedComponents

def pagerank(G, alpha=0.85, max_iter=100, tol=1.0e-6, weight='weight'):
    """
    Drop-in for networkx.pagerank using MetalPageRank.
    """
    num_nodes = G.number_of_nodes()
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # We need CSR of the TRANSPOSE graph, with values 1.0 / out_degree
    # First get out-degrees
    out_degree = {u: 0.0 for u in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = d.get(weight, 1.0)
        out_degree[u] += w
        if not G.is_directed():
            out_degree[v] += w
            
    # Now build transposed CSR
    # In transposed CSR, row_ptr[v] points to incoming edges to v
    incoming_edges = {u: [] for u in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = d.get(weight, 1.0)
        if out_degree[u] > 0:
            incoming_edges[v].append((u, w / out_degree[u]))
        if not G.is_directed() and out_degree[v] > 0:
            incoming_edges[u].append((v, w / out_degree[v]))
            
    row_ptr = np.zeros(num_nodes + 1, dtype=np.uint32)
    col_idx = []
    weights = []
    
    current_edge_idx = 0
    for i, u in enumerate(node_list):
        row_ptr[i] = current_edge_idx
        for v, w in incoming_edges[u]:
            col_idx.append(node_to_idx[v])
            weights.append(w)
            current_edge_idx += 1
    row_ptr[num_nodes] = current_edge_idx
    
    col_idx = np.array(col_idx, dtype=np.uint32)
    weights = np.array(weights, dtype=np.float32)
    
    pr = MetalPageRank()
    
    # Metal backend call
    ranks_array = pr.compute(weights, col_idx, row_ptr, alpha, max_iter, False)
    
    # Map back to dict
    return {node_list[i]: ranks_array[i] for i in range(num_nodes)}

def shortest_path(G, source=None, target=None, weight=None, method='dijkstra'):
    """
    Intercepts shortest_path for single-source queries using Delta-Stepping.
    Fallbacks to NetworkX for complex queries (all-pairs, negative weights, etc.)
    """
    if source is not None and target is None and method == 'dijkstra':
        row_ptr, col_idx, weights, node_list, node_to_idx = get_csr(G, weight=weight)
        
        ds = MetalDeltaStepping()
        source_idx = node_to_idx[source]
        
        # Metal backend call
        # Note: Delta parameter tuning could be exposed or auto-computed
        distances = ds.compute(row_ptr, col_idx, weights, source_idx, 1.0)
        
        # We'd typically also need predecessor array to reconstruct paths,
        # but for demonstration we return the distances dict.
        return {node_list[i]: distances[i] for i in range(G.number_of_nodes())}
    
    # Fallback to networkx
    return nx.shortest_path(G, source, target, weight, method)

def connected_components(G):
    """
    Drop-in for networkx.connected_components using Shiloach-Vishkin on Metal.
    """
    if G.is_directed():
        raise nx.NetworkXNotImplemented("Not implemented for directed graphs")
        
    row_ptr, col_idx, _, node_list, _ = get_csr(G)
    
    cc = MetalConnectedComponents()
    num_nodes = G.number_of_nodes()
    
    # Metal backend call
    parent_array = cc.compute(row_ptr, col_idx, num_nodes)
    
    # Group nodes by parent
    components = {}
    for i, p in enumerate(parent_array):
        # Resolve path compression fully on CPU just in case
        root = p
        while parent_array[root] != root:
            root = parent_array[root]
            
        if root not in components:
            components[root] = set()
        components[root].add(node_list[i])
        
    return iter(components.values())
