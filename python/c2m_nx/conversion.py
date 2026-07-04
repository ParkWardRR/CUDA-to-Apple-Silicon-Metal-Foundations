import networkx as nx
import numpy as np

def get_csr(G, weight='weight', default_weight=1.0):
    """
    Converts a NetworkX graph to CSR format numpy arrays for Metal processing.
    Caches the result on the graph object to avoid repeated conversion overhead.
    """
    # Check if cached and topology hasn't changed
    # A naive check using edge count (in production, we'd need a robust graph hash)
    if hasattr(G, '_c2m_csr_cache') and getattr(G, '_c2m_csr_edges', -1) == G.number_of_edges():
        return G._c2m_csr_cache
        
    num_nodes = G.number_of_nodes()
    
    # We need a stable node mapping if nodes aren't integers 0..N-1
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    row_ptr = np.zeros(num_nodes + 1, dtype=np.uint32)
    col_idx = []
    weights = []
    
    current_edge_idx = 0
    
    for i, u in enumerate(node_list):
        row_ptr[i] = current_edge_idx
        for v in G.neighbors(u):
            col_idx.append(node_to_idx[v])
            
            # Extract weight
            if nx.is_weighted(G) or weight in G[u][v]:
                w = G[u][v].get(weight, default_weight)
            else:
                w = default_weight
            weights.append(w)
            
            current_edge_idx += 1
            
    row_ptr[num_nodes] = current_edge_idx
    
    col_idx_np = np.array(col_idx, dtype=np.uint32)
    weights_np = np.array(weights, dtype=np.float32)
    
    csr_tuple = (row_ptr, col_idx_np, weights_np, node_list, node_to_idx)
    
    # Cache it
    G._c2m_csr_cache = csr_tuple
    G._c2m_csr_edges = G.number_of_edges()
    
    return csr_tuple
