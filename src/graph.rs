use pyo3::prelude::*;

/// Represents a single node in the graph, typical for Pathfinding algorithms.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Node {
    pub id: u32,
    pub edge_start: u32,
    pub edge_count: u32,
    pub distance: f32,      // Used for algorithms like Dijkstra/Bellman-Ford
    pub predecessor: i32,    // Predecessor node ID for path reconstruction (-1 = none)
    pub visited: u32,        // 0 = false, 1 = true (boolean aligned to 4 bytes for GPU)
}

/// Represents a directed edge to another node.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Edge {
    pub target_node: u32,
    pub weight: f32,
}

/// A graph structure that holds our buffers in a format ready for PyO3 and Metal.
#[pyclass]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
}

#[pymethods]
impl Graph {
    #[new]
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph and return its index
    pub fn add_node(&mut self, distance: f32) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            id,
            edge_start: 0,
            edge_count: 0,
            distance,
            predecessor: -1,
            visited: 0,
        });
        id
    }

    /// Add an edge between two nodes. 
    /// Note: this assumes edges are added in order of the source node to keep `edge_start` contiguous.
    pub fn add_edge(&mut self, source: u32, target: u32, weight: f32) {
        let edge_idx = self.edges.len() as u32;
        self.edges.push(Edge {
            target_node: target,
            weight,
        });
        
        // Update the source node's edge range
        let node = &mut self.nodes[source as usize];
        if node.edge_count == 0 {
            node.edge_start = edge_idx;
        }
        node.edge_count += 1;
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}
