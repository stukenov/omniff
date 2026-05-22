use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use omniff_core::node::OmniNode;
use omniff_core::prompt::PromptControl;

use crate::edge::Edge;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniGraph {
    pub id: String,
    pub nodes: Vec<OmniNode>,
    pub edges: Vec<Edge>,
    pub side_data: Option<PromptControl>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl OmniGraph {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            side_data: None,
            metadata: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: OmniNode) {
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) {
        self.edges.push(Edge {
            from: from.into(),
            to: to.into(),
        });
    }

    pub fn topological_order(&self) -> Result<Vec<&OmniNode>, String> {
        // TODO: implement topological sort over DAG
        Ok(self.nodes.iter().collect())
    }
}
