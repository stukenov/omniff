use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::graph::OmniGraph;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphTemplate {
    pub name: String,
    pub description: String,
    pub input_modality: String,
    pub output_modality: String,
    pub graph: OmniGraph,
}

impl GraphTemplate {
    pub fn load_from_yaml(path: &Path) -> Result<Self, omniff_core::error::OmniError> {
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content).map_err(|e| {
            omniff_core::error::OmniError::ConfigError(e.to_string())
        })
    }
}
