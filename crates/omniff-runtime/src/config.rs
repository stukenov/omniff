use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use omniff_core::model::OmniModelSpec;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniFFConfig {
    pub name: String,
    pub version: String,
    pub router: RouterConfig,
    pub experts: HashMap<String, OmniModelSpec>,
    #[serde(default)]
    pub graph_templates_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    #[serde(rename = "type")]
    pub router_type: String,
    pub path: String,
}

impl OmniFFConfig {
    pub fn load(path: &Path) -> Result<Self, omniff_core::error::OmniError> {
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content).map_err(|e| {
            omniff_core::error::OmniError::ConfigError(e.to_string())
        })
    }
}
