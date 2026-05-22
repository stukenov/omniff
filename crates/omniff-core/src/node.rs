use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniNode {
    pub id: String,
    pub node_type: NodeType,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    Model(ModelNodeConfig),
    Filter(String),
    Tool(String),
    Validator(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelNodeConfig {
    pub backend: ModelBackend,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelBackend {
    Transformers,
    Diffusers,
    OnnxRuntime,
    Candle,
    Gguf,
    ExternalApi,
    Custom,
}
