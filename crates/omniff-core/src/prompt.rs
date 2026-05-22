use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptControl {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validator_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validator_threshold: Option<f32>,
    #[serde(default)]
    pub control_maps: Vec<String>,
    #[serde(default)]
    pub preserve: PreservePolicy,
    #[serde(default)]
    pub constraints: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreservePolicy {
    #[serde(default)]
    pub identity: bool,
    #[serde(default)]
    pub layout: bool,
    #[serde(default)]
    pub audio: bool,
    #[serde(default)]
    pub camera: bool,
}
