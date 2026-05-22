use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniValidator {
    pub name: String,
    pub validator_type: ValidatorType,
    pub threshold: f32,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidatorType {
    Language,
    Schema,
    PromptAdherence,
    IdentityPreservation,
    LayoutPreservation,
    TemporalConsistency,
    Safety,
    Factuality,
    Format,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub passed: bool,
    pub score: f32,
    pub validator: String,
    pub details: Option<String>,
}
