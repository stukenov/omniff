use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniModelSpec {
    pub name: String,
    pub model_type: ModelType,
    pub path: String,
    pub loading: LoadingPolicy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    CausalLm,
    SpeechToText,
    VisionLanguage,
    DiffusionImageEdit,
    DiffusionVideoEdit,
    TextToSpeech,
    EncoderClassifier,
    Custom(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadingPolicy {
    Hot,
    Warm,
    Cold,
}
