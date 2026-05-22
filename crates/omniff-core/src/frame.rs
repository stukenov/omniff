use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::modality::Modality;
use crate::prompt::PromptControl;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniFrame {
    pub id: Uuid,
    pub modality: Modality,
    pub data: FrameData,
    pub side_data: Option<PromptControl>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum FrameData {
    TextTokens { tokens: Vec<u32> },
    RawText { text: String },
    Embedding { values: Vec<f32> },
    TensorRef { path: String, shape: Vec<usize> },
    FilePath { path: String },
}

impl OmniFrame {
    pub fn raw_text(text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            modality: Modality::Text,
            data: FrameData::RawText { text: text.into() },
            side_data: None,
            metadata: serde_json::Value::Null,
        }
    }
}
