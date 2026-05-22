use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::modality::Modality;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniPacket {
    pub id: Uuid,
    pub modality: Modality,
    pub data: PacketData,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum PacketData {
    TextChunk(String),
    Bytes(Vec<u8>),
    FilePath(String),
    JsonMessage(serde_json::Value),
}

impl OmniPacket {
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            modality: Modality::Text,
            data: PacketData::TextChunk(content.into()),
            metadata: serde_json::Value::Null,
        }
    }

    pub fn from_file(path: impl Into<String>, modality: Modality) -> Self {
        Self {
            id: Uuid::new_v4(),
            modality,
            data: PacketData::FilePath(path.into()),
            metadata: serde_json::Value::Null,
        }
    }
}
