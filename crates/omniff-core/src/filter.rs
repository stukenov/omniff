use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniFilter {
    pub name: String,
    pub filter_type: FilterType,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterType {
    Resize,
    Crop,
    Normalize,
    ExtractDepth,
    ExtractEdges,
    ExtractPose,
    DetectFaces,
    TrackObjects,
    SplitShots,
    Summarize,
    Translate,
    StyleTransfer,
    Custom(String),
}
