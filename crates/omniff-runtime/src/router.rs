use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    pub selected_route: RouteClass,
    pub confidence: f32,
    pub risk: RiskLevel,
    pub thinking: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RouteClass {
    TextSimple,
    TextNormal,
    TextComplex,
    AudioTranscribeOnly,
    AudioQa,
    ImageCaption,
    ImageEdit,
    TextToImage,
    TextToVideo,
    ImageToVideo,
    VideoSummary,
    VideoToVideo,
    DocumentOcrQa,
    DocumentToDocument,
    RejectOrHumanReview,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

pub trait Router {
    fn route(&self, input: &str) -> Result<RouteDecision, omniff_core::error::OmniError>;
}
