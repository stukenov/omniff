use omniff_core::modality::Modality;

use crate::graph::OmniGraph;

#[derive(Debug, Clone)]
pub struct ThinkingLevel {
    pub mode: ThinkingMode,
}

#[derive(Debug, Clone, Copy)]
pub enum ThinkingMode {
    Off,
    Fast,
    Normal,
    Deep,
    Research,
}

pub trait GraphPlanner {
    fn plan(
        &self,
        input_modality: Modality,
        output_modality: Modality,
        thinking: ThinkingMode,
        prompt: &str,
    ) -> Result<OmniGraph, omniff_core::error::OmniError>;
}
