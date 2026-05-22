use omniff_core::frame::OmniFrame;

pub trait OutputEncoder {
    fn encode(&self, frames: Vec<OmniFrame>, output_path: &str) -> Result<(), omniff_core::error::OmniError>;
}
