use omniff_core::modality::Modality;
use omniff_core::packet::OmniPacket;

pub trait Demuxer {
    fn detect_modality(&self, input: &str) -> Result<Modality, omniff_core::error::OmniError>;
    fn demux(&self, input: &str) -> Result<Vec<OmniPacket>, omniff_core::error::OmniError>;
}
