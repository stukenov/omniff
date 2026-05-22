use crate::config::OmniFFConfig;

pub struct OmniFFRuntime {
    pub config: OmniFFConfig,
}

impl OmniFFRuntime {
    pub fn from_config(config: OmniFFConfig) -> Self {
        Self { config }
    }

    pub fn from_yaml(path: &std::path::Path) -> Result<Self, omniff_core::error::OmniError> {
        let config = OmniFFConfig::load(path)?;
        Ok(Self::from_config(config))
    }
}
