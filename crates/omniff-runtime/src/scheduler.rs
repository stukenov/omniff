use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevicePlacement {
    pub node_id: String,
    pub device: Device,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    Cpu,
    Gpu(usize),
}

pub trait Scheduler {
    fn assign_devices(
        &self,
        graph: &omniff_graph::graph::OmniGraph,
    ) -> Result<Vec<DevicePlacement>, omniff_core::error::OmniError>;

    fn should_preload(&self, model_name: &str) -> bool;
    fn should_unload(&self, model_name: &str) -> bool;
}
