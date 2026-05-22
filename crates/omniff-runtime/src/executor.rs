use omniff_core::frame::OmniFrame;
use omniff_graph::graph::OmniGraph;

pub trait GraphExecutor {
    fn execute(
        &self,
        graph: &OmniGraph,
        inputs: Vec<OmniFrame>,
    ) -> Result<Vec<OmniFrame>, omniff_core::error::OmniError>;
}
