use thiserror::Error;

#[derive(Debug, Error)]
pub enum OmniError {
    #[error("unsupported modality conversion: {from} → {to}")]
    UnsupportedConversion { from: String, to: String },

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("graph execution failed at node '{node}': {reason}")]
    GraphExecutionFailed { node: String, reason: String },

    #[error("validation failed: {0}")]
    ValidationFailed(String),

    #[error("router error: {0}")]
    RouterError(String),

    #[error("scheduler error: {0}")]
    SchedulerError(String),

    #[error("config error: {0}")]
    ConfigError(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}
