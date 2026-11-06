//! Spark-X2.6-1.7B as a Burn module.
//!
//! The architecture follows the published shape: 28 layers, hidden size 2,048,
//! 8 query heads, 2 key/value heads, a 512-token sliding window on the local
//! layers, a 6,656-wide GELU-gated feed-forward, tied embeddings, partial RoPE,
//! and RMSNorm. Sliding layers rotate the full head at theta 10,000. Global
//! layers rotate the first quarter of the head at theta 5,000,000. Context
//! length is 1,048,576, and every fourth layer is global.

mod attention;
mod checkpoint;
mod config;
mod device;
mod lora;
mod master_adam;
mod model;
mod rope;

pub use attention::{AttentionKind, attention_mask};
pub use checkpoint::{
    CheckpointError, LogitReport, compare_logits, fuse_qkv_weights, huggingface_key_to_burn,
    load_spark_checkpoint, sidecar_dir,
};
pub use config::{HiddenAct, RopeSpec, SparkConfig, SparkConfigError};
pub use device::{accelerator_name, bf16_device};
pub use lora::{
    LoraSpec, LoraTarget, attach_lora, expected_lora_parameters, load_lora_adapters,
    reference_policy, save_lora_adapters,
};
pub use master_adam::Fp32MasterAdam;
pub use model::{BatchedLayerKv, BatchedModelCache, ModelCache, SparkModel};
pub use rope::apply_rope;
