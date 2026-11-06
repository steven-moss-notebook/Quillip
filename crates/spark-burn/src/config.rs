//! Published Spark-X2.6-1.7B shape.
//!
//! The October 27, 2026 `config.json` uses a GELU-gated MLP, RMSNorm epsilon
//! 1e-6, context length 1,048,576, and a different RoPE setting on sliding and
//! global layers. Sliding layers rotate the full head at theta 10,000. Global
//! layers rotate the first quarter of the head at theta 5,000,000.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::attention::AttentionKind;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RopeSpec {
    pub theta: f32,
    pub partial_rotary_factor: f32,
}

impl RopeSpec {
    pub fn sliding() -> Self {
        Self {
            theta: 10_000.0,
            partial_rotary_factor: 1.0,
        }
    }

    pub fn global() -> Self {
        Self {
            theta: 5_000_000.0,
            partial_rotary_factor: 0.25,
        }
    }

    /// Width of the rotary slice. Spark truncates `head_dim * factor` and the
    /// rotation needs an even width.
    pub fn rotary_dim(&self, head_dim: usize) -> usize {
        let dim = (head_dim as f32 * self.partial_rotary_factor) as usize;
        dim - (dim % 2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HiddenAct {
    Gelu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparkConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// Local-attention lookback, including the current token.
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub sliding_rope: RopeSpec,
    pub global_rope: RopeSpec,
    /// Global attention on every Nth layer when `layer_types` is empty. Zero means every layer is local.
    pub global_every: usize,
    pub layer_types: Vec<AttentionKind>,
    pub hidden_act: HiddenAct,
    pub tie_word_embeddings: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub initializer_range: f64,
    pub pad_token_id: usize,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SparkConfigError {
    #[error("hidden size {hidden} must equal num_attention_heads {heads} * head_dim {head_dim}")]
    HeadDim {
        hidden: usize,
        heads: usize,
        head_dim: usize,
    },
    #[error("num_attention_heads {heads} must be divisible by num_key_value_heads {kv}")]
    Gqa { heads: usize, kv: usize },
    #[error("model dimensions must be non-zero")]
    Empty,
    #[error("sliding window must be non-zero")]
    Window,
    #[error("RoPE width must be a positive even slice of the head")]
    Rope,
    #[error("layer_types length {got} must equal num_hidden_layers {expected}")]
    Layers { got: usize, expected: usize },
}

impl SparkConfig {
    /// Shape of the October 27, 2026 Spark-X2.6-1.7B release.
    pub fn x2_6_1_7b() -> Self {
        let mut config = Self {
            vocab_size: 131_072,
            hidden_size: 2_048,
            intermediate_size: 6_656,
            num_hidden_layers: 28,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            sliding_window: 512,
            max_position_embeddings: 1_048_576,
            rms_norm_eps: 1e-6,
            sliding_rope: RopeSpec::sliding(),
            global_rope: RopeSpec::global(),
            global_every: 4,
            layer_types: Vec::new(),
            hidden_act: HiddenAct::Gelu,
            tie_word_embeddings: true,
            attention_bias: false,
            mlp_bias: false,
            initializer_range: 0.0221,
            pad_token_id: 2,
        };
        config.layer_types = (0..config.num_hidden_layers)
            .map(|index| {
                if config.global_every > 0 && (index + 1).is_multiple_of(config.global_every) {
                    AttentionKind::Global
                } else {
                    AttentionKind::Sliding
                }
            })
            .collect();
        config
    }

    /// A CPU-sized model with the same block structure, for tests and smoke steps.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            sliding_window: 4,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            sliding_rope: RopeSpec::sliding(),
            global_rope: RopeSpec::global(),
            global_every: 2,
            layer_types: Vec::new(),
            hidden_act: HiddenAct::Gelu,
            tie_word_embeddings: true,
            attention_bias: false,
            mlp_bias: false,
            initializer_range: 0.0221,
            pad_token_id: 2,
        }
    }

    pub fn validate(&self) -> Result<(), SparkConfigError> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
        {
            return Err(SparkConfigError::Empty);
        }
        if self.sliding_window == 0 {
            return Err(SparkConfigError::Window);
        }
        if self.hidden_size != self.num_attention_heads * self.head_dim {
            return Err(SparkConfigError::HeadDim {
                hidden: self.hidden_size,
                heads: self.num_attention_heads,
                head_dim: self.head_dim,
            });
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(SparkConfigError::Gqa {
                heads: self.num_attention_heads,
                kv: self.num_key_value_heads,
            });
        }
        if !self.layer_types.is_empty() && self.layer_types.len() != self.num_hidden_layers {
            return Err(SparkConfigError::Layers {
                got: self.layer_types.len(),
                expected: self.num_hidden_layers,
            });
        }
        for spec in [self.sliding_rope, self.global_rope] {
            let dim = spec.rotary_dim(self.head_dim);
            if spec.theta <= 0.0
                || spec.partial_rotary_factor <= 0.0
                || spec.partial_rotary_factor > 1.0
                || dim == 0
                || dim > self.head_dim
            {
                return Err(SparkConfigError::Rope);
            }
        }
        Ok(())
    }

    pub fn rope_spec(&self, index: usize) -> RopeSpec {
        if self.is_global_layer(index) {
            self.global_rope
        } else {
            self.sliding_rope
        }
    }

    pub fn is_global_layer(&self, index: usize) -> bool {
        if let Some(kind) = self.layer_types.get(index) {
            return *kind == AttentionKind::Global;
        }
        self.global_every > 0 && (index + 1).is_multiple_of(self.global_every)
    }

    /// (local layers, global layers).
    pub fn attention_layout(&self) -> (usize, usize) {
        let global = (0..self.num_hidden_layers)
            .filter(|index| self.is_global_layer(*index))
            .count();
        (self.num_hidden_layers - global, global)
    }

    pub fn qkv_features(&self) -> usize {
        self.num_attention_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim
    }

    /// Trainable weights, counting tied embeddings once and excluding RoPE frequencies.
    pub fn parameter_count(&self) -> u64 {
        let hidden = self.hidden_size as u64;
        let ffn = self.intermediate_size as u64;
        let qkv = hidden * self.qkv_features() as u64;
        let gate = hidden * self.num_attention_heads as u64;
        let output = (self.num_attention_heads * self.head_dim) as u64 * hidden;
        let norms = hidden * 2;
        let mlp = 3 * hidden * ffn;
        let mut per_layer = qkv + gate + output + norms + mlp;
        if self.attention_bias {
            per_layer += self.qkv_features() as u64
                + self.num_attention_heads as u64
                + (self.num_attention_heads * self.head_dim) as u64;
        }
        if self.mlp_bias {
            per_layer += 2 * ffn + hidden;
        }
        let embed = if self.tie_word_embeddings {
            self.vocab_size as u64 * hidden
        } else {
            2 * self.vocab_size as u64 * hidden
        };
        embed + per_layer * self.num_hidden_layers as u64 + hidden
    }

    /// LoRA parameter count for rank-`rank` factors.
    ///
    /// Attention-only targets the fused QKV projection and the attention output
    /// projection. Adding the three MLP projections is the later expansion.
    pub fn lora_parameter_count(&self, rank: u64, include_mlp: bool) -> u64 {
        let hidden = self.hidden_size as u64;
        let qkv_out = self.qkv_features() as u64;
        let attn = rank * (hidden + qkv_out) + rank * (hidden + hidden);
        let mlp = if include_mlp {
            let ffn = self.intermediate_size as u64;
            2 * rank * (hidden + ffn) + rank * (ffn + hidden)
        } else {
            0
        };
        (attn + mlp) * self.num_hidden_layers as u64
    }
}

#[derive(Debug, Deserialize)]
struct RopeFile {
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
    #[serde(default)]
    rope_theta: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default)]
    full_attention: Option<RopeFile>,
    #[serde(default)]
    sliding_attention: Option<RopeFile>,
}

/// Optional checkpoint `config.json`. Missing fields keep the 1.7B defaults.
#[derive(Debug, Deserialize)]
struct CheckpointConfigFile {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    sliding_window: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f32>,
    rope_parameters: Option<RopeParameters>,
    global_every: Option<usize>,
    layer_types: Option<Vec<String>>,
    hidden_act: Option<String>,
    tie_word_embeddings: Option<bool>,
    attention_bias: Option<bool>,
    mlp_bias: Option<bool>,
    initializer_range: Option<f64>,
    pad_token_id: Option<usize>,
    headwise_attn_output_gate: Option<bool>,
    gate_attn_act_mode: Option<String>,
    attention_dropout: Option<f64>,
}

impl SparkConfig {
    /// `config.json` beside the checkpoint, or the published 1.7B shape when it is absent.
    pub fn load_sidecar(checkpoint: &std::path::Path) -> Result<Self, String> {
        let path = crate::checkpoint::sidecar_dir(checkpoint).join("config.json");
        if !path.is_file() {
            return Ok(Self::x2_6_1_7b());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|err| format!("reading {}: {err}", path.display()))?;
        Self::from_config_json(&text)
    }

    pub fn from_config_json(text: &str) -> Result<Self, String> {
        let file: CheckpointConfigFile =
            serde_json::from_str(text).map_err(|err| err.to_string())?;
        let mut config = Self::x2_6_1_7b();
        assign(&mut config.vocab_size, file.vocab_size);
        assign(&mut config.hidden_size, file.hidden_size);
        assign(&mut config.intermediate_size, file.intermediate_size);
        assign(&mut config.num_hidden_layers, file.num_hidden_layers);
        assign(&mut config.num_attention_heads, file.num_attention_heads);
        assign(&mut config.num_key_value_heads, file.num_key_value_heads);
        assign(&mut config.head_dim, file.head_dim);
        assign(&mut config.sliding_window, file.sliding_window);
        assign(
            &mut config.max_position_embeddings,
            file.max_position_embeddings,
        );
        assign(&mut config.rms_norm_eps, file.rms_norm_eps);
        assign(&mut config.sliding_rope.theta, file.rope_theta);
        if let Some(parameters) = file.rope_parameters {
            apply_rope_file(&mut config.global_rope, parameters.full_attention);
            apply_rope_file(&mut config.sliding_rope, parameters.sliding_attention);
        }
        if let Some(value) = file.global_every {
            config.global_every = value;
            if file.layer_types.is_none() {
                config.layer_types.clear();
            }
        }
        if let Some(types) = file.layer_types {
            config.layer_types = types
                .iter()
                .map(|kind| match kind.as_str() {
                    "sliding_attention" => Ok(AttentionKind::Sliding),
                    "full_attention" => Ok(AttentionKind::Global),
                    other => Err(format!("unknown layer type {other}")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            config.num_hidden_layers = config.layer_types.len();
        }
        if let Some(act) = file.hidden_act {
            if act != "gelu" {
                return Err(format!("Spark MLP only implements gelu, config says {act}"));
            }
            config.hidden_act = HiddenAct::Gelu;
        }
        assign(&mut config.tie_word_embeddings, file.tie_word_embeddings);
        assign(&mut config.attention_bias, file.attention_bias);
        assign(&mut config.mlp_bias, file.mlp_bias);
        assign(&mut config.initializer_range, file.initializer_range);
        assign(&mut config.pad_token_id, file.pad_token_id);
        if file.headwise_attn_output_gate == Some(false) {
            return Err("Spark attention requires headwise_attn_output_gate".into());
        }
        if let Some(mode) = file.gate_attn_act_mode
            && mode != "sigmoid"
        {
            return Err(format!(
                "Spark attention gate only implements sigmoid, config says {mode}"
            ));
        }
        if let Some(dropout) = file.attention_dropout
            && dropout != 0.0
        {
            return Err(format!(
                "Spark attention dropout is 0 in the published config, got {dropout}"
            ));
        }
        config.validate().map_err(|err| err.to_string())?;
        Ok(config)
    }
}

fn assign<T>(slot: &mut T, value: Option<T>) {
    if let Some(value) = value {
        *slot = value;
    }
}

fn apply_rope_file(spec: &mut RopeSpec, file: Option<RopeFile>) {
    let Some(file) = file else {
        return;
    };
    assign(&mut spec.theta, file.rope_theta);
    assign(&mut spec.partial_rotary_factor, file.partial_rotary_factor);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_shape_matches_reported_counts() {
        let config = SparkConfig::x2_6_1_7b();
        config.validate().unwrap();
        assert_eq!(config.attention_layout(), (21, 7));
        assert_eq!(config.parameter_count(), 1_707_657_216);
        assert_eq!(config.lora_parameter_count(16, false), 4_128_768);
        assert_eq!(config.lora_parameter_count(16, true), 15_826_944);
        assert!(config.is_global_layer(3));
        assert!(!config.is_global_layer(0));
        assert!(config.is_global_layer(27));
        assert_eq!(config.max_position_embeddings, 1_048_576);
        assert_eq!(config.hidden_act, HiddenAct::Gelu);
        assert_eq!(config.sliding_rope, RopeSpec::sliding());
        assert_eq!(config.global_rope, RopeSpec::global());
        assert_eq!(config.global_rope.rotary_dim(config.head_dim), 64);
        assert_eq!(config.initializer_range, 0.0221);
        assert_eq!(config.pad_token_id, 2);
    }

    #[test]
    fn config_json_reads_the_published_rope_and_activation() {
        let config = SparkConfig::from_config_json(
            r#"{
                "vocab_size": 131072,
                "hidden_act": "gelu",
                "max_position_embeddings": 1048576,
                "initializer_range": 0.0221,
                "attention_bias": false,
                "mlp_bias": false,
                "headwise_attn_output_gate": true,
                "gate_attn_act_mode": "sigmoid",
                "attention_dropout": 0.0,
                "rope_parameters": {
                    "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000.0},
                    "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000.0}
                }
            }"#,
        )
        .unwrap();
        assert_eq!(config.sliding_rope.theta, 10_000.0);
        assert_eq!(config.global_rope.theta, 5_000_000.0);
        assert_eq!(config.global_rope.partial_rotary_factor, 0.25);
        assert_eq!(config.max_position_embeddings, 1_048_576);
        assert_eq!(config.num_hidden_layers, 28);
    }
}
