//! Import a Spark safetensors checkpoint and compare logits to a reference dump.
//!
//! Hugging Face exports often store separate Q, K and V matrices in PyTorch
//! layout `[out, in]`. This loader concatenates them along the output axis,
//! renames keys onto the Burn module tree, and lets Burn's PyTorch adapter
//! transpose linear weights. RMSNorm `weight` is renamed to `gamma` by that
//! adapter. Tied `lm_head` tensors are skipped.
//!
//! Logit parity is a separate step. Loading weights does not by itself prove
//! the forward pass matches the original runtime.

use std::collections::HashMap;
use std::path::Path;

use burn::store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use safetensors::tensor::{SafeTensors, TensorView};
use safetensors::{Dtype, SafeTensorError, View, serialize};
use thiserror::Error;

use crate::model::SparkModel;

#[derive(Debug, Error)]
pub enum CheckpointError {
    #[error("{0}")]
    Message(String),
}

impl From<std::io::Error> for CheckpointError {
    fn from(value: std::io::Error) -> Self {
        Self::Message(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogitReport {
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub matched: bool,
}

pub fn compare_logits(
    actual: &[f32],
    reference: &[f32],
    atol: f32,
) -> Result<LogitReport, CheckpointError> {
    if actual.len() != reference.len() {
        return Err(CheckpointError::Message(format!(
            "logit length {} does not match reference {}",
            actual.len(),
            reference.len()
        )));
    }
    if actual.is_empty() {
        return Err(CheckpointError::Message("logit dump is empty".into()));
    }
    let mut max_abs = 0.0_f32;
    let mut sum = 0.0_f32;
    for (left, right) in actual.iter().zip(reference) {
        let diff = (left - right).abs();
        max_abs = max_abs.max(diff);
        sum += diff;
    }
    let mean = sum / actual.len() as f32;
    Ok(LogitReport {
        max_abs_diff: max_abs,
        mean_abs_diff: mean,
        matched: max_abs <= atol,
    })
}

struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Map a Hugging Face key onto a Burn parameter path.
///
/// Returns `None` for tensors this model does not store, such as a tied head.
pub fn huggingface_key_to_burn(key: &str) -> Option<String> {
    if key == "embed.weight"
        || key.starts_with("layers.")
        || key == "norm.weight"
        || key.starts_with("norm.")
    {
        return Some(key.to_string());
    }
    let rules: &[(&str, &str)] = &[
        ("model.embed_tokens.", "embed."),
        ("model.embedding.", "embed."),
        ("model.norm.", "norm."),
    ];
    if let Some((prefix, replacement)) = rules.iter().find(|(prefix, _)| key.starts_with(prefix)) {
        return Some(format!("{replacement}{}", &key[prefix.len()..]));
    }
    let layer = layer_index(key)?;
    let rest = key.split(&format!("model.layers.{layer}.")).nth(1)?;
    let mapped = if let Some(suffix) = rest.strip_prefix("input_layernorm.") {
        format!("layers.{layer}.input_norm.{suffix}")
    } else if let Some(suffix) = rest.strip_prefix("post_attention_layernorm.") {
        format!("layers.{layer}.post_norm.{suffix}")
    } else if let Some(suffix) = rest.strip_prefix("self_attn.qkv_proj.") {
        format!("layers.{layer}.attn.qkv.{suffix}")
    } else if let Some(suffix) = rest.strip_prefix("self_attn.q_k_v_proj.") {
        format!("layers.{layer}.attn.qkv.{suffix}")
    } else if let Some(suffix) = rest.strip_prefix("self_attn.o_proj.") {
        format!("layers.{layer}.attn.output.{suffix}")
    } else if let Some(suffix) = rest.strip_prefix("self_attn.out_proj.") {
        format!("layers.{layer}.attn.output.{suffix}")
    } else if let Some(suffix) = strip_gate_suffix(rest) {
        format!("layers.{layer}.attn.gate.{suffix}")
    } else if let Some(suffix) = rest.strip_prefix("mlp.gate_proj.") {
        format!("layers.{layer}.mlp.gate.{suffix}")
    } else if let Some(suffix) = rest.strip_prefix("mlp.up_proj.") {
        format!("layers.{layer}.mlp.up.{suffix}")
    } else if let Some(suffix) = rest.strip_prefix("mlp.down_proj.") {
        format!("layers.{layer}.mlp.down.{suffix}")
    } else if rest.starts_with("self_attn.q_proj.")
        || rest.starts_with("self_attn.k_proj.")
        || rest.starts_with("self_attn.v_proj.")
    {
        return None;
    } else if key.contains("lm_head") {
        return None;
    } else {
        return None;
    };
    Some(mapped)
}

fn strip_gate_suffix(rest: &str) -> Option<&str> {
    for prefix in [
        "self_attn.gate_proj.",
        "self_attn.attn_gate.",
        "self_attn.output_gate.",
        "self_attn.g_proj.",
    ] {
        if let Some(suffix) = rest.strip_prefix(prefix) {
            return Some(suffix);
        }
    }
    None
}

fn layer_index(key: &str) -> Option<usize> {
    let marker = "model.layers.";
    let start = key.find(marker)? + marker.len();
    let end = key[start..].find('.')? + start;
    key[start..end].parse().ok()
}

/// Concatenate PyTorch Q, K and V tensors along the output axis.
///
/// Each input is contiguous `[out, in]`. The result is `[q_out + k_out + v_out, in]`.
pub fn fuse_qkv_weights(q: &[u8], k: &[u8], v: &[u8]) -> Vec<u8> {
    let mut fused = Vec::with_capacity(q.len() + k.len() + v.len());
    fused.extend_from_slice(q);
    fused.extend_from_slice(k);
    fused.extend_from_slice(v);
    fused
}

fn tensor_record(view: TensorView<'_>) -> OwnedTensor {
    OwnedTensor {
        dtype: view.dtype(),
        shape: view.shape().to_vec(),
        data: view.data().to_vec(),
    }
}

fn absorb_file(
    file: &SafeTensors<'_>,
    owned: &mut HashMap<String, OwnedTensor>,
) -> Result<(), CheckpointError> {
    let names = file.names();
    let mut layers = Vec::new();
    for name in &names {
        if let Some(layer) = layer_index(name)
            && name.ends_with("self_attn.q_proj.weight")
        {
            layers.push(layer);
        }
    }
    layers.sort_unstable();
    layers.dedup();

    for layer in layers {
        let q_name = format!("model.layers.{layer}.self_attn.q_proj.weight");
        let k_name = format!("model.layers.{layer}.self_attn.k_proj.weight");
        let v_name = format!("model.layers.{layer}.self_attn.v_proj.weight");
        let q = file
            .tensor(&q_name)
            .map_err(|err| CheckpointError::Message(err.to_string()))?;
        let k = file
            .tensor(&k_name)
            .map_err(|err| CheckpointError::Message(err.to_string()))?;
        let v = file
            .tensor(&v_name)
            .map_err(|err| CheckpointError::Message(err.to_string()))?;
        if q.dtype() != k.dtype() || k.dtype() != v.dtype() {
            return Err(CheckpointError::Message(format!(
                "QKV dtypes differ in layer {layer}"
            )));
        }
        if q.shape().len() != 2 || k.shape()[1] != q.shape()[1] || v.shape()[1] != q.shape()[1] {
            return Err(CheckpointError::Message(format!(
                "QKV shapes are not concatenable in layer {layer}"
            )));
        }
        let data = fuse_qkv_weights(q.data(), k.data(), v.data());
        let shape = vec![q.shape()[0] + k.shape()[0] + v.shape()[0], q.shape()[1]];
        owned.insert(
            format!("layers.{layer}.attn.qkv.weight"),
            OwnedTensor {
                dtype: q.dtype(),
                shape,
                data,
            },
        );
    }

    for name in names {
        if name.contains("self_attn.q_proj.")
            || name.contains("self_attn.k_proj.")
            || name.contains("self_attn.v_proj.")
        {
            continue;
        }
        let Some(mapped) = huggingface_key_to_burn(&name) else {
            continue;
        };
        if owned.contains_key(&mapped) {
            continue;
        }
        let view = file
            .tensor(&name)
            .map_err(|err| CheckpointError::Message(err.to_string()))?;
        owned.insert(mapped, tensor_record(view));
    }
    Ok(())
}

fn serialize_owned(owned: &HashMap<String, OwnedTensor>) -> Result<Vec<u8>, CheckpointError> {
    if owned.is_empty() {
        return Err(CheckpointError::Message(
            "checkpoint contained no tensors this model stores".into(),
        ));
    }
    let pairs: Vec<(String, &OwnedTensor)> = owned
        .iter()
        .map(|(key, value)| (key.clone(), value))
        .collect();
    serialize(pairs, None).map_err(|err| CheckpointError::Message(err.to_string()))
}

fn prepare_checkpoint_bytes(bytes: &[u8]) -> Result<Vec<u8>, CheckpointError> {
    let bytes = covered_bytes(bytes, "checkpoint")?;
    let file = SafeTensors::deserialize(bytes)
        .map_err(|err| CheckpointError::Message(format!("checkpoint: {err}")))?;
    let mut owned = HashMap::new();
    absorb_file(&file, &mut owned)?;
    serialize_owned(&owned)
}

/// The bytes the header actually describes.
///
/// The strict reader rejects a file whose length is not exactly the last tensor
/// offset. Hugging Face shards sometimes carry a few trailing padding bytes.
/// A short file is a truncated download, and the missing span cannot be guessed.
fn covered_bytes<'a>(bytes: &'a [u8], label: &str) -> Result<&'a [u8], CheckpointError> {
    match SafeTensors::deserialize(bytes) {
        Ok(_) => Ok(bytes),
        Err(SafeTensorError::MetadataIncompleteBuffer) => {
            let declared = declared_file_len(bytes)
                .map_err(|err| CheckpointError::Message(format!("{label}: {err}")))?;
            if declared < bytes.len() {
                Ok(&bytes[..declared])
            } else {
                Err(CheckpointError::Message(format!(
                    "{label} is truncated: {} bytes on disk, the header covers {declared} bytes. Download the shard again",
                    bytes.len()
                )))
            }
        }
        Err(err) => Err(CheckpointError::Message(format!("{label}: {err}"))),
    }
}

fn declared_file_len(bytes: &[u8]) -> Result<usize, String> {
    if bytes.len() < 8 {
        return Err("header is shorter than 8 bytes".into());
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| "header length overflows".to_string())?;
    if bytes.len() < header_end {
        return Err(format!(
            "header claims {header_len} bytes but the file has {}",
            bytes.len()
        ));
    }
    let header = std::str::from_utf8(&bytes[8..header_end])
        .map_err(|err| format!("header is not utf-8: {err}"))?;
    let value: serde_json::Value =
        serde_json::from_str(header).map_err(|err| format!("header is not json: {err}"))?;
    let entries = value
        .as_object()
        .ok_or_else(|| "header is not an object".to_string())?;
    let mut data_end = 0usize;
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(|value| value.as_array())
            .ok_or_else(|| format!("tensor {name} has no data_offsets"))?;
        let end = offsets
            .get(1)
            .and_then(|value| value.as_u64())
            .ok_or_else(|| format!("tensor {name} has no end offset"))? as usize;
        data_end = data_end.max(end);
    }
    header_end
        .checked_add(data_end)
        .ok_or_else(|| "declared file length overflows".to_string())
}

#[derive(serde::Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

fn prepare_directory(dir: &Path) -> Result<Vec<u8>, CheckpointError> {
    let index_path = dir.join("model.safetensors.index.json");
    let mut owned = HashMap::new();
    if index_path.is_file() {
        let text = std::fs::read_to_string(&index_path).map_err(|err| {
            CheckpointError::Message(format!("reading {}: {err}", index_path.display()))
        })?;
        let index: WeightIndex =
            serde_json::from_str(&text).map_err(|err| CheckpointError::Message(err.to_string()))?;
        let mut shards: Vec<String> = index.weight_map.into_values().collect();
        shards.sort();
        shards.dedup();
        if shards.is_empty() {
            return Err(CheckpointError::Message(format!(
                "{} has an empty weight map",
                index_path.display()
            )));
        }
        for shard in shards {
            absorb_shard(&dir.join(&shard), &mut owned)?;
        }
    } else {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                files.push(path);
            }
        }
        files.sort();
        if files.is_empty() {
            return Err(CheckpointError::Message(format!(
                "{} has no safetensors weights",
                dir.display()
            )));
        }
        for path in files {
            absorb_shard(&path, &mut owned)?;
        }
    }
    serialize_owned(&owned)
}

fn absorb_shard(
    path: &Path,
    owned: &mut HashMap<String, OwnedTensor>,
) -> Result<(), CheckpointError> {
    let label = path.display().to_string();
    let bytes = std::fs::read(path)
        .map_err(|err| CheckpointError::Message(format!("reading {label}: {err}")))?;
    let bytes = covered_bytes(&bytes, &label)?;
    let file = SafeTensors::deserialize(bytes)
        .map_err(|err| CheckpointError::Message(format!("{label}: {err}")))?;
    absorb_file(&file, owned)
}

/// Directory that holds `config.json` and `tokenizer.json` beside the weights.
///
/// A directory is used as itself. A weights file uses its parent.
pub fn sidecar_dir(path: &Path) -> std::path::PathBuf {
    if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.to_path_buf())
    }
}

fn checkpoint_directory(path: &Path) -> Option<&Path> {
    if path.is_dir() {
        Some(path)
    } else if path
        .file_name()
        .is_some_and(|name| name == "model.safetensors.index.json")
    {
        path.parent()
    } else {
        None
    }
}

pub fn load_spark_checkpoint(
    model: SparkModel,
    path: &Path,
) -> Result<SparkModel, CheckpointError> {
    let prepared = if let Some(dir) = checkpoint_directory(path) {
        prepare_directory(dir)?
    } else {
        let bytes = std::fs::read(path).map_err(|err| {
            CheckpointError::Message(format!("reading {}: {err}", path.display()))
        })?;
        prepare_checkpoint_bytes(&bytes)?
    };
    let mut store = SafetensorsStore::from_bytes(Some(prepared))
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true);
    let mut model = model;
    model
        .load_from(&mut store)
        .map_err(|err| CheckpointError::Message(format!("{err:?}")))?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SparkConfig;
    use burn::tensor::Device;

    #[test]
    fn key_map_rewrites_a_fused_attention_weight_and_skips_the_tied_head() {
        assert_eq!(
            huggingface_key_to_burn("model.layers.3.self_attn.qkv_proj.weight").as_deref(),
            Some("layers.3.attn.qkv.weight")
        );
        assert_eq!(
            huggingface_key_to_burn("model.layers.0.mlp.up_proj.weight").as_deref(),
            Some("layers.0.mlp.up.weight")
        );
        assert_eq!(
            huggingface_key_to_burn("model.embed_tokens.weight").as_deref(),
            Some("embed.weight")
        );
        assert_eq!(
            huggingface_key_to_burn("model.embedding.weight").as_deref(),
            Some("embed.weight")
        );
        assert_eq!(
            huggingface_key_to_burn("model.layers.0.self_attn.q_k_v_proj.weight").as_deref(),
            Some("layers.0.attn.qkv.weight")
        );
        assert_eq!(
            huggingface_key_to_burn("model.layers.0.self_attn.out_proj.weight").as_deref(),
            Some("layers.0.attn.output.weight")
        );
        assert_eq!(
            huggingface_key_to_burn("model.layers.0.self_attn.g_proj.weight").as_deref(),
            Some("layers.0.attn.gate.weight")
        );
        assert_eq!(huggingface_key_to_burn("lm_head.weight"), None);
        assert_eq!(
            huggingface_key_to_burn("model.layers.1.self_attn.q_proj.weight"),
            None
        );
    }

    #[test]
    fn qkv_byte_concat_stacks_output_rows() {
        assert_eq!(
            fuse_qkv_weights(&[1, 2], &[3], &[4, 5]),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn logit_comparison_accepts_a_match_and_rejects_a_shift() {
        let matched = compare_logits(&[1.0, 2.0], &[1.0, 2.0001], 1e-3).unwrap();
        assert!(matched.matched);
        let shifted = compare_logits(&[1.0, 2.0], &[1.0, 3.0], 1e-3).unwrap();
        assert!(!shifted.matched);
        assert!(compare_logits(&[1.0], &[1.0, 2.0], 1e-3).is_err());
    }

    #[test]
    fn partial_embedding_checkpoint_round_trips_through_burn() {
        let config = SparkConfig::tiny();
        let device = Device::flex();
        let model = SparkModel::new(&config, &device);
        let elements = config.vocab_size * config.hidden_size;
        let mut data = vec![0.0_f32; elements];
        data[0] = 1.25;
        let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
        let tensor = OwnedTensor {
            dtype: Dtype::F32,
            shape: vec![config.vocab_size, config.hidden_size],
            data: bytes,
        };
        let packed = serialize([("model.embed_tokens.weight", &tensor)], None).unwrap();
        let prepared = prepare_checkpoint_bytes(&packed).unwrap();
        let mut store = SafetensorsStore::from_bytes(Some(prepared))
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(true);
        let mut model = model;
        model.load_from(&mut store).unwrap();
        let loaded = model.embed.weight.val().into_data();
        let values = loaded.as_slice::<f32>().unwrap();
        assert!((values[0] - 1.25).abs() < 1e-6);
    }

    #[test]
    fn loaded_checkpoint_preserves_pad_embedding_row() {
        let config = SparkConfig::tiny();
        let device = Device::flex();
        let model = SparkModel::new(&config, &device);
        let elements = config.vocab_size * config.hidden_size;
        let mut data = vec![0.0_f32; elements];
        let pad_offset = config.pad_token_id * config.hidden_size;
        data[pad_offset] = 3.5;
        let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
        let tensor = OwnedTensor {
            dtype: Dtype::F32,
            shape: vec![config.vocab_size, config.hidden_size],
            data: bytes,
        };
        let packed = serialize([("model.embed_tokens.weight", &tensor)], None).unwrap();
        let temp_dir = std::env::temp_dir().join(format!("spark_test_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let file_path = temp_dir.join("model.safetensors");
        std::fs::write(&file_path, packed).unwrap();
        let loaded = load_spark_checkpoint(model, &file_path).unwrap();
        let _ = std::fs::remove_dir_all(&temp_dir);
        let weight = loaded.embed.weight.val().into_data();
        let values = weight.as_slice::<f32>().unwrap();
        assert!(
            (values[pad_offset] - 3.5).abs() < 1e-6,
            "pad token embedding row must not be wiped on checkpoint load"
        );
    }

    #[test]
    fn trailing_padding_loads_and_a_short_file_is_truncated() {
        let tensor = OwnedTensor {
            dtype: Dtype::F32,
            shape: vec![1, 2],
            data: 1.0_f32
                .to_le_bytes()
                .into_iter()
                .chain(2.0_f32.to_le_bytes())
                .collect(),
        };
        let packed = serialize([("model.embed_tokens.weight", &tensor)], None).unwrap();
        let mut padded = packed.clone();
        padded.extend_from_slice(b"padding");
        let prepared = prepare_checkpoint_bytes(&padded).unwrap();
        let file = SafeTensors::deserialize(&prepared).unwrap();
        assert_eq!(file.tensor("embed.weight").unwrap().data(), tensor.data);

        let short = &packed[..packed.len() - 3];
        let message = prepare_checkpoint_bytes(short).unwrap_err().to_string();
        assert!(message.contains("truncated"), "{message}");
        assert!(message.contains("header covers"), "{message}");
    }

    #[test]
    fn sharded_index_maps_embedding_and_norm_from_separate_files() {
        let dir = std::env::temp_dir().join(format!("spark-shards-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let embed = OwnedTensor {
            dtype: Dtype::F32,
            shape: vec![1, 2],
            data: 1.5_f32
                .to_le_bytes()
                .into_iter()
                .chain(2.5_f32.to_le_bytes())
                .collect(),
        };
        let norm = OwnedTensor {
            dtype: Dtype::F32,
            shape: vec![2],
            data: 0.25_f32
                .to_le_bytes()
                .into_iter()
                .chain(0.5_f32.to_le_bytes())
                .collect(),
        };
        let first = serialize([("model.embedding.weight", &embed)], None).unwrap();
        let second = serialize([("model.norm.weight", &norm)], None).unwrap();
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), first).unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), second).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embedding.weight":"model-00001-of-00002.safetensors","model.norm.weight":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let prepared = prepare_directory(&dir).unwrap();
        let file = SafeTensors::deserialize(&prepared).unwrap();
        let mut names: Vec<String> = file.names().into_iter().map(str::to_string).collect();
        names.sort();
        assert_eq!(
            names,
            vec!["embed.weight".to_string(), "norm.weight".to_string()]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
