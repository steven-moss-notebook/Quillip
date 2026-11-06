//! LoRA on the Spark projections.
//!
//! Burn 0.22 attaches adapters with `Module::apply_lora`. Rank 16 and alpha 32
//! on the fused QKV and attention-output projections is 4,128,768 trainable
//! parameters for the 1.7B shape. Adding the three MLP projections raises that
//! to 15,826,944. The base checkpoint stays frozen, so it remains the reference
//! policy.

use std::path::Path;

use burn::module::{Lora, Module, ParamGroup};
use burn::store::{ModuleSnapshot, SafetensorsStore};

use crate::config::SparkConfig;
use crate::model::SparkModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTarget {
    Attention,
    AttentionAndMlp,
}

#[derive(Debug, Clone, Copy)]
pub struct LoraSpec {
    pub rank: usize,
    pub alpha: f64,
    pub target: LoraTarget,
}

impl LoraSpec {
    /// Initial experiment: rank 16, alpha 32, attention projections only.
    pub fn attention_default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            target: LoraTarget::Attention,
        }
    }

    pub fn include_mlp(&self) -> bool {
        matches!(self.target, LoraTarget::AttentionAndMlp)
    }
}

pub fn lora_group(target: LoraTarget) -> Result<ParamGroup, String> {
    let patterns: &[&str] = match target {
        LoraTarget::Attention => &[r"\.attn\.(qkv|output)\.weight$"],
        LoraTarget::AttentionAndMlp => &[
            r"\.attn\.(qkv|output)\.weight$",
            r"\.mlp\.(gate|up|down)\.weight$",
        ],
    };
    ParamGroup::from_any_regexes(patterns.to_vec()).map_err(|err| format!("{err:?}"))
}

/// Freeze the base weights and attach adapters. Call this on a model that was
/// created on an autodiff device.
pub fn attach_lora(model: SparkModel, spec: LoraSpec) -> Result<SparkModel, String> {
    let group = lora_group(spec.target)?;
    let lora = Lora::new(spec.rank, spec.alpha).set_param_group(group);
    Ok(model.apply_lora(lora).train())
}

/// Reference policy: a validation copy taken before adapters are attached.
///
/// Optimizer steps replace parameters in the returned module, so this clone
/// keeps the original dense weights.
pub fn reference_policy(model: &SparkModel) -> SparkModel {
    model.clone().valid()
}

pub fn expected_lora_parameters(config: &SparkConfig, spec: &LoraSpec) -> u64 {
    config.lora_parameter_count(spec.rank as u64, spec.include_mlp())
}

/// Read LoRA factors saved by [`save_lora_adapters`] into an already adapted model.
///
/// The base checkpoint is left as it was. Missing base tensors are expected,
/// because the file only contains adapter factors.
pub fn load_lora_adapters(mut model: SparkModel, path: &Path) -> Result<SparkModel, String> {
    let mut store = SafetensorsStore::from_file(path)
        .with_regex(r"\.lora\.(a|b)$")
        .allow_partial(true);
    let result = model
        .load_from(&mut store)
        .map_err(|err| format!("loading {}: {err}", path.display()))?;
    if !result.errors.is_empty() {
        return Err(format!("loading {}: {result}", path.display()));
    }
    if result.applied.is_empty() {
        return Err(format!("{} contains no LoRA factors", path.display()));
    }
    Ok(model)
}

/// Write only the LoRA factors. The frozen Spark checkpoint stays where it was.
pub fn save_lora_adapters(model: &SparkModel, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    let mut store = SafetensorsStore::from_file(path)
        .with_regex(r"\.lora\.(a|b)$")
        .overwrite(true);
    model.save_into(&mut store).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Module;
    use burn::tensor::Device;

    #[test]
    fn attention_lora_skips_embeddings_and_the_mlp() {
        let device = Device::flex().autodiff();
        let model = SparkModel::new(&SparkConfig::tiny(), &device);
        let reference = reference_policy(&model);
        let adapted = attach_lora(model, LoraSpec::attention_default()).unwrap();

        assert!(adapted.embed.weight.adapter().is_none());
        assert!(adapted.layers[0].attn.qkv.weight.adapter().is_some());
        assert!(adapted.layers[0].attn.output.weight.adapter().is_some());
        assert!(adapted.layers[0].attn.gate.weight.adapter().is_none());
        assert!(adapted.layers[0].mlp.gate.weight.adapter().is_none());
        assert!(adapted.layers[0].mlp.down.weight.adapter().is_none());
        assert!(reference.layers[0].attn.qkv.weight.adapter().is_none());
        let _ = adapted.num_params();
    }

    #[test]
    fn saved_adapters_are_only_lora_factors() {
        let device = Device::flex().autodiff();
        let model = SparkModel::new(&SparkConfig::tiny(), &device);
        let adapted = attach_lora(model, LoraSpec::attention_default()).unwrap();
        let path =
            std::env::temp_dir().join(format!("spark-lora-{}.safetensors", std::process::id()));
        let _ = std::fs::remove_file(&path);
        save_lora_adapters(&adapted, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let names = tensors.names();
        assert_eq!(names.len(), 16, "names: {names:?}");
        assert!(names.iter().all(|name| name.contains(".lora.")));
        assert!(names.iter().all(|name| !name.contains("embed")));

        let fresh = SparkModel::new(&SparkConfig::tiny(), &device);
        let loaded = load_lora_adapters(
            attach_lora(fresh, LoraSpec::attention_default()).unwrap(),
            &path,
        )
        .unwrap();
        let saved_a = host_sum(adapted.layers[0].attn.qkv.weight.adapter().unwrap().a.val());
        let loaded_a = host_sum(loaded.layers[0].attn.qkv.weight.adapter().unwrap().a.val());
        assert!(
            (saved_a - loaded_a).abs() < 1e-4,
            "loaded A {loaded_a} does not match saved A {saved_a}"
        );
        let _ = std::fs::remove_file(&path);
    }

    fn host_sum(tensor: burn::tensor::Tensor<2>) -> f32 {
        tensor.into_data().as_slice::<f32>().unwrap().iter().sum()
    }
}
