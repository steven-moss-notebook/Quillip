//! AdamW against an f32 copy of each trainable parameter.
//!
//! The forward weights stay in the checkpoint dtype, which is bfloat16. A
//! step of size `1e-6` is smaller than the bfloat16 ulp of a LoRA A factor
//! (about `5e-4` near `0.06`), so adding it in place is a no-op. The master
//! copy accumulates those steps until the change is large enough to round
//! back into the forward weight.

use std::collections::HashMap;

use burn::module::{Module, ModuleMapper, Param, ParamId};
use burn::optim::GradientsParams;
use burn::tensor::{FloatDType, Tensor, TensorData};

struct FactorMaster {
    values: Vec<f32>,
    moment_1: Vec<f32>,
    moment_2: Vec<f32>,
    step: i32,
}

impl FactorMaster {
    fn new(values: Vec<f32>) -> Self {
        let zeros = vec![0.0; values.len()];
        Self {
            values,
            moment_1: zeros.clone(),
            moment_2: zeros,
            step: 0,
        }
    }

    fn apply(
        &mut self,
        grad: &[f32],
        learning_rate: f32,
        beta_1: f32,
        beta_2: f32,
        epsilon: f32,
        clip_norm: f32,
    ) {
        assert_eq!(
            self.values.len(),
            grad.len(),
            "fp32 master does not match the gradient"
        );
        let scale = clip_scale(grad, clip_norm);
        self.step += 1;
        let bias_1 = 1.0 - beta_1.powi(self.step);
        let bias_2 = 1.0 - beta_2.powi(self.step);
        for ((value, moment_1), (moment_2, grad)) in self
            .values
            .iter_mut()
            .zip(&mut self.moment_1)
            .zip(self.moment_2.iter_mut().zip(grad))
        {
            let grad = *grad * scale;
            *moment_1 = beta_1 * *moment_1 + (1.0 - beta_1) * grad;
            *moment_2 = beta_2 * *moment_2 + (1.0 - beta_2) * grad * grad;
            let corrected_1 = *moment_1 / bias_1;
            let corrected_2 = *moment_2 / bias_2;
            *value -= learning_rate * corrected_1 / (corrected_2.sqrt() + epsilon);
        }
    }
}

fn clip_scale(grad: &[f32], max_norm: f32) -> f32 {
    let mut sum = 0.0_f32;
    for value in grad {
        sum += value * value;
    }
    let norm = sum.sqrt();
    if norm > max_norm && norm.is_finite() {
        max_norm / norm
    } else {
        1.0
    }
}

/// AdamW with f32 moments and an f32 master weight.
///
/// Matches Burn's defaults: `beta1 = 0.9`, `beta2 = 0.999`, `epsilon = 1e-5`,
/// weight decay 0, and a per-tensor norm clip of 1.
pub struct Fp32MasterAdam {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    clip_norm: f32,
    factors: HashMap<ParamId, FactorMaster>,
}

impl Fp32MasterAdam {
    pub fn lora_default() -> Self {
        Self {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-5,
            clip_norm: 1.0,
            factors: HashMap::new(),
        }
    }

    /// Per-tensor L2 clip, matching `GradientClippingConfig::Norm`.
    pub fn with_clip_norm(mut self, clip_norm: f32) -> Self {
        self.clip_norm = clip_norm;
        self
    }

    /// Apply `grads` to the f32 masters and write the rounded weights back.
    pub fn step<M: Module>(&mut self, learning_rate: f64, model: M, grads: GradientsParams) -> M {
        let mut mapper = MasterStep {
            learning_rate: learning_rate as f32,
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            epsilon: self.epsilon,
            clip_norm: self.clip_norm,
            factors: &mut self.factors,
            grads,
        };
        model.map(&mut mapper)
    }
}

struct MasterStep<'a> {
    learning_rate: f32,
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    clip_norm: f32,
    factors: &'a mut HashMap<ParamId, FactorMaster>,
    grads: GradientsParams,
}

impl ModuleMapper for MasterStep<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let Some(grad) = self.grads.remove::<D>(id) else {
            return Param::from_mapped_value(id, tensor, mapper);
        };
        let dims = tensor.dims();
        let dtype = tensor.dtype();
        let device = tensor.device();
        let require_grad = tensor.is_require_grad();
        let grad_values = host_f32(grad);
        if !self.factors.contains_key(&id) {
            self.factors
                .insert(id, FactorMaster::new(host_f32(tensor.clone())));
        }
        self.factors.get_mut(&id).expect("master").apply(
            &grad_values,
            self.learning_rate,
            self.beta_1,
            self.beta_2,
            self.epsilon,
            self.clip_norm,
        );
        let values = self.factors.get(&id).expect("master").values.clone();
        let updated = Tensor::<D>::from_data(TensorData::new(values, dims), &device)
            .cast(dtype)
            .set_require_grad(require_grad);
        Param::from_mapped_value(id, updated, mapper)
    }
}

fn host_f32<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .detach()
        .cast(FloatDType::F32)
        .into_data()
        .as_slice::<f32>()
        .expect("f32 view of a float tensor")
        .to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_steps_accumulate_past_a_bfloat16_ulp() {
        let start = 0.0625_f32;
        let mut once = FactorMaster::new(vec![start]);
        once.apply(&[1.0], 1e-6, 0.9, 0.999, 1e-5, 1.0);
        let one_step = (once.values[0] - start).abs();
        assert!(
            one_step < 5e-4,
            "one 1e-6 step should sit under the bf16 ulp near 0.06, moved {one_step}"
        );

        let mut master = FactorMaster::new(vec![start]);
        for _ in 0..1000 {
            master.apply(&[1.0], 1e-6, 0.9, 0.999, 1e-5, 1.0);
        }
        let moved = (master.values[0] - start).abs();
        assert!(
            moved > 5e-4,
            "a thousand steps must survive where one bf16 add would not, moved {moved}"
        );
    }

    #[test]
    fn a_second_step_still_receives_a_gradient() {
        use crate::{LoraSpec, SparkConfig, SparkModel, attach_lora};
        use burn::tensor::Device;

        let device = Device::flex().autodiff();
        let model = SparkModel::new(&SparkConfig::tiny(), &device);
        let mut policy = attach_lora(model, LoraSpec::attention_default()).unwrap();
        let mut optimizer = Fp32MasterAdam::lora_default();
        let mut previous = 0.0_f32;
        for _ in 0..2 {
            let loss = policy.layers[0]
                .attn
                .qkv
                .weight
                .adapter()
                .unwrap()
                .b
                .val()
                .sum();
            let grads = GradientsParams::from_grads(loss.backward(), &policy);
            assert!(!grads.is_empty(), "the adapter produced no gradient");
            policy = optimizer.step(1e-2, policy, grads);
            let now = policy.layers[0]
                .attn
                .qkv
                .weight
                .adapter()
                .unwrap()
                .b
                .val()
                .into_data()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .map(|value| value.abs())
                .sum::<f32>();
            assert!(
                now > previous + 1e-6,
                "step did not move the adapter ({previous} -> {now})"
            );
            previous = now;
        }
    }
}
