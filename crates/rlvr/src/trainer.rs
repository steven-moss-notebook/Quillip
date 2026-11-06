//! One policy update from stored log probabilities.
//!
//! The old-policy and reference log probabilities are detached. The new-policy
//! log probabilities must still be on the autodiff tape. Callers score the
//! posterior with the frozen rollout snapshot before this update; this function
//! does not rescore anything.

use burn::module::Module;
use burn::optim::{GradientsAccumulator, GradientsParams, ModuleOptimizer};
use burn::tensor::{FloatDType, Tensor};

/// Largest log-ratio passed to `exp`. `exp(20)` stays finite in f32; a raw
/// gap past ~88 nats overflows, and `inf * 0` on a masked position is NaN.
const LOG_RATIO_CLAMP: f32 = 20.0;

#[derive(Debug, Clone, Copy)]
pub struct RolloutBudget {
    pub prompts: usize,
    pub group_size: usize,
    pub max_new_tokens: usize,
}

impl RolloutBudget {
    pub fn initial() -> Self {
        Self {
            prompts: 8,
            group_size: 4,
            max_new_tokens: 1_024,
        }
    }

    pub fn candidates(&self) -> usize {
        self.prompts * self.group_size
    }

    pub fn output_token_budget(&self) -> usize {
        self.candidates() * self.max_new_tokens
    }

    /// Prior scoring plus posterior rescoring. Reference scoring is separate.
    pub fn scoring_passes(&self) -> usize {
        self.candidates() * 2
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrainHyperparams {
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub group_size: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub clip_epsilon: f32,
    pub cpo_alpha: f32,
    pub kl_coefficient: f32,
    pub learning_rate: f64,
    pub grad_clip_norm: f32,
    pub max_new_tokens: usize,
    pub max_context: usize,
}

impl TrainHyperparams {
    pub fn initial() -> Self {
        Self {
            lora_rank: 16,
            lora_alpha: 32.0,
            group_size: 4,
            temperature: 0.9,
            top_p: 0.95,
            clip_epsilon: 0.2,
            cpo_alpha: 0.01,
            kl_coefficient: 0.001,
            learning_rate: 1e-6,
            grad_clip_norm: 1.0,
            max_new_tokens: 1_024,
            max_context: 4_096,
        }
    }
}

fn policy_loss_values(
    new_logprobs: Tensor<2>,
    old_logprobs: Tensor<2>,
    reference_logprobs: Tensor<2>,
    advantages: Tensor<2>,
    mask: Tensor<2>,
    clip_epsilon: f32,
    kl_coefficient: f32,
) -> Tensor<2> {
    // The ratio and k3 both call `exp`. That has to happen in f32, and only on
    // positions the mask keeps: a padded log-prob gap overflows, and `inf * 0`
    // is NaN, which wipes the whole backward.
    let new_logprobs = new_logprobs.cast(FloatDType::F32);
    let old_logprobs = old_logprobs.detach().cast(FloatDType::F32);
    let reference_logprobs = reference_logprobs.detach().cast(FloatDType::F32);
    let advantages = advantages.cast(FloatDType::F32);
    let mask = mask.cast(FloatDType::F32);
    let log_ratio = ((new_logprobs.clone() - old_logprobs) * mask.clone())
        .clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP);
    let ratio = log_ratio.exp();
    let unclipped = ratio.clone() * advantages.clone();
    let clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages;
    let policy = unclipped.min_pair(clipped).mul_scalar(-1.0);
    let delta = ((reference_logprobs - new_logprobs) * mask.clone())
        .clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP);
    let kl = delta.clone().exp() - delta - 1.0;
    (policy + kl.mul_scalar(kl_coefficient)) * mask
}

/// One loss per story: the mean of its masked tokens.
///
/// A global token mean lets a long completion own the step. Dividing by each
/// story's own token count, then averaging stories, gives every story the same
/// weight.
fn per_story_loss(values: Tensor<2>, mask: Tensor<2>) -> Tensor<2> {
    let counts = mask.cast(FloatDType::F32).sum_dim(1).clamp_min(1e-8);
    values.sum_dim(1) / counts
}

pub fn clipped_policy_loss(
    new_logprobs: Tensor<2>,
    old_logprobs: Tensor<2>,
    reference_logprobs: Tensor<2>,
    advantages: Tensor<2>,
    mask: Tensor<2>,
    clip_epsilon: f32,
    kl_coefficient: f32,
) -> Tensor<1> {
    let values = policy_loss_values(
        new_logprobs,
        old_logprobs,
        reference_logprobs,
        advantages,
        mask.clone(),
        clip_epsilon,
        kl_coefficient,
    );
    per_story_loss(values, mask).mean()
}

/// Same objective as [`clipped_policy_loss`], divided by a fixed story count.
///
/// A chunked backward passes the number of stories in the whole mini-batch.
/// The chunk gradients then add up to the mean over those stories, not over tokens.
pub fn clipped_policy_loss_normalized(
    new_logprobs: Tensor<2>,
    old_logprobs: Tensor<2>,
    reference_logprobs: Tensor<2>,
    advantages: Tensor<2>,
    mask: Tensor<2>,
    clip_epsilon: f32,
    kl_coefficient: f32,
    stories: f32,
) -> Tensor<1> {
    let values = policy_loss_values(
        new_logprobs,
        old_logprobs,
        reference_logprobs,
        advantages,
        mask.clone(),
        clip_epsilon,
        kl_coefficient,
    );
    per_story_loss(values, mask).sum().div_scalar(stories)
}

pub fn apply_update<M: Module>(
    model: M,
    optimizer: &mut ModuleOptimizer,
    loss: Tensor<1>,
    learning_rate: f64,
) -> M {
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    optimizer.step(learning_rate, model, grads)
}

/// Backward one mini-batch and add its gradients. The optimizer is not stepped.
pub fn accumulate_grads<M: Module>(
    model: &M,
    accumulator: &mut GradientsAccumulator<M>,
    loss: Tensor<1>,
) {
    let grads = GradientsParams::from_grads(loss.backward(), model);
    accumulator.accumulate(model, grads);
}

/// Apply the summed mini-batch gradients and clear the accumulator.
pub fn apply_accumulated<M: Module>(
    model: M,
    optimizer: &mut ModuleOptimizer,
    accumulator: &mut GradientsAccumulator<M>,
    learning_rate: f64,
) -> M {
    optimizer.step(learning_rate, model, accumulator.grads())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::k3_kl;
    use burn::tensor::{Device, TensorData};

    #[test]
    fn budget_matches_the_initial_experiment() {
        let budget = RolloutBudget::initial();
        assert_eq!(budget.candidates(), 32);
        assert_eq!(budget.output_token_budget(), 32_768);
        assert_eq!(budget.scoring_passes(), 64);
    }

    #[test]
    fn clipped_loss_matches_the_token_formula() {
        let device = Device::flex();
        let new = Tensor::<2>::from_data(TensorData::new(vec![0.1_f32, 0.0], [1, 2]), &device);
        let old = Tensor::<2>::from_data(TensorData::new(vec![0.0_f32, 0.0], [1, 2]), &device);
        let reference = old.clone();
        let advantages =
            Tensor::<2>::from_data(TensorData::new(vec![1.0_f32, 1.0], [1, 2]), &device);
        let mask = Tensor::<2>::from_data(TensorData::new(vec![1.0_f32, 1.0], [1, 2]), &device);
        let loss = clipped_policy_loss(new, old, reference, advantages, mask, 0.2, 0.0);
        let value = loss.into_data().as_slice::<f32>().unwrap()[0];

        let ratio0 = (0.1_f32).exp();
        let expected = -(ratio0.min(1.2) + 1.0) / 2.0;
        assert!(
            (value - expected).abs() < 1e-4,
            "loss {value} expected {expected}"
        );
        assert!(k3_kl(0.0, 0.0).abs() < 1e-6);
    }

    #[test]
    fn chunk_losses_sum_to_the_full_batch_loss() {
        let device = Device::flex();
        let new = Tensor::<2>::from_data(
            TensorData::new(vec![0.1_f32, 0.0, -0.2, 0.3], [2, 2]),
            &device,
        );
        let old = Tensor::<2>::from_data(
            TensorData::new(vec![0.0_f32, 0.0, 0.0, 0.1], [2, 2]),
            &device,
        );
        let reference = old.clone();
        let advantages = Tensor::<2>::from_data(
            TensorData::new(vec![1.0_f32, -1.0, 0.5, 0.25], [2, 2]),
            &device,
        );
        let mask = Tensor::<2>::from_data(
            TensorData::new(vec![0.0_f32, 1.0, 1.0, 1.0], [2, 2]),
            &device,
        );
        let full = host_scalar(clipped_policy_loss(
            new.clone(),
            old.clone(),
            reference.clone(),
            advantages.clone(),
            mask.clone(),
            0.2,
            0.001,
        ));
        // Two stories, not three masked tokens. Each story is one unit.
        let stories = 2.0;
        let first = host_scalar(clipped_policy_loss_normalized(
            new.clone().slice([0..1, 0..2]),
            old.clone().slice([0..1, 0..2]),
            reference.clone().slice([0..1, 0..2]),
            advantages.clone().slice([0..1, 0..2]),
            mask.clone().slice([0..1, 0..2]),
            0.2,
            0.001,
            stories,
        ));
        let second = host_scalar(clipped_policy_loss_normalized(
            new.slice([1..2, 0..2]),
            old.slice([1..2, 0..2]),
            reference.slice([1..2, 0..2]),
            advantages.slice([1..2, 0..2]),
            mask.slice([1..2, 0..2]),
            0.2,
            0.001,
            stories,
        ));
        assert!(
            (full - (first + second)).abs() < 1e-5,
            "full {full} chunks {}",
            first + second
        );
    }

    #[test]
    fn masked_log_ratio_cannot_overflow_and_small_kl_stays_positive() {
        let device = Device::flex();
        let new = Tensor::<2>::from_data(TensorData::new(vec![-1.0_f32, 100.0], [1, 2]), &device);
        let old = Tensor::<2>::from_data(TensorData::new(vec![-1.0_f32, -100.0], [1, 2]), &device);
        let reference =
            Tensor::<2>::from_data(TensorData::new(vec![-1.02_f32, 0.0], [1, 2]), &device);
        let advantages =
            Tensor::<2>::from_data(TensorData::new(vec![0.0_f32, 1.0], [1, 2]), &device);
        let mask = Tensor::<2>::from_data(TensorData::new(vec![1.0_f32, 0.0], [1, 2]), &device);
        let loss = host_scalar(clipped_policy_loss(
            new, old, reference, advantages, mask, 0.2, 1.0,
        ));
        let expected = k3_kl(-1.02, -1.0);
        assert!(
            expected > 1e-5 && expected < 1e-3,
            "a hundredth of a nat must stay a small positive penalty, got {expected}"
        );
        assert!(
            loss.is_finite() && (loss - expected).abs() < 1e-5,
            "loss {loss} expected {expected}"
        );
    }

    fn host_scalar(tensor: Tensor<1>) -> f32 {
        tensor.into_data().as_slice::<f32>().unwrap()[0]
    }
}
