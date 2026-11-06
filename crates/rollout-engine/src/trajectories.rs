use serde::{Deserialize, Serialize};

/// One sampled story and the log probabilities recorded before the update.
///
/// `old_logprobs` and `posterior_logprobs` are aligned to the completion, not
/// to the prompt. The posterior was scored by the same frozen policy that
/// produced `old_logprobs`, conditioned on verifier guidance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    pub challenge_id: String,
    pub prompt: String,
    pub completion: String,
    pub token_ids: Vec<u32>,
    pub prompt_len: usize,
    pub old_logprobs: Vec<f32>,
    pub posterior_logprobs: Vec<f32>,
    pub reference_logprobs: Vec<f32>,
}

impl Trajectory {
    pub fn contrastive_deltas(&self) -> Vec<f32> {
        self.posterior_logprobs
            .iter()
            .zip(&self.old_logprobs)
            .map(|(posterior, prior)| posterior - prior)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltas_are_posterior_minus_prior() {
        let trajectory = Trajectory {
            challenge_id: "train-last_letter-0".into(),
            prompt: String::new(),
            completion: String::new(),
            token_ids: vec![1, 2],
            prompt_len: 1,
            old_logprobs: vec![-1.0, -0.5],
            posterior_logprobs: vec![-1.2, -0.1],
            reference_logprobs: vec![-1.0, -0.5],
        };
        let deltas = trajectory.contrastive_deltas();
        assert!((deltas[0] + 0.2).abs() < 1e-5);
        assert!((deltas[1] - 0.4).abs() < 1e-5);
    }
}
