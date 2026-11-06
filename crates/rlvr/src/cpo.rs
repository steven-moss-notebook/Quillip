//! Token-level advantage shaping from posterior-prior disagreement.
//!
//! For a nonzero group advantage the adjustment cannot cross zero, so a local
//! contrast cannot reverse the trajectory's success or failure. A zero-advantage
//! group, which ordinary GRPO cannot learn from, takes a clipped contrastive
//! advantage instead.
//!
//! This is the behavior the training plan specifies. It is not a claim that the
//! July 2026 paper's equation numbering was copied line for line. The posterior
//! here is conditioned on the verifier trace, which the paper lists as future
//! work for tasks without a reference answer. Callers should withhold this
//! signal until `disagreement_probe` shows that injected narrative errors get
//! more negative disagreement than the surrounding tokens.

#[derive(Debug, Clone, Copy)]
pub struct CpoConfig {
    pub alpha: f32,
    pub zero_advantage_clip: f32,
    pub adjustment_clip: f32,
}

impl Default for CpoConfig {
    fn default() -> Self {
        Self {
            alpha: 0.01,
            zero_advantage_clip: 0.2,
            adjustment_clip: 1.0,
        }
    }
}

impl CpoConfig {
    pub fn paper_starting_point() -> Self {
        Self {
            alpha: 0.025,
            zero_advantage_clip: 0.2,
            adjustment_clip: 1.0,
        }
    }
}

pub fn shape_advantage(advantage: f32, delta: f32, config: &CpoConfig) -> f32 {
    let adjustment = (config.alpha * delta).clamp(-config.adjustment_clip, config.adjustment_clip);
    if advantage > 0.0 {
        (advantage + adjustment).max(0.0)
    } else if advantage < 0.0 {
        (advantage + adjustment).min(0.0)
    } else {
        adjustment.clamp(-config.zero_advantage_clip, config.zero_advantage_clip)
    }
}

/// Per-token advantages. Without deltas, every token keeps the group advantage.
pub fn token_advantages(
    group_advantage: f32,
    deltas: Option<&[f32]>,
    length: usize,
    config: Option<&CpoConfig>,
) -> Vec<f32> {
    match (deltas, config) {
        (Some(deltas), Some(config)) => deltas
            .iter()
            .map(|delta| shape_advantage(group_advantage, *delta, config))
            .collect(),
        _ => vec![group_advantage; length],
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisagreementProbe {
    pub error_mean: f32,
    pub context_mean: f32,
    pub error_more_negative: bool,
    pub margin: f32,
}

/// Compare contrastive disagreement on injected error tokens with the rest.
pub fn disagreement_probe(
    deltas: &[f32],
    error_mask: &[bool],
) -> Result<DisagreementProbe, String> {
    if deltas.len() != error_mask.len() || deltas.is_empty() {
        return Err("delta and error mask lengths must match and be non-empty".into());
    }
    let mut error_sum = 0.0;
    let mut error_count = 0_usize;
    let mut context_sum = 0.0;
    let mut context_count = 0_usize;
    for (delta, is_error) in deltas.iter().zip(error_mask) {
        if *is_error {
            error_sum += delta;
            error_count += 1;
        } else {
            context_sum += delta;
            context_count += 1;
        }
    }
    if error_count == 0 || context_count == 0 {
        return Err("probe needs both error tokens and context tokens".into());
    }
    let error_mean = error_sum / error_count as f32;
    let context_mean = context_sum / context_count as f32;
    let margin = context_mean - error_mean;
    Ok(DisagreementProbe {
        error_mean,
        context_mean,
        error_more_negative: error_mean < context_mean,
        margin,
    })
}

pub fn cpo_authorized(probe: &DisagreementProbe, minimum_margin: f32) -> bool {
    probe.error_more_negative && probe.margin >= minimum_margin
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaping_preserves_sign_and_fills_zero_advantage_groups() {
        let config = CpoConfig {
            alpha: 1.0,
            zero_advantage_clip: 0.2,
            adjustment_clip: 1.0,
        };
        assert_eq!(shape_advantage(1.0, -100.0, &config), 0.0);
        assert_eq!(shape_advantage(-0.5, 100.0, &config), 0.0);
        assert!(shape_advantage(1.0, 0.2, &config) > 1.0);
        let flat = token_advantages(0.0, Some(&[2.0, -4.0]), 2, Some(&config));
        assert_eq!(flat, vec![0.2, -0.2]);
        let grpo_only = token_advantages(0.0, None, 2, None);
        assert_eq!(grpo_only, vec![0.0, 0.0]);
    }

    #[test]
    fn probe_authorizes_only_when_errors_are_more_negative() {
        let caught = disagreement_probe(&[-0.1, -1.5, -0.2], &[false, true, false]).unwrap();
        assert!(cpo_authorized(&caught, 0.5));
        let missed = disagreement_probe(&[0.2, 0.4, 0.1], &[false, true, false]).unwrap();
        assert!(!cpo_authorized(&missed, 0.5));
    }
}
