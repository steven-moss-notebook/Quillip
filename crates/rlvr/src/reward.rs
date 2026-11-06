//! Layered reward.
//!
//! The weights are experimental starting values, not fitted coefficients.
//! Exploration is zero unless the tier-1 verifier passed, and it is capped so
//! a strange but invalid story cannot outscore a coherent one. Tier-3 literary
//! judgment is intentionally absent: it belongs in held-out human review.

use serde::{Deserialize, Serialize};

use narrative_world::VerificationReport;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RewardWeights {
    pub consistency: f32,
    pub causality: f32,
    pub completion: f32,
    pub structure: f32,
    pub exploration: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            consistency: 0.30,
            causality: 0.25,
            completion: 0.20,
            structure: 0.15,
            exploration: 0.10,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reward {
    pub total: f32,
    pub binary_pass: bool,
    pub hard_pass: bool,
    pub exploration: f32,
    pub progression: f32,
}

pub fn score_reward(
    report: &VerificationReport,
    novelty: f32,
    weights: &RewardWeights,
    novelty_cap: f32,
) -> Reward {
    score_reward_with_progression(report, novelty, 1.0, weights, novelty_cap)
}

pub fn score_reward_with_progression(
    report: &VerificationReport,
    novelty: f32,
    progression: f32,
    weights: &RewardWeights,
    novelty_cap: f32,
) -> Reward {
    let exploration = if report.hard_pass {
        let factor = (0.35 + 0.65 * progression.clamp(0.0, 1.0)).min(1.0);
        (novelty * factor).clamp(0.0, novelty_cap)
    } else {
        0.0
    };
    let total = weights.consistency * report.consistency
        + weights.causality * report.causality
        + weights.completion * report.completion
        + weights.structure * report.structure
        + weights.exploration * exploration;
    Reward {
        total,
        binary_pass: report.binary_pass,
        hard_pass: report.hard_pass,
        exploration,
        progression,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(hard_pass: bool) -> VerificationReport {
        VerificationReport {
            parsed: true,
            consistency: 1.0,
            causality: 1.0,
            completion: 1.0,
            structure: 1.0,
            tier2_confidence: 1.0,
            tier2_uncertain: false,
            hard_pass,
            binary_pass: hard_pass,
            violations: Vec::new(),
            contradiction_count: 0,
            protagonist_belief_changed: false,
            other_belief_changed: false,
            reveal_order: Vec::new(),
            resolved: Vec::new(),
            acknowledged: Vec::new(),
            affected: Vec::new(),
        }
    }

    #[test]
    fn novelty_is_ignored_when_verification_fails_and_capped_when_it_passes() {
        let weights = RewardWeights::default();
        let failed = score_reward(&report(false), 1.0, &weights, 0.5);
        assert_eq!(failed.exploration, 0.0);
        let passed = score_reward(&report(true), 1.0, &weights, 0.5);
        assert_eq!(passed.exploration, 0.5);
        assert!(passed.total > failed.total);
    }

    #[test]
    fn stagnant_progression_reduces_exploration_reward() {
        let weights = RewardWeights::default();
        let report = report(true);
        let evolving = score_reward_with_progression(&report, 1.0, 1.0, &weights, 0.5);
        let circular = score_reward_with_progression(&report, 1.0, 0.1, &weights, 0.5);
        assert!(evolving.exploration > circular.exploration);
        assert!(evolving.total > circular.total);
    }
}
