use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeldOutSet {
    pub ids: BTreeSet<String>,
}

impl HeldOutSet {
    pub fn from_ids(ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            ids: ids.into_iter().map(Into::into).collect(),
        }
    }

    /// Training may not consume a held-out id, whatever its scores were.
    pub fn allows_training(&self, id: &str) -> bool {
        !self.ids.contains(id) && !id.starts_with("heldout-")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EvalSnapshot {
    pub verified_pass_rate: f32,
    pub mean_contradictions: f32,
    pub distinct_structures: usize,
    pub mean_kl: Option<f32>,
    pub human_preference: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct PromotionLimits {
    pub max_pass_drop: f32,
    pub max_kl: f32,
}

impl Default for PromotionLimits {
    fn default() -> Self {
        Self {
            max_pass_drop: 0.02,
            max_kl: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionDecision {
    pub accept: bool,
    pub reasons: Vec<String>,
}

pub fn promote(
    baseline: &EvalSnapshot,
    candidate: &EvalSnapshot,
    limits: &PromotionLimits,
) -> PromotionDecision {
    let mut reasons = Vec::new();
    if candidate.verified_pass_rate + limits.max_pass_drop < baseline.verified_pass_rate {
        reasons.push("held-out pass rate regressed".into());
    }
    if candidate.mean_contradictions > baseline.mean_contradictions + 1e-4 {
        reasons.push("contradictions increased".into());
    }
    if candidate.distinct_structures < baseline.distinct_structures {
        reasons.push("fewer distinct verified structures".into());
    }
    if let Some(kl) = candidate.mean_kl {
        if kl > limits.max_kl {
            reasons.push("KL from the original policy exceeded the bound".into());
        }
    }
    match (baseline.human_preference, candidate.human_preference) {
        (Some(before), Some(after)) if after + 1e-4 < before => {
            reasons.push("blind human preference declined".into());
        }
        (Some(_), None) => reasons.push("human preference was not remeasured".into()),
        _ => {}
    }
    PromotionDecision {
        accept: reasons.is_empty(),
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> EvalSnapshot {
        EvalSnapshot {
            verified_pass_rate: 0.6,
            mean_contradictions: 0.4,
            distinct_structures: 5,
            mean_kl: Some(0.05),
            human_preference: Some(0.7),
        }
    }

    #[test]
    fn held_out_ids_never_enter_training() {
        let held = HeldOutSet::from_ids(["heldout-last_letter-0"]);
        assert!(!held.allows_training("heldout-last_letter-0"));
        assert!(!held.allows_training("heldout-other-1"));
        assert!(held.allows_training("train-last_letter-1"));
    }

    #[test]
    fn promotion_rejects_reward_hacking_signals() {
        let baseline = snapshot();
        let mut worse = snapshot();
        worse.verified_pass_rate = 0.9;
        worse.human_preference = Some(0.4);
        let decision = promote(&baseline, &worse, &PromotionLimits::default());
        assert!(!decision.accept);
        assert!(
            decision
                .reasons
                .iter()
                .any(|reason| reason.contains("human preference"))
        );

        let mut drifted = snapshot();
        drifted.mean_kl = Some(2.0);
        assert!(!promote(&baseline, &drifted, &PromotionLimits::default()).accept);
    }
}
