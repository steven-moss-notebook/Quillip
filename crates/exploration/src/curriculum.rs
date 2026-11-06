//! Keep a mixture of easy, mid, and hard challenge families.
//!
//! Families whose observed success sits near 20–80% are sampled most often.
//! Nearly solved and still-failing families stay in the mix at a lower weight
//! so the reward does not collapse to always-zero or always-one.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurriculumStage {
    ShortArc,
    MultiScene,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FamilyStats {
    pub attempts: u32,
    pub successes: u32,
}

impl FamilyStats {
    pub fn record(&mut self, success: bool) {
        self.attempts += 1;
        if success {
            self.successes += 1;
        }
    }

    pub fn rate(&self) -> Option<f32> {
        if self.attempts == 0 {
            None
        } else {
            Some(self.successes as f32 / self.attempts as f32)
        }
    }
}

pub fn sampling_weight(rate: Option<f32>) -> f32 {
    match rate {
        Some(rate) if (0.2..=0.8).contains(&rate) => 1.0,
        Some(_) => 0.35,
        None => 1.0,
    }
}

/// Move to multi-scene tasks only after short arcs are mostly solved and the
/// archive is not a single structure. The caller should keep sampling short
/// arcs afterward.
pub fn advance_to_multiscene(short_arc_success: f32, mean_distinct_structures: f32) -> bool {
    short_arc_success >= 0.8 && mean_distinct_structures >= 2.0
}

pub fn sample_family(weights: &BTreeMap<String, f32>, draw: f32) -> Option<String> {
    let total: f32 = weights.values().sum();
    if total <= 0.0 || weights.is_empty() {
        return None;
    }
    let mut cursor = draw.clamp(0.0, 0.999_999) * total;
    for (family, weight) in weights {
        cursor -= weight;
        if cursor <= 0.0 {
            return Some(family.clone());
        }
    }
    weights.keys().next().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mid_success_families_outweigh_the_extremes() {
        assert_eq!(sampling_weight(Some(0.5)), 1.0);
        assert_eq!(sampling_weight(Some(0.1)), 0.35);
        assert_eq!(sampling_weight(Some(0.95)), 0.35);
        assert!(advance_to_multiscene(0.85, 2.5));
        assert!(!advance_to_multiscene(0.85, 1.0));
        let mut stats = FamilyStats::default();
        stats.record(true);
        stats.record(false);
        assert_eq!(stats.rate(), Some(0.5));
    }

    #[test]
    fn sampler_respects_weights() {
        let weights = BTreeMap::from([
            ("easy".into(), 0.35),
            ("mid".into(), 1.0),
            ("hard".into(), 0.35),
        ]);
        let mut counts = BTreeMap::<String, usize>::new();
        for step in 0..1000 {
            let draw = step as f32 / 1000.0;
            let family = sample_family(&weights, draw).unwrap();
            *counts.entry(family).or_default() += 1;
        }
        assert!(counts["mid"] > counts["easy"]);
        assert!(counts["mid"] > counts["hard"]);
    }
}
