//! The first experiment, run on scripted candidates when no Spark checkpoint
//! is loaded.
//!
//! This measures the verifier, the group advantage, the contrastive probe, and
//! the archive. It does not claim the language model became a better writer.

use std::collections::BTreeSet;

use exploration::{
    NarrativeArchive, features_from_report, intra_story_trajectory, novelty_score, token_set,
};
use narrative_world::{generate_split, scripted_plans, verify_plan};
use rlvr::{
    CpoConfig, GrpoConfig, RewardWeights, cpo_authorized, disagreement_probe, group_advantages,
    score_reward_with_progression, token_advantages, zero_variance,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ArmReport {
    pub verified_pass_rate: f32,
    pub distinct_structures: usize,
    pub human_preference: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Comparison {
    pub cpo_improves_verification: bool,
    pub cpo_keeps_diversity: bool,
    pub human_preference_not_worse: bool,
    /// True only when verification and diversity improved and a human score
    /// was actually recorded. Unmeasured prose is not treated as a win.
    pub extend_environment: bool,
}

pub fn compare_arms(grpo: &ArmReport, cpo: &ArmReport) -> Comparison {
    let cpo_improves_verification = cpo.verified_pass_rate > grpo.verified_pass_rate;
    let cpo_keeps_diversity = cpo.distinct_structures >= grpo.distinct_structures;
    let human_preference_not_worse = match (grpo.human_preference, cpo.human_preference) {
        (Some(before), Some(after)) => after + 1e-4 >= before,
        _ => false,
    };
    Comparison {
        cpo_improves_verification,
        cpo_keeps_diversity,
        human_preference_not_worse,
        extend_environment: cpo_improves_verification
            && cpo_keeps_diversity
            && human_preference_not_worse,
    }
}

#[derive(Debug, Clone)]
pub struct CycleReport {
    pub challenges: usize,
    pub candidates: usize,
    pub pass_rate: f32,
    pub distinct_structures: usize,
    pub mean_reward: f32,
    pub zero_variance_groups: usize,
    pub stuffed_exploration_reward: f32,
    pub cpo_probe_authorized: bool,
    pub inverted_probe_authorized: bool,
}

/// Score self-generated-looking candidates without updating a neural policy.
pub fn run_scripted_cycle(seed: u64, challenges: usize) -> CycleReport {
    let set = generate_split(seed, challenges, 0).expect("challenge generator");
    let weights = RewardWeights::default();
    let grpo = GrpoConfig::default();
    let cpo = CpoConfig::default();
    let mut archive = NarrativeArchive::default();
    let mut passes = 0_usize;
    let mut reward_sum = 0.0_f32;
    let mut candidates = 0_usize;
    let mut zero_variance_groups = 0_usize;
    let mut stuffed_exploration_reward = 0.0_f32;
    let mut seen = BTreeSet::new();

    for challenge in &set.train {
        let plans = scripted_plans(challenge);
        let reports: Vec<_> = plans
            .iter()
            .map(|plan| verify_plan(challenge, plan))
            .collect();
        let rewards: Vec<_> = reports
            .iter()
            .zip(&plans)
            .map(|(report, plan)| {
                let signature = features_from_report(report);
                let prose = token_set(&plan.prose);
                let novelty = novelty_score(&archive, &signature, &prose);
                let progression = intra_story_trajectory(&plan.prose);
                score_reward_with_progression(report, novelty, progression, &weights, 0.5)
            })
            .collect();
        let totals: Vec<f32> = rewards.iter().map(|reward| reward.total).collect();
        if zero_variance(&totals) {
            zero_variance_groups += 1;
        }
        let advantages = group_advantages(&totals, &grpo);
        for advantage in &advantages {
            let _shaped = token_advantages(*advantage, Some(&[0.5, -0.5, 0.0, 0.1]), 4, Some(&cpo));
        }
        for ((report, plan), reward) in reports.iter().zip(&plans).zip(&rewards) {
            candidates += 1;
            reward_sum += reward.total;
            if report.hard_pass {
                passes += 1;
                let signature = features_from_report(report);
                seen.insert(signature.features.clone());
                archive.insert(signature, token_set(&plan.prose));
            }
        }
        stuffed_exploration_reward = rewards[3].exploration;
    }

    let caught = disagreement_probe(&[-0.05, -1.2, -0.1], &[false, true, false]).expect("probe");
    let missed = disagreement_probe(&[0.1, 0.4, 0.0], &[false, true, false]).expect("probe");
    CycleReport {
        challenges: set.train.len(),
        candidates,
        pass_rate: passes as f32 / candidates.max(1) as f32,
        distinct_structures: seen.len(),
        mean_reward: reward_sum / candidates.max(1) as f32,
        zero_variance_groups,
        stuffed_exploration_reward,
        cpo_probe_authorized: cpo_authorized(&caught, 0.5),
        inverted_probe_authorized: cpo_authorized(&missed, 0.5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripted_cycle_keeps_two_structures_and_refuses_stuffing_novelty() {
        let report = run_scripted_cycle(5, 4);
        assert_eq!(report.candidates, 16);
        assert!((report.pass_rate - 0.5).abs() < 1e-5);
        assert!(report.distinct_structures >= 2);
        assert_eq!(report.stuffed_exploration_reward, 0.0);
        assert!(report.cpo_probe_authorized);
        assert!(!report.inverted_probe_authorized);
    }

    #[test]
    fn missing_human_scores_do_not_authorize_a_larger_environment() {
        let grpo = ArmReport {
            verified_pass_rate: 0.4,
            distinct_structures: 3,
            human_preference: None,
        };
        let cpo = ArmReport {
            verified_pass_rate: 0.55,
            distinct_structures: 4,
            human_preference: None,
        };
        let comparison = compare_arms(&grpo, &cpo);
        assert!(comparison.cpo_improves_verification);
        assert!(!comparison.extend_environment);
    }
}
