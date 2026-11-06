//! Group-relative policy optimization with contrastive advantage shaping.
//!
//! CPO is not a second optimizer. It rewrites the token advantages that GRPO
//! would have used. The binary pass/fail bit is the correctness signal. The
//! weighted narrative score stays in the group-relative advantage and is not
//! treated as the paper's binary reward.

mod cpo;
mod grpo;
mod kl;
mod reward;
mod trainer;

pub use cpo::{
    CpoConfig, DisagreementProbe, cpo_authorized, disagreement_probe, shape_advantage,
    token_advantages,
};
pub use grpo::{GrpoConfig, group_advantages, group_centered, zero_variance};
pub use kl::k3_kl;
pub use reward::{Reward, RewardWeights, score_reward, score_reward_with_progression};
pub use trainer::{
    RolloutBudget, TrainHyperparams, accumulate_grads, apply_accumulated, apply_update,
    clipped_policy_loss, clipped_policy_loss_normalized,
};

pub use narrative_world::VerificationReport;
