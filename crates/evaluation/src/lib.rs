//! Evaluation that is not allowed to write training rewards.
//!
//! The held-out set is immutable. Promotion looks at pass rate, contradictions,
//! structural diversity, KL, and human preference when a person has actually
//! scored the stories. Missing human scores do not count as a prose win.

mod benchmark;
mod comparison;
mod consistency;

pub use benchmark::{EvalSnapshot, HeldOutSet, PromotionDecision, PromotionLimits, promote};
pub use comparison::{ArmReport, Comparison, CycleReport, compare_arms, run_scripted_cycle};
pub use consistency::adversarial_cases;
