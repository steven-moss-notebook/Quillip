//! Self-play narrative environment.
//!
//! The generator builds challenges from coherent scenario templates. The
//! verifier checks invariants of the resulting state trace. It does not score
//! whether an ending is moving, original, or well written.

mod characters;
mod events;
mod generator;
mod state;
mod verifier;

pub use characters::{Character, Relationship};
pub use events::{Effect, Ending, NarrativePlan, PlanEvent, Transformation};
pub use generator::{ChallengeSet, GeneratorError, generate_split};
pub use state::{Constraint, Fact, Genre, NarrativeState, StoryChallenge, ThinkingMode};
pub use verifier::{
    Guidance, VerificationReport, Violation, parse_model_output, render_prompt, scripted_plans,
    verify_output, verify_plan,
};
