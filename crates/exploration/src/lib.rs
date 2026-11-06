//! Diversity pressure that does not replace the verifier.
//!
//! A story earns novelty only after the caller has already decided it passed
//! tier 1. The archive stores structural features of those passing stories,
//! not a taxonomy the model is required to hit.

mod archive;
mod curriculum;
mod novelty;

pub use archive::{NarrativeArchive, NarrativeSignature, features_from_report};
pub use curriculum::{
    CurriculumStage, FamilyStats, advance_to_multiscene, sample_family, sampling_weight,
};
pub use novelty::{
    intra_story_trajectory, lexical_diversity, novelty_score, progressive_novelty_score, token_set,
};
