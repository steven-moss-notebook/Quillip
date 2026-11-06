//! Quillip: plan a short story, write it, then hear it.
//!
//! Spark-X2.6-1.7B with the attention LoRA drafts the plan and each beat.
//! Kokoro narrates with `af_nicole`, and each character gets a different voice.

mod audio;
mod engine;
mod fetch;
mod narrate;
mod plan;
mod prompts;
mod qa;
mod ui;

use std::path::PathBuf;

use clap::Parser;

use crate::engine::EngineConfig;

#[derive(Parser)]
#[command(name = "quillip")]
struct Args {
    /// Directory or weights file for XHToken/Spark-X2.6-1.7B.
    #[arg(long, default_value = "checkpoints/XHToken/Spark-X2.6-1.7B")]
    checkpoint: PathBuf,
    /// New tokens for the plan.
    #[arg(long, default_value_t = 640)]
    plan_tokens: usize,
    /// New tokens for each beat.
    #[arg(long, default_value_t = 384)]
    beat_tokens: usize,
    /// New tokens for the coherence pass.
    #[arg(long, default_value_t = 1536)]
    qa_tokens: usize,
}

fn main() {
    let args = Args::parse();
    ui::run(EngineConfig {
        checkpoint: args.checkpoint,
        plan_tokens: args.plan_tokens,
        beat_tokens: args.beat_tokens,
        qa_tokens: args.qa_tokens,
    });
}
