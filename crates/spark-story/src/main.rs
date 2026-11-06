//! Commands for the continual storytelling experiment.
//!
//! `cycle` scores scripted candidates through the verifier, GRPO, and the CPO
//! probe. `train` loads Spark-X2.6-1.7B, draws format seeds, and updates
//! attention LoRA on the stories the policy writes. `train --tiny` is a small
//! stand-in update. A verifier pass is not a judgment of literary quality.

mod dashboard;
mod download;
mod seeds;
mod separate;
mod train;

use std::fs;
use std::path::PathBuf;

use burn::optim::GradientsParams;
use burn::tensor::{FloatDType, Int, Tensor, TensorData};
use clap::{Parser, Subcommand};
use evaluation::{ArmReport, compare_arms, run_scripted_cycle};
use narrative_world::{ThinkingMode, generate_split};
use rlvr::{RolloutBudget, TrainHyperparams, clipped_policy_loss, k3_kl};
use spark_burn::{
    Fp32MasterAdam, LoraSpec, SparkConfig, SparkModel, accelerator_name, attach_lora, bf16_device,
    compare_logits, reference_policy,
};

#[derive(Parser)]
#[command(name = "spark-story")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the 1.7B shape, LoRA size, and initial hyperparameters.
    Config,
    /// Write procedurally generated challenges. Held-out ids are separate.
    Challenges {
        #[arg(long, default_value_t = 1000)]
        count: usize,
        #[arg(long, default_value_t = 100)]
        heldout: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value = "data/generated/challenges.json")]
        out: PathBuf,
    },
    /// Show verifier results for two valid plans and two failures.
    VerifyDemo,
    /// Run the injected-error disagreement probe on synthetic log probabilities.
    ProbeCpo,
    /// Score a scripted GRPO batch and the contrastive probe. No weight update.
    Cycle {
        #[arg(long, default_value_t = 4)]
        challenges: usize,
        #[arg(long, default_value_t = 5)]
        seed: u64,
    },
    /// Print the initial rollout budget.
    Budget,
    /// Download XHToken/Spark-X2.6-1.7B weights, config, and tokenizer.
    Download {
        #[arg(long, default_value = download::DEFAULT_DIR)]
        out: PathBuf,
    },
    /// Compare two logit dumps. This does not run the reference model.
    Logits {
        actual: PathBuf,
        reference: PathBuf,
        #[arg(long, default_value_t = 1e-3)]
        atol: f32,
    },
    /// Train attention LoRA on Spark-X2.6-1.7B.
    ///
    /// Without `--checkpoint`, this downloads XHToken/Spark-X2.6-1.7B first.
    /// `--tiny` trains the small stand-in instead of the 1.7B checkpoint.
    /// At startup the frozen model writes up to 512 situations, appends the ones
    /// that come back with text beside `--save` (up to 128,000 seeds across runs),
    /// and trains on those. When that file already exists, training uses the new situations
    /// and the saved ones. Each optimizer step draws up to 64 of
    /// them, sixteen times, and samples with the checkpoint's generation config. The policy writes about 1,024 tokens for each seed.
    /// The frozen checkpoint scores every rubric line in one pass and combines
    /// them with a harmonic mean. Stories of one format are ranked together,
    /// and that score is contrasted with the two formats a reader could confuse
    /// with it, each of those formats weighted once. Mini-batches draw the
    /// formats equally. Every mini-batch is recorded on the dashboard. The
    /// backward tape holds four stories at 512 tokens and shrinks for the
    /// longer rows. The highest-scoring story in each mini-batch is appended
    /// beside `--save`. `--seq-len` caps the trained window.
    Train {
        #[arg(long)]
        tiny: bool,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        #[arg(long, default_value_t = 1e-6)]
        lr: f64,
        #[arg(long, default_value_t = train::DEFAULT_SEQ_LEN)]
        seq_len: usize,
        #[arg(
            long,
            default_value = "checkpoints/lora/spark-x2.6-1.7b-attention.safetensors"
        )]
        save: PathBuf,
    },
    /// Plot format-instruction residuals against situation residuals.
    ///
    /// Several prompts share a format instruction and differ in the situation.
    /// The picture is the final residual after the last norm, drawn in the
    /// plane that separates the format groups. Omit `--kind` to draw
    /// every format on one figure. Pass `--lora` to add the shift of each
    /// group mean and a neighborhood map of the tangent displacement.
    InstructionSpace {
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        lora: Option<PathBuf>,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long, default_value_t = 8)]
        situations: usize,
        #[arg(long, default_value = "checkpoints/lora/instruction-separation.svg")]
        out: PathBuf,
    },
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    match Cli::parse().command {
        Command::Config => {
            let config = SparkConfig::x2_6_1_7b();
            let hyper = TrainHyperparams::initial();
            println!("Spark-X2.6-1.7B parameters: {}", config.parameter_count());
            println!(
                "attention layout (local, global): {:?}",
                config.attention_layout()
            );
            println!(
                "LoRA rank {} alpha {} attention-only parameters: {}",
                hyper.lora_rank,
                hyper.lora_alpha,
                config.lora_parameter_count(hyper.lora_rank as u64, false)
            );
            println!(
                "LoRA including MLP parameters: {}",
                config.lora_parameter_count(hyper.lora_rank as u64, true)
            );
            println!(
                "lr {} clip {} cpo alpha {} kl {}",
                hyper.learning_rate, hyper.clip_epsilon, hyper.cpo_alpha, hyper.kl_coefficient
            );
            println!(
                "context {} gelu mlp rms epsilon {}",
                config.max_position_embeddings, config.rms_norm_eps
            );
            println!(
                "sliding rope theta {} partial {} | global rope theta {} partial {}",
                config.sliding_rope.theta,
                config.sliding_rope.partial_rotary_factor,
                config.global_rope.theta,
                config.global_rope.partial_rotary_factor
            );
        }
        Command::Challenges {
            count,
            heldout,
            seed,
            out,
        } => {
            let set = generate_split(seed, count, heldout).map_err(|err| err.to_string())?;
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
            let json = serde_json::to_string_pretty(&set).map_err(|err| err.to_string())?;
            fs::write(&out, json).map_err(|err| err.to_string())?;
            println!(
                "wrote {} train and {} held-out challenges to {}",
                set.train.len(),
                set.held_out.len(),
                out.display()
            );
        }
        Command::VerifyDemo => {
            let set = generate_split(3, 1, 0).map_err(|err| err.to_string())?;
            let challenge = &set.train[0];
            println!(
                "{}",
                narrative_world::render_prompt(challenge, ThinkingMode::High)
            );
            for (index, plan) in narrative_world::scripted_plans(challenge)
                .iter()
                .enumerate()
            {
                let report = narrative_world::verify_plan(challenge, plan);
                println!(
                    "candidate {index}: hard_pass={} binary_pass={} contradictions={}",
                    report.hard_pass, report.binary_pass, report.contradiction_count
                );
            }
        }
        Command::ProbeCpo => {
            let report = run_scripted_cycle(1, 1);
            println!(
                "injected error probe authorized: {}",
                report.cpo_probe_authorized
            );
            println!(
                "inverted probe authorized: {}",
                report.inverted_probe_authorized
            );
            println!(
                "a checkpoint forward pass was not run; this probe uses constructed log probabilities"
            );
        }
        Command::Cycle { challenges, seed } => {
            let report = run_scripted_cycle(seed, challenges);
            println!(
                "challenges {} candidates {} pass_rate {:.3} distinct_structures {} mean_reward {:.3}",
                report.challenges,
                report.candidates,
                report.pass_rate,
                report.distinct_structures,
                report.mean_reward
            );
            println!(
                "zero_variance_groups {} stuffed_exploration {:.3}",
                report.zero_variance_groups, report.stuffed_exploration_reward
            );
            println!(
                "cpo probe {} inverted {}",
                report.cpo_probe_authorized, report.inverted_probe_authorized
            );
            let comparison = compare_arms(
                &ArmReport {
                    verified_pass_rate: 0.40,
                    distinct_structures: report.distinct_structures.saturating_sub(1),
                    human_preference: None,
                },
                &ArmReport {
                    verified_pass_rate: report.pass_rate,
                    distinct_structures: report.distinct_structures,
                    human_preference: None,
                },
            );
            println!(
                "extend environment: {} (human preference was not measured)",
                comparison.extend_environment
            );
        }
        Command::Budget => {
            let budget = RolloutBudget::initial();
            println!(
                "candidates {} output-token budget {} scoring passes {}",
                budget.candidates(),
                budget.output_token_budget(),
                budget.scoring_passes()
            );
        }
        Command::Download { out } => {
            let snapshot = download::download_spark(&out)?;
            println!("snapshot {}", snapshot.directory.display());
            println!("config {}", snapshot.config.display());
            for weight in &snapshot.weights {
                println!("weights {}", weight.display());
            }
        }
        Command::Logits {
            actual,
            reference,
            atol,
        } => {
            let actual = read_f32s(&actual)?;
            let reference = read_f32s(&reference)?;
            let report =
                compare_logits(&actual, &reference, atol).map_err(|err| err.to_string())?;
            println!(
                "max_abs {:.6} mean_abs {:.6} matched {}",
                report.max_abs_diff, report.mean_abs_diff, report.matched
            );
            if !report.matched {
                return Err("logits do not match the reference within tolerance".into());
            }
        }
        Command::Train {
            tiny,
            checkpoint,
            steps,
            lr,
            seq_len,
            save,
        } => {
            if tiny {
                let loss = train_tiny(steps, lr)?;
                println!("tiny LoRA update finished, last loss {loss:.6}");
            } else {
                let checkpoint = match checkpoint {
                    Some(path) => path,
                    None => {
                        let snapshot =
                            download::download_spark(std::path::Path::new(download::DEFAULT_DIR))?;
                        println!("snapshot {}", snapshot.directory.display());
                        snapshot.directory
                    }
                };
                train::train_spark(&checkpoint, steps, lr, seq_len, &save)?;
            }
        }
        Command::InstructionSpace {
            checkpoint,
            lora,
            kind,
            situations,
            out,
        } => {
            let kind = match kind {
                Some(kind) => Some(seeds::SeedKind::from_label(&kind).ok_or_else(|| {
                    format!("unknown format {kind}; expected scene, dialogue, monologue, character-background, or group-conversation")
                })?),
                None => None,
            };
            let checkpoint = match checkpoint {
                Some(path) => path,
                None => {
                    let snapshot =
                        download::download_spark(std::path::Path::new(download::DEFAULT_DIR))?;
                    println!("snapshot {}", snapshot.directory.display());
                    snapshot.directory
                }
            };
            separate::instruction_space(&checkpoint, lora.as_deref(), kind, situations, &out)?;
        }
    }
    Ok(())
}

fn mean(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|value| f64::from(*value)).sum::<f64>() / values.len() as f64
}

fn read_f32s(path: &PathBuf) -> Result<Vec<f32>, String> {
    let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
    serde_json::from_str(&text).map_err(|err| err.to_string())
}

fn host_f32<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .cast(FloatDType::F32)
        .into_data()
        .as_slice::<f32>()
        .expect("f32 view of a float tensor")
        .to_vec()
}

fn train_tiny(steps: usize, learning_rate: f64) -> Result<f32, String> {
    let device = bf16_device(true)?;
    println!(
        "training device {} dtype {:?}",
        accelerator_name(),
        device.settings().float_dtype
    );
    let config = SparkConfig::tiny();
    let model = SparkModel::new(&config, &device);
    let reference = reference_policy(&model);
    let mut policy = attach_lora(model, LoraSpec::attention_default())?;
    let adapter_before = adapter_abs(&policy);
    let mut optimizer = Fp32MasterAdam::lora_default();
    let tokens =
        Tensor::<2, Int>::from_data(TensorData::new(vec![1_i32, 2, 3, 4], [1, 4]), &device);
    let _terminal = dashboard::TerminalGuard::arm();
    let mut dashboard = dashboard::TrainingDashboard::open(steps);
    let mut last = 0.0;
    let mut behavior = None;
    let mut frozen = None;
    for step in 1..=steps {
        if dashboard.stopped() {
            break;
        }
        let old = policy.aligned_logprobs(tokens.clone()).detach();
        let reference_logprobs = reference.aligned_logprobs(tokens.clone()).detach();
        behavior = Some(old.clone());
        frozen = Some(reference_logprobs.clone());
        let new = policy.aligned_logprobs(tokens.clone());
        let new_values = host_f32(new.clone().detach());
        let reference_values = host_f32(reference_logprobs.clone());
        let advantages = Tensor::<2>::from_data(
            TensorData::new(vec![0.0_f32, 1.0, 1.0, 1.0], [1, 4]),
            &device,
        );
        let mask = Tensor::<2>::from_data(
            TensorData::new(vec![0.0_f32, 1.0, 1.0, 1.0], [1, 4]),
            &device,
        );
        let loss = clipped_policy_loss(new, old, reference_logprobs, advantages, mask, 0.2, 0.001);
        last = host_f32(loss.clone())[0];
        if !last.is_finite() {
            return Err("policy loss was not finite".into());
        }
        let mut logprob = 0.0_f64;
        let mut kl = 0.0_f64;
        let mut counted = 0.0_f64;
        for (index, (new_logprob, reference_logprob)) in
            new_values.iter().zip(&reference_values).enumerate()
        {
            if index == 0 {
                continue;
            }
            logprob += f64::from(*new_logprob);
            kl += f64::from(k3_kl(*reference_logprob, *new_logprob));
            counted += 1.0;
        }
        let denom = counted.max(1.0);
        let metrics = dashboard::StepMetrics {
            loss: f64::from(last),
            reward: 1.0,
            rubric: 0.8,
            raw_rubric: 0.8,
            progression: 0.75,
            prog_penalty: 0.0,
            story_len: 4.0,
            diversity: 1.0,
            advantage: 1.0,
            centered: 1.0,
            kl: kl / denom,
            logprob: logprob / denom,
            learning_rate,
            stories: 1,
        };
        dashboard.record(step, &metrics)?;
        let grads = GradientsParams::from_grads(loss.backward(), &policy);
        policy = optimizer.step(learning_rate, policy, grads);
    }
    let adapter_after = adapter_abs(&policy);
    let advantages = Tensor::<2>::from_data(
        TensorData::new(vec![0.0_f32, 1.0, 1.0, 1.0], [1, 4]),
        &device,
    );
    let behavior = behavior.ok_or("no training step ran")?;
    let updated = policy.aligned_logprobs(tokens);
    let ratio = mean(&host_f32((updated.clone() - behavior.clone()).exp()));
    let changed_loss = host_f32(clipped_policy_loss(
        updated,
        behavior,
        frozen.ok_or("no training step ran")?,
        advantages.clone(),
        advantages,
        0.2,
        0.001,
    ))[0];
    let summary = dashboard.summary_report();
    let closed = dashboard.close();
    println!("\n{summary}");
    println!(
        "pre-update loss {last:.6}; after the step, loss {changed_loss:.6}, mean probability ratio {ratio:.6}, adapter shift {:.8}",
        adapter_after - adapter_before
    );
    closed?;
    Ok(changed_loss)
}

fn adapter_abs(model: &SparkModel) -> f32 {
    host_f32(model.layers[0].attn.qkv.weight.adapter().unwrap().b.val())
        .iter()
        .map(|value| value.abs())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(tensor: burn::tensor::Tensor<2>) -> f32 {
        tensor.into_data().as_slice::<f32>().unwrap().iter().sum()
    }

    #[test]
    fn tiny_step_trains_attention_adapters_and_leaves_embeddings_fixed() {
        let device = burn::tensor::Device::flex().autodiff();
        let model = SparkModel::new(&SparkConfig::tiny(), &device);
        let reference = reference_policy(&model);
        let policy = attach_lora(model, LoraSpec::attention_default()).unwrap();
        let embed_before = flat(policy.embed.weight.val());
        let adapter_before = flat(policy.layers[0].attn.qkv.weight.adapter().unwrap().b.val());
        assert!(adapter_before.abs() < 1e-8);
        let mut optimizer = Fp32MasterAdam::lora_default();
        let tokens =
            Tensor::<2, Int>::from_data(TensorData::new(vec![1_i32, 2, 3, 4], [1, 4]), &device);
        let old = policy.aligned_logprobs(tokens.clone()).detach();
        let reference_logprobs = reference.aligned_logprobs(tokens.clone()).detach();
        let new = policy.aligned_logprobs(tokens);
        let advantages = Tensor::<2>::from_data(
            TensorData::new(vec![0.0_f32, 1.0, 1.0, 1.0], [1, 4]),
            &device,
        );
        let mask = advantages.clone();
        let loss = clipped_policy_loss(new, old, reference_logprobs, advantages, mask, 0.2, 0.0);
        let grads = GradientsParams::from_grads(loss.backward(), &policy);
        let updated = optimizer.step(1e-2, policy, grads);
        let embed_after = flat(updated.embed.weight.val());
        let adapter_after = flat(updated.layers[0].attn.qkv.weight.adapter().unwrap().b.val());
        assert!((embed_before - embed_after).abs() < 1e-5);
        assert!(
            (adapter_after - adapter_before).abs() > 1e-8,
            "adapter B did not move"
        );
    }
}
