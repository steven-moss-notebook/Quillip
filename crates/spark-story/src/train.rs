//! One GRPO step on the published Spark checkpoint.
//!
//! Each mini-batch draws up to 64 seeds from the situations written at startup.
//! The writer aims for 512 new situations each run and writes each one to the
//! seed file as soon as it is kept, continuing across runs until reaching 128,000 seeds.
//! When the seed file already exists, the bank is the new situations plus the saved ones. The seeds ask for a
//! scene, a monologue, a dialogue, a character background, or several people
//! talking. The policy writes 1,024 tokens from the seed. The frozen
//! checkpoint, with no LoRA, scores that text on the attributes of the
//! requested format. Stories of one format are ranked together. The negative
//! is the mean score of the two formats a reader could confuse with it, each
//! of those formats counted once, so a good description still loses to a
//! dialogue when the seed asked for dialogue. Mini-batches take each format
//! equally and do not repeat a seed until that format has been used through.
//! Gradients accumulate for 16 mini-batches, and the optimizer steps once on
//! that full batch. The highest-scoring story in each mini-batch is appended
//! beside the saved adapters.
//!
//! The backward tape holds four stories at a time near 512 tokens. A 64-story
//! tape measured about 52GB. Longer rows shrink that slice with the square of
//! the sequence length, because the attention scores dominate. A 1,024-token
//! story is one row on the tape.

use std::io::Write;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::optim::GradientsAccumulator;
use burn::tensor::{FloatDType, Int, Tensor, TensorData};
use rayon::prelude::*;
use rlvr::{
    GrpoConfig, TrainHyperparams, accumulate_grads, clipped_policy_loss,
    clipped_policy_loss_normalized, group_advantages, group_centered, k3_kl,
};
use rollout_engine::{
    SamplingConfig, SparkTokenizer, Tokenizer, generate_batch, load_story_sampling,
};
use spark_burn::{
    Fp32MasterAdam, LoraSpec, SparkConfig, SparkModel, accelerator_name, attach_lora, bf16_device,
    expected_lora_parameters, load_spark_checkpoint, reference_policy, save_lora_adapters,
    sidecar_dir,
};

use crate::dashboard;
use crate::seeds::{
    self, GENERATED_TOKENS, MAX_SEEDS, SEED_BANK, SEED_NEW_TOKENS, Seed, SeedKind, SeedStream,
    contrast_advantages, contrast_baselines_from_own, contrast_centered, contrast_reward, criteria,
    criteria_with_atmosphere, rubric_preamble, softened_rubric_mean,
};

/// Longest window passed to the 1.7B update: the seed prompt plus the generated story.
pub const DEFAULT_SEQ_LEN: usize = 1536;

/// Stories scored together and recorded as one dashboard point.
pub const STORIES_PER_MINI_BATCH: usize = 64;

/// Stories whose backward tape is live at once, at a 512-token row.
///
/// Sixty-four stories used about 52GB. The two checkpoints are about 8GB of
/// that, and the rest scales with the rows on the tape. Four rows is
/// `8 + 44 * 4/64`, about 11GB.
const TAPE_STORIES: usize = 4;
const TAPE_SEQ: usize = 512;

/// Stories prefetched and decoded together.
///
/// A 1,536-token cache is a few dozen megabytes per story, under the memory
/// the one-row backward tape already occupies. Generation is dropped before
/// that tape is built.
pub const ROLLOUT_BATCH: usize = 16;

/// Stories scored together against rubrics or reference log-probabilities.
pub const EVAL_BATCH: usize = 16;

/// How many stories to backprop together for a row of `seq` tokens.
fn backward_width(seq: usize) -> usize {
    let growth = (seq.max(1) as f64 / TAPE_SEQ as f64).powi(2).max(1.0);
    let stories = (TAPE_STORIES as f64 / growth).floor() as usize;
    stories.clamp(1, STORIES_PER_MINI_BATCH)
}

/// Mini-batches summed before one AdamW step.
pub const ACCUMULATION_STEPS: usize = 16;

pub fn highest_reward(rewards: &[f32]) -> Option<f32> {
    rewards.iter().copied().reduce(f32::max)
}

/// Prose from a plan, with terminal control characters removed.
pub fn terminal_text(text: &str) -> String {
    text.chars()
        .filter(|character| *character == '\n' || *character == '\t' || !character.is_control())
        .collect()
}

fn story_text(completion: &str) -> String {
    serde_json::from_str::<serde_json::Value>(completion)
        .ok()
        .and_then(|value| {
            value
                .get("prose")
                .and_then(|prose| prose.as_str())
                .map(str::to_string)
        })
        .filter(|prose| !prose.trim().is_empty())
        .unwrap_or_else(|| completion.to_string())
}

fn stories_log_path(save: &Path) -> PathBuf {
    let stem = save
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("spark");
    save.with_file_name(format!("{stem}-stories.txt"))
}

fn metrics_log_path(save: &Path) -> PathBuf {
    let stem = save
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("spark");
    save.with_file_name(format!("{stem}-metrics.jsonl"))
}

fn seeds_path(save: &Path) -> PathBuf {
    let stem = save
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("spark");
    save.with_file_name(format!("{stem}-seeds.jsonl"))
}

/// Draw for one situation. `draw_offset` moves the stream forward by a whole
/// run so a later startup does not repeat situations already written.
fn situation_draw(kind: SeedKind, number: usize, draw_offset: usize) -> u64 {
    ((kind.ordinal() as u64) << 32)
        .wrapping_add(draw_offset as u64)
        .wrapping_add(number as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn join_seed_bank(generated: Vec<Seed>, saved: Vec<Seed>) -> Vec<Seed> {
    let mut bank = generated;
    bank.extend(saved);
    bank
}

fn load_saved_seeds(path: &Path) -> Result<Vec<Seed>, String> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("reading {}: {err}", path.display()))?;
    let mut seeds = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        seeds.push(
            parse_seed_line(line)
                .map_err(|err| format!("{} line {}: {err}", path.display(), number + 1))?,
        );
    }
    Ok(seeds)
}

fn parse_seed_line(line: &str) -> Result<Seed, String> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|err| format!("seed record is not json: {err}"))?;
    let index = value
        .get("index")
        .and_then(|index| index.as_u64())
        .ok_or("seed record is missing index")? as usize;
    let label = value
        .get("kind")
        .and_then(|kind| kind.as_str())
        .ok_or("seed record is missing kind")?;
    let kind = SeedKind::from_label(label)
        .ok_or_else(|| format!("seed record has unknown kind {label}"))?;
    let prompt = value
        .get("prompt")
        .and_then(|prompt| prompt.as_str())
        .ok_or("seed record is missing prompt")?
        .to_string();
    let atmosphere = value
        .get("atmosphere")
        .and_then(|a| a.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| seeds::atmosphere_for(kind, index).to_string());
    Ok(Seed {
        index,
        kind,
        prompt,
        atmosphere,
    })
}

fn open_seed_log(path: &Path) -> Result<std::fs::File, String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("opening {}: {err}", path.display()))
}

fn write_seed(out: &mut impl Write, seed: &Seed) -> Result<(), String> {
    let line = serde_json::to_string(&serde_json::json!({
        "index": seed.index,
        "kind": seed.kind.label(),
        "prompt": seed.prompt,
        "atmosphere": seed.atmosphere,
    }))
    .map_err(|err| err.to_string())?;
    writeln!(out, "{line}").map_err(|err| err.to_string())?;
    out.flush().map_err(|err| err.to_string())
}

fn append_seeds(path: &Path, seeds: &[Seed]) -> Result<(), String> {
    if seeds.is_empty() {
        return Ok(());
    }
    let mut file = open_seed_log(path)?;
    for seed in seeds {
        write_seed(&mut file, seed)?;
    }
    Ok(())
}

fn open_story_log(path: &Path) -> Result<std::fs::File, String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    std::fs::File::create(path).map_err(|err| format!("creating {}: {err}", path.display()))
}

fn open_metrics_log(path: &Path) -> Result<std::fs::File, String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|err| format!("creating {}: {err}", path.display()))
}

fn append_metrics_json(
    out: &mut impl Write,
    step: usize,
    micro: usize,
    recorded: usize,
    metrics: &dashboard::StepMetrics,
) -> Result<(), String> {
    let line = serde_json::json!({
        "step": step,
        "micro": micro,
        "slice": recorded,
        "loss": metrics.loss,
        "reward": metrics.reward,
        "rubric": metrics.rubric,
        "raw_rubric": metrics.raw_rubric,
        "progression": metrics.progression,
        "prog_penalty": metrics.prog_penalty,
        "story_len": metrics.story_len,
        "diversity": metrics.diversity,
        "advantage": metrics.advantage,
        "centered": metrics.centered,
        "kl": metrics.kl,
        "logprob": metrics.logprob,
        "learning_rate": metrics.learning_rate,
        "stories": metrics.stories,
    });
    writeln!(out, "{line}").map_err(|err| err.to_string())?;
    out.flush().map_err(|err| err.to_string())
}

fn append_batch_story(
    out: &mut impl Write,
    step: usize,
    batch: usize,
    kind: &str,
    reward: f32,
    rubric: f32,
    raw_rubric: f32,
    progression: f32,
    story_len: usize,
    diversity: f32,
    story: &str,
) -> Result<(), String> {
    let prose = terminal_text(&story_text(story));
    let penalty = (raw_rubric - rubric).max(0.0);
    writeln!(
        out,
        "step {step} mini-batch {batch}  {kind}  reward {reward:.4}  rubric {rubric:.4} (raw {raw_rubric:.4}, penalty {penalty:.4})  prog {progression:.4}  len {story_len} tok  div {diversity:.4}"
    )
    .map_err(|err| err.to_string())?;
    writeln!(out, "{prose}").map_err(|err| err.to_string())?;
    writeln!(out).map_err(|err| err.to_string())?;
    out.flush().map_err(|err| err.to_string())
}

pub struct PolicyBatch {
    pub tokens: Vec<i32>,
    pub advantages: Vec<f32>,
    pub mask: Vec<f32>,
    /// Sampler log probability of each token. Prompt and padding positions are 0.
    pub old_logprobs: Vec<f32>,
    pub batch: usize,
    pub seq: usize,
    /// Unpadded token count of each row. The rectangular buffers pad to `seq`.
    pub lengths: Vec<usize>,
    /// Frozen-policy log probabilities, rectangular like `tokens`. Filled
    /// before the backward. Pad positions stay zero.
    pub reference_logprobs: Vec<f32>,
    pub rewards: Vec<f32>,
    /// Within-format score minus the format mean, before `1/std`.
    pub centered: Vec<f32>,
    /// Frozen rubric score of each story, before the contrast margin.
    pub rubric: Vec<f32>,
    /// Raw harmonic rubric score before progression penalty.
    pub raw_rubric: Vec<f32>,
    /// Thematic progression score across the passage.
    pub progression: Vec<f32>,
    /// Story completion length in tokens.
    pub story_lengths: Vec<usize>,
    /// Lexical diversity (type-token ratio).
    pub diversity: Vec<f32>,
    pub stories: Vec<String>,
}

impl PolicyBatch {
    fn best_index(&self) -> Option<usize> {
        self.rewards
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
    }

    fn best_indices_by_kind(&self, kinds: &[SeedKind]) -> Vec<usize> {
        let mut best: std::collections::BTreeMap<SeedKind, (usize, f32)> =
            std::collections::BTreeMap::new();
        for (index, (&kind, &reward)) in kinds.iter().zip(&self.rewards).enumerate() {
            best.entry(kind)
                .and_modify(|(best_idx, best_reward)| {
                    if reward > *best_reward {
                        *best_idx = index;
                        *best_reward = reward;
                    }
                })
                .or_insert((index, reward));
        }
        best.into_values().map(|(idx, _)| idx).collect()
    }

    fn best_story(&self) -> Option<(f32, &str)> {
        let index = self.best_index()?;
        let story = self.stories.get(index)?;
        Some((self.rewards[index], story.as_str()))
    }
}

pub fn policy_batch(
    tokenizer: &impl Tokenizer,
    prompt: &str,
    completions: &[String],
    rewards: &[f32],
    seq_len: usize,
    pad_id: i32,
) -> Result<PolicyBatch, String> {
    if completions.is_empty() || completions.len() != rewards.len() {
        return Err("each completion needs one reward".into());
    }
    if seq_len < 2 {
        return Err("sequence length must be at least 2".into());
    }
    let prompt_ids = tokenizer.encode(prompt);
    let advantages = group_advantages(rewards, &GrpoConfig::default());
    let centered = group_centered(rewards);
    let mut rows = Vec::with_capacity(completions.len());
    let mut story_lengths = Vec::with_capacity(completions.len());
    let mut diversities = Vec::with_capacity(completions.len());
    for (completion, advantage) in completions.iter().zip(advantages) {
        let completion_ids = tokenizer.encode(completion);
        if completion_ids.is_empty() {
            return Err("a completion encoded to no tokens".into());
        }
        story_lengths.push(completion_ids.len());
        diversities.push(exploration::lexical_diversity(completion));
        let (tokens, mask) = fit_window(&prompt_ids, &completion_ids, seq_len);
        if !mask.iter().any(|value| *value > 0.0) {
            return Err("sequence length left no completion token to train".into());
        }
        // Window tests do not sample, so there is no behavior log probability to store.
        let old_logprobs = vec![0.0; tokens.len()];
        rows.push((tokens, mask, old_logprobs, advantage));
    }
    let raw_rubric = rewards.to_vec();
    let progression = vec![0.5; rewards.len()];
    Ok(pack_rows(
        &rows,
        rewards,
        &centered,
        rewards,
        &raw_rubric,
        &progression,
        &story_lengths,
        &diversities,
        completions.iter().map(|text| story_text(text)).collect(),
        pad_id,
    ))
}

fn fit_window(prompt: &[u32], completion: &[u32], seq_len: usize) -> (Vec<i32>, Vec<f32>) {
    let prompt_floor = if prompt.is_empty() {
        0
    } else {
        1.min(seq_len.saturating_sub(1))
    };
    let completion_take = completion.len().min(seq_len - prompt_floor);
    let prompt_take = prompt.len().min(seq_len - completion_take);
    let prompt_start = prompt.len() - prompt_take;
    let mut tokens = Vec::with_capacity(prompt_take + completion_take);
    let mut mask = Vec::with_capacity(prompt_take + completion_take);
    for id in &prompt[prompt_start..] {
        tokens.push(*id as i32);
        mask.push(0.0);
    }
    for id in completion.iter().take(completion_take) {
        tokens.push(*id as i32);
        mask.push(1.0);
    }
    (tokens, mask)
}

pub fn train_spark(
    checkpoint: &Path,
    steps: usize,
    learning_rate: f64,
    seq_len: usize,
    save: &Path,
) -> Result<(), String> {
    let config = SparkConfig::load_sidecar(checkpoint)?;
    let directory = sidecar_dir(checkpoint);
    let tokenizer = SparkTokenizer::open(&directory.join("tokenizer.json"))?;
    let sampling = load_story_sampling(&directory, GENERATED_TOKENS)?;
    let rubrics = rubric_tokens(&tokenizer)?;
    println!(
        "allocating {} parameters in bfloat16 from {}",
        config.parameter_count(),
        checkpoint.display()
    );
    let device = bf16_device(true)?;
    println!(
        "training device {accelerator} dtype {:?} | temperature {} top_p {} top_k {} repetition {} | {SEED_BANK} new model-written seeds written as they are generated, mini-batch {STORIES_PER_MINI_BATCH} | stop at eos, cap {GENERATED_TOKENS} tokens | backward tape {TAPE_STORIES} stories at {TAPE_SEQ} tokens, one story when the row is longer | accumulate {ACCUMULATION_STEPS} steps",
        device.settings().float_dtype,
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        sampling.repetition_penalty,
        accelerator = accelerator_name(),
    );
    let model = SparkModel::new(&config, &device);
    let model = load_spark_checkpoint(model, checkpoint).map_err(|err| err.to_string())?;
    println!("loaded {}", model.num_params());
    let spec = LoraSpec::attention_default();
    let reference = reference_policy(&model);
    let seeds_file = seeds_path(save);
    let saved = load_saved_seeds(&seeds_file)?;
    if seeds_file.is_file() {
        println!(
            "loaded {} saved seeds from {}",
            saved.len(),
            seeds_file.display()
        );
    }
    let (bank, new_count, saved_count) = if saved.len() >= MAX_SEEDS {
        println!(
            "reusing {} existing seeds from {} (>= {MAX_SEEDS}), skipping seed generation",
            saved.len(),
            seeds_file.display()
        );
        let count = saved.len();
        (saved, 0, count)
    } else {
        let draw_offset = saved.len().saturating_mul(SEED_BANK);
        let index_offset = saved
            .iter()
            .map(|seed| seed.index)
            .max()
            .map(|index| index.saturating_add(1))
            .unwrap_or(0);
        let mut seeds_out = open_seed_log(&seeds_file)?;
        println!(
            "writing new seeds to {} as they are generated",
            seeds_file.display()
        );
        let generated = generate_seed_bank(
            &reference,
            &tokenizer,
            &sampling,
            draw_offset,
            index_offset,
            &mut seeds_out,
        )?;
        println!(
            "wrote {} seeds to {}",
            generated.len(),
            seeds_file.display()
        );
        let new_count = generated.len();
        let saved_count = saved.len();
        let bank = join_seed_bank(generated, saved);
        (bank, new_count, saved_count)
    };
    if bank.len() < 2 {
        return Err(format!(
            "seed bank has {} situations ({} new, {} saved), need at least two to contrast",
            bank.len(),
            new_count,
            saved_count
        ));
    }
    println!(
        "seed bank ready: {} situations ({} new, {} saved)",
        bank.len(),
        new_count,
        saved_count
    );
    let mut policy = attach_lora(model, spec)?;
    println!(
        "trainable attention LoRA parameters {}",
        expected_lora_parameters(&config, &spec)
    );
    let embed_before = tensor_prefix(policy.embed.weight.val());
    let base_before = tensor_prefix(policy.layers[0].attn.qkv.weight.base());
    let adapter_before = adapter_abs(&policy);
    let hyper = TrainHyperparams::initial();
    let mut optimizer = Fp32MasterAdam::lora_default().with_clip_norm(hyper.grad_clip_norm);
    let stories_per_batch = STORIES_PER_MINI_BATCH.min(bank.len());
    let chunks_per_batch = stories_per_batch.div_ceil(backward_width(seq_len));
    let total_batches = steps
        .saturating_mul(ACCUMULATION_STEPS)
        .saturating_mul(chunks_per_batch);
    let stories_path = stories_log_path(save);
    let metrics_path = metrics_log_path(save);
    println!(
        "top stories of each format in each mini-batch will be appended to {}",
        stories_path.display()
    );
    println!("step metrics will be logged to {}", metrics_path.display());
    let mut stories = open_story_log(&stories_path)?;
    let mut metrics_file = open_metrics_log(&metrics_path)?;
    let _terminal = dashboard::TerminalGuard::arm();
    let mut dashboard = dashboard::TrainingDashboard::open(total_batches);
    let mut accumulator = GradientsAccumulator::new();
    let mut last = 0.0;
    let mut recorded = 0usize;
    let mut seed_stream = SeedStream::new(&bank, 0xA11CE);
    for step in 1..=steps {
        if dashboard.stopped() {
            break;
        }
        let mut loss_total = 0.0;
        let mut scored = 0usize;
        for micro in 0..ACCUMULATION_STEPS {
            if dashboard.stopped() {
                break;
            }
            let seed = (step as u64) * ACCUMULATION_STEPS as u64 + micro as u64;
            // `valid()` shares the weight storage, so this snapshot is not a
            // second copy of the checkpoint. Drop it before the backward so
            // the tape is the memory peak.
            let rollout = policy.clone().valid();
            let mut prepared = roll_mini_batch(
                &rollout,
                &reference,
                &tokenizer,
                &rubrics,
                &sampling,
                &bank,
                &mut seed_stream,
                seed,
                seq_len,
                config.pad_token_id as i32,
            )?;
            prepared.batch.reference_logprobs =
                precompute_reference(&reference, &prepared.batch, config.pad_token_id as i32)?;
            drop(rollout);
            for index in prepared.batch.best_indices_by_kind(&prepared.kinds) {
                let kind = prepared
                    .kinds
                    .get(index)
                    .map(|kind| kind.label())
                    .unwrap_or("story");
                let reward = prepared.batch.rewards[index];
                let rubric = prepared.batch.rubric[index];
                let raw_rubric = prepared.batch.raw_rubric[index];
                let progression = prepared.batch.progression[index];
                let story_len = prepared.batch.story_lengths[index];
                let diversity = prepared.batch.diversity[index];
                let story = &prepared.batch.stories[index];
                append_batch_story(
                    &mut stories,
                    step,
                    micro + 1,
                    kind,
                    reward,
                    rubric,
                    raw_rubric,
                    progression,
                    story_len,
                    diversity,
                    story,
                )?;
            }
            scored += prepared.batch.rewards.len();
            let width = backward_width(prepared.batch.seq);
            let stories = prepared.batch.batch as f32;
            let mut value = 0.0f32;
            let mut row = 0usize;
            while row < prepared.batch.batch {
                let rows = width.min(prepared.batch.batch - row);
                let packed = pack_chunk(&prepared.batch, row, rows, config.pad_token_id as i32);
                let seq = packed.seq;
                let adv_mag = {
                    let mut sum_adv = 0.0f32;
                    let mut count_adv = 0.0f32;
                    for (i, m) in packed.mask.iter().enumerate() {
                        if *m > 0.0 {
                            sum_adv += packed.advantages[i].abs();
                            count_adv += 1.0;
                        }
                    }
                    if count_adv > 0.0 {
                        sum_adv / count_adv
                    } else {
                        0.0
                    }
                };
                let tokens = Tensor::<2, Int>::from_data(
                    TensorData::new(packed.tokens, [rows, seq]),
                    &device,
                );
                let advantages = Tensor::<2>::from_data(
                    TensorData::new(packed.advantages, [rows, seq]),
                    &device,
                );
                let mask = Tensor::<2>::from_data(
                    TensorData::new(packed.mask.clone(), [rows, seq]),
                    &device,
                );
                let old = Tensor::<2>::from_data(
                    TensorData::new(packed.old_logprobs, [rows, seq]),
                    &device,
                );
                // The reference log probabilities were scored with no tape
                // while the batch was prepared. `old` is the sampler log
                // probability, not a detach of this forward.
                let reference_logprobs = Tensor::<2>::from_data(
                    TensorData::new(packed.reference_logprobs.clone(), [rows, seq]),
                    &device,
                );
                let new = policy.aligned_logprobs(tokens);
                let loss = clipped_policy_loss_normalized(
                    new.clone(),
                    old,
                    reference_logprobs.clone(),
                    advantages,
                    mask,
                    hyper.clip_epsilon,
                    hyper.kl_coefficient,
                    stories,
                )
                .div_scalar(ACCUMULATION_STEPS as f32);
                let loss_for_grad = loss.clone();
                let loss_scalar = loss.detach().reshape([1]);
                let new_flat = new.detach().flatten(0, 1);
                let combined = Tensor::cat(vec![loss_scalar, new_flat], 0);
                accumulate_grads(&policy, &mut accumulator, loss_for_grad);

                let combined_host = host_f32(combined);
                let chunk = combined_host[0];
                if !chunk.is_finite() {
                    return Err("policy loss was not finite".into());
                }
                value += chunk;
                let new_values = &combined_host[1..];
                let reference_values = packed.reference_logprobs;
                let mask_host = packed.mask;
                let sums = masked_token_sums(new_values, &reference_values, &mask_host);
                // Each backward is its own chart point. The host sums are the
                // masked means, so a 4-story slice is not scaled down to 0.
                let denom = sums.tokens.max(1.0);
                let slice_rewards = &prepared.batch.rewards[row..row + rows];
                let slice_rubric = &prepared.batch.rubric[row..row + rows];
                let slice_raw = &prepared.batch.raw_rubric[row..row + rows];
                let slice_prog = &prepared.batch.progression[row..row + rows];
                let slice_lens = &prepared.batch.story_lengths[row..row + rows];
                let slice_div = &prepared.batch.diversity[row..row + rows];
                let slice_centered = &prepared.batch.centered[row..row + rows];

                let avg_reward = slice_rewards.iter().sum::<f32>() / rows as f32;
                let avg_rubric = slice_rubric.iter().sum::<f32>() / rows as f32;
                let avg_raw = slice_raw.iter().sum::<f32>() / rows as f32;
                let avg_prog = slice_prog.iter().sum::<f32>() / rows as f32;
                let avg_len = slice_lens.iter().sum::<usize>() as f32 / rows as f32;
                let avg_div = slice_div.iter().sum::<f32>() / rows as f32;
                let avg_centered = slice_centered.iter().sum::<f32>() / rows as f32;

                let step_metrics = dashboard::StepMetrics {
                    loss: f64::from(chunk),
                    reward: f64::from(avg_reward),
                    rubric: f64::from(avg_rubric),
                    raw_rubric: f64::from(avg_raw),
                    progression: f64::from(avg_prog),
                    prog_penalty: f64::from((avg_raw - avg_rubric).max(0.0)),
                    story_len: f64::from(avg_len),
                    diversity: f64::from(avg_div),
                    advantage: f64::from(adv_mag),
                    centered: f64::from(avg_centered),
                    kl: sums.kl / denom,
                    logprob: sums.logprob / denom,
                    learning_rate,
                    stories: rows,
                };

                recorded += 1;
                dashboard.record(recorded, &step_metrics)?;
                append_metrics_json(&mut metrics_file, step, micro + 1, recorded, &step_metrics)?;
                row += rows;
            }
            if !value.is_finite() {
                return Err("policy loss was not finite".into());
            }
            loss_total += value;
        }
        if scored == 0 {
            break;
        }
        last = loss_total;
        policy = optimizer.step(learning_rate, policy, accumulator.grads());
    }
    let outcome = (|| {
        let embed_after = tensor_prefix(policy.embed.weight.val());
        let base_after = tensor_prefix(policy.layers[0].attn.qkv.weight.base());
        let adapter_after = adapter_abs(&policy);
        if (embed_before - embed_after).abs() > 1e-3 || (base_before - base_after).abs() > 1e-3 {
            return Err("a frozen Spark weight changed during the LoRA update".into());
        }
        if (adapter_after - adapter_before).abs() <= 1e-8 {
            return Err(format!(
                "LoRA adapter did not move (before {adapter_before:.8}, after {adapter_after:.8}, loss {last:.6})"
            ));
        }
        save_lora_adapters(&policy, save)?;
        Ok((last, adapter_after - adapter_before))
    })();
    let summary = dashboard.summary_report();
    let closed = dashboard.close();
    println!("\n{summary}");
    if let Ok((loss, shift)) = &outcome {
        println!(
            "spark LoRA update finished, last loss {loss:.6}, adapter shift {shift:.6}, saved {}",
            save.display()
        );
    }
    println!("top stories are in {}", stories_path.display());
    println!("step metrics logged to {}", metrics_path.display());
    closed?;
    outcome.map(|_| ())
}

struct PreparedStep {
    batch: PolicyBatch,
    kinds: Vec<SeedKind>,
}

struct KindRubric {
    suffix: Vec<u32>,
    positions: Vec<usize>,
}

struct RubricTokens {
    digits: [usize; 6],
    kinds: Vec<KindRubric>,
    atmospheres: Vec<Vec<KindRubric>>,
}

struct RolloutRow {
    prompt: Vec<u32>,
    completion: Vec<u32>,
    sampler_logprobs: Vec<f32>,
    kind: SeedKind,
    atmosphere: String,
    text: String,
}

const SEED_GEN_BATCH: usize = 16;

fn generate_seed_bank(
    reference: &SparkModel,
    tokenizer: &SparkTokenizer,
    sampling: &SamplingConfig,
    draw_offset: usize,
    index_offset: usize,
    out: &mut impl Write,
) -> Result<Vec<Seed>, String> {
    println!(
        "writing up to {SEED_BANK} new situations with the published sampler in batches of {SEED_GEN_BATCH} (cap {SEED_NEW_TOKENS} tokens)"
    );
    let mut seed_sampling = *sampling;
    seed_sampling.max_new_tokens = SEED_NEW_TOKENS;
    let mut bank = Vec::with_capacity(SEED_BANK);
    let mut skipped = 0_usize;
    let mut attempt = 0_usize;
    for kind in SeedKind::ALL {
        let quota = seeds::kind_quota(kind);

        let mut number = 0;
        while number < quota {
            let batch_size = SEED_GEN_BATCH.min(quota - number);
            let batch_items: Result<Vec<_>, String> = (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    let current_index = number + i;
                    let seed_index = index_offset.saturating_add(bank.len()).saturating_add(i);
                    let request = seeds::seed_request_guided(kind, seed_index);
                    let prompt = tokenizer.try_encode(&request)?;
                    if prompt.is_empty() {
                        return Err("a seed request encoded to no tokens".into());
                    }
                    let len = prompt.len();
                    let draw = situation_draw(kind, current_index, draw_offset);
                    Ok((len, prompt, draw))
                })
                .collect();
            let batch_items = batch_items?;
            let mut prompts = Vec::with_capacity(batch_size);
            let mut prompt_lens = Vec::with_capacity(batch_size);
            let mut draws = Vec::with_capacity(batch_size);
            for (len, prompt, draw) in batch_items {
                prompt_lens.push(len);
                prompts.push(prompt);
                draws.push(draw);
            }
            attempt += batch_size;

            let mut texts: Vec<Option<String>> = vec![None; batch_size];
            for try_idx in 0..2 {
                let mut retry_indices = Vec::new();
                let mut retry_prompts = Vec::new();
                let mut retry_draws = Vec::new();
                for (slot, draw) in draws.iter().enumerate() {
                    if texts[slot].is_none() {
                        retry_indices.push(slot);
                        retry_prompts.push(prompts[slot].clone());
                        retry_draws.push(draw.wrapping_add(try_idx as u64));
                    }
                }
                if retry_indices.is_empty() {
                    break;
                }
                let results =
                    generate_batch(reference, &retry_prompts, seed_sampling, &retry_draws);
                let decoded: Vec<Option<String>> = retry_indices
                    .par_iter()
                    .zip(&results)
                    .map(|(&pos, generated)| {
                        let prompt_len = prompt_lens[pos];
                        if generated.tokens.len() > prompt_len {
                            let text = complete_paragraph(
                                &tokenizer.decode(&generated.tokens[prompt_len..]),
                            );
                            if !text.is_empty() {
                                return Some(text);
                            }
                        }
                        None
                    })
                    .collect();
                for (pos, text_opt) in retry_indices.into_iter().zip(decoded) {
                    if let Some(text) = text_opt {
                        texts[pos] = Some(text);
                    }
                }
            }

            for situation in texts {
                let Some(situation) = situation else {
                    skipped += 1;
                    continue;
                };
                let index = index_offset.saturating_add(bank.len());
                let seed = seeds::seed(index, kind, situation);
                write_seed(out, &seed)?;
                bank.push(seed);
                if bank.len().is_multiple_of(32) {
                    println!("seeds {} kept, attempt {attempt}/{SEED_BANK}", bank.len());
                }
            }
            number += batch_size;
        }
    }
    println!("wrote {} new situations, {skipped} skipped", bank.len());
    Ok(bank)
}

/// Drop a trailing fragment when the length cap stops mid-sentence.
///
/// Text that already ends a sentence is kept. Text with no sentence boundary
/// is kept too. Cutting it would throw away the only paragraph the model wrote.
fn complete_paragraph(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() || ends_sentence(trimmed) {
        return trimmed.to_string();
    }
    let end = trimmed
        .char_indices()
        .rev()
        .find_map(|(index, ch)| matches!(ch, '.' | '!' | '?').then_some(index));
    match end {
        Some(index) => trimmed[..=index].trim().to_string(),
        None => trimmed.to_string(),
    }
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end()
        .trim_end_matches(['"', '\'', '\u{201d}', '\u{2019}', ')', ']'])
        .chars()
        .next_back()
        .is_some_and(|ch| matches!(ch, '.' | '!' | '?'))
}

fn rubric_tokens(tokenizer: &SparkTokenizer) -> Result<RubricTokens, String> {
    let mut digits = [0_usize; 6];
    for digit in 0..6 {
        digits[digit] = digit_token(tokenizer, digit)? as usize;
    }
    let kinds = SeedKind::ALL
        .into_iter()
        .map(|kind| rubric_layout(tokenizer, kind))
        .collect::<Result<Vec<_>, _>>()?;
    let mut atmospheres = Vec::with_capacity(SeedKind::ALL.len());
    for kind in SeedKind::ALL {
        let mut kind_atmospheres = Vec::with_capacity(seeds::INTENTIONAL_ATMOSPHERES.len());
        for atmosphere in seeds::INTENTIONAL_ATMOSPHERES {
            kind_atmospheres.push(rubric_layout_with_atmosphere(tokenizer, kind, atmosphere)?);
        }
        atmospheres.push(kind_atmospheres);
    }
    Ok(RubricTokens {
        digits,
        kinds,
        atmospheres,
    })
}

fn digit_token(tokenizer: &SparkTokenizer, digit: usize) -> Result<u32, String> {
    let word = digit.to_string();
    let full = tokenizer.try_encode(&format!(":{word}"))?;
    let stem = tokenizer.try_encode(":")?;
    if full.len() == stem.len() + 1 && full.starts_with(&stem) {
        return Ok(full[stem.len()]);
    }
    for candidate in [format!(" {word}"), word.clone()] {
        let ids = tokenizer.try_encode(&candidate)?;
        if ids.len() == 1 {
            return Ok(ids[0]);
        }
    }
    Err(format!(
        "the rubric needs digit {word} as one token, got {full:?}"
    ))
}

fn rubric_layout(tokenizer: &SparkTokenizer, kind: SeedKind) -> Result<KindRubric, String> {
    let preamble = rubric_preamble();
    let mut text = preamble.to_string();
    let mut positions = Vec::new();
    for line in criteria(kind) {
        text.push('\n');
        text.push_str(line);
        text.push(':');
        let encoded = tokenizer.try_encode(&text)?;
        if encoded.is_empty() {
            return Err("a rubric line encoded to no tokens".into());
        }
        positions.push(encoded.len() - 1);
    }
    let suffix = tokenizer.try_encode(&text)?;
    if positions.len() != criteria(kind).len()
        || positions.last().copied() != Some(suffix.len() - 1)
    {
        return Err("rubric score positions do not match the encoded lines".into());
    }
    Ok(KindRubric { suffix, positions })
}

fn rubric_layout_with_atmosphere(
    tokenizer: &SparkTokenizer,
    kind: SeedKind,
    atmosphere: &str,
) -> Result<KindRubric, String> {
    let preamble = rubric_preamble();
    let mut text = preamble.to_string();
    let mut positions = Vec::new();
    let crit = criteria_with_atmosphere(kind, atmosphere);
    for line in &crit {
        text.push('\n');
        text.push_str(line);
        text.push(':');
        let encoded = tokenizer.try_encode(&text)?;
        if encoded.is_empty() {
            return Err("a rubric line encoded to no tokens".into());
        }
        positions.push(encoded.len() - 1);
    }
    let suffix = tokenizer.try_encode(&text)?;
    if positions.len() != crit.len() || positions.last().copied() != Some(suffix.len() - 1) {
        return Err("rubric score positions do not match the encoded lines".into());
    }
    Ok(KindRubric { suffix, positions })
}

fn roll_mini_batch(
    rollout: &SparkModel,
    reference: &SparkModel,
    tokenizer: &SparkTokenizer,
    rubrics: &RubricTokens,
    sampling: &SamplingConfig,
    bank: &[Seed],
    seed_stream: &mut SeedStream,
    draw: u64,
    seq_len: usize,
    pad_id: i32,
) -> Result<PreparedStep, String> {
    let seeds = seed_stream.next_batch(bank, STORIES_PER_MINI_BATCH);
    if seeds.len() < 2 {
        return Err(format!(
            "mini-batch has {} stories, need at least two to contrast",
            seeds.len()
        ));
    }
    let sampling = *sampling;
    let mut rows = Vec::with_capacity(seeds.len());
    let mut offset = 0;
    while offset < seeds.len() {
        let end = (offset + ROLLOUT_BATCH).min(seeds.len());
        let mut prompts = Vec::with_capacity(end - offset);
        let mut story_seeds = Vec::with_capacity(end - offset);
        let mut kinds = Vec::with_capacity(end - offset);
        for (local, seed) in seeds[offset..end].iter().enumerate() {
            let prompt = tokenizer.try_encode(&seed.prompt)?;
            if prompt.is_empty() {
                return Err("a seed encoded to no tokens".into());
            }
            let story_seed = draw
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((offset + local) as u64);
            prompts.push(prompt);
            story_seeds.push(story_seed);
            kinds.push(seed.kind);
        }
        let generated = generate_batch(rollout, &prompts, sampling, &story_seeds);
        for (local, generated) in generated.into_iter().enumerate() {
            let prompt = prompts[local].clone();
            if generated.tokens.len() <= prompt.len() {
                return Err("the policy wrote no story tokens".into());
            }
            let completion = generated.tokens[prompt.len()..].to_vec();
            if generated.sampler_logprobs.len() != completion.len() {
                return Err("the sampler did not score every story token".into());
            }
            rows.push(RolloutRow {
                prompt,
                completion,
                sampler_logprobs: generated.sampler_logprobs,
                kind: kinds[local],
                atmosphere: seeds[offset + local].atmosphere.clone(),
                text: String::new(),
            });
        }
        offset = end;
    }
    rows.par_iter_mut().for_each(|row| {
        row.text = tokenizer.decode(&row.completion);
    });
    let kinds: Vec<SeedKind> = rows.iter().map(|row| row.kind).collect();
    let mut own_raw = vec![0.0_f32; rows.len()];
    // Sort all stories in the mini-batch by completion length to minimize padding within chunks
    let mut sorted_indexes: Vec<usize> = (0..rows.len()).collect();
    sorted_indexes.sort_by_key(|&idx| rows[idx].completion.len());
    let mut start = 0;
    while start < sorted_indexes.len() {
        let end = (start + EVAL_BATCH).min(sorted_indexes.len());
        let chosen = &sorted_indexes[start..end];
        let scores = score_rubric_batch(reference, &rows, rubrics, chosen)?;
        for (&index, score) in chosen.iter().zip(scores) {
            own_raw[index] = score;
        }
        start = end;
    }
    let (own_raw, foreign) = contrast_baselines_from_own(&kinds, &own_raw);
    let (progression, (own, (story_lengths, diversity))): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) =
        own_raw
            .par_iter()
            .zip(&rows)
            .map(|(score, row)| {
                let prog = exploration::intra_story_trajectory(&row.text);
                let penalized = seeds::apply_progression_penalty(*score, prog);
                let len = row.completion.len();
                let div = exploration::lexical_diversity(&row.text);
                (prog, (penalized, (len, div)))
            })
            .unzip();
    let advantages = contrast_advantages(&kinds, &own, &foreign);
    let centered = contrast_centered(&kinds, &own);
    let rewards: Vec<f32> = own
        .iter()
        .zip(&foreign)
        .map(|(own, foreign)| contrast_reward(*own, *foreign))
        .collect();
    let batch = batch_from_rows(
        &rows,
        &advantages,
        &rewards,
        &centered,
        &own,
        &own_raw,
        &progression,
        &story_lengths,
        &diversity,
        seq_len,
        pad_id,
    )?;
    Ok(PreparedStep { batch, kinds })
}

fn rubric_score(
    model: &SparkModel,
    story: &[u32],
    rubric: &KindRubric,
    digits: &[usize; 6],
) -> Result<f32, String> {
    if story.is_empty() {
        return Err("cannot score an empty story".into());
    }
    let mut ids = Vec::with_capacity(story.len() + rubric.suffix.len());
    ids.extend(story.iter().map(|id| *id as i32));
    ids.extend(rubric.suffix.iter().map(|id| *id as i32));
    let positions = rubric
        .positions
        .iter()
        .map(|position| story.len() + position)
        .collect::<Vec<_>>();
    let length = ids.len();
    if positions.iter().any(|position| *position >= length) {
        return Err("a rubric score position fell outside the passage".into());
    }
    let device = model.embed.weight.val().device();
    let input = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, length]), &device);
    let logits = host_f32(model.digit_logits_padded(input, &positions, &[], digits));
    if logits.len() != positions.len() * 6 {
        return Err("rubric logits were not 6 values per criterion".into());
    }
    let mut scores = Vec::with_capacity(positions.len());
    for row in 0..positions.len() {
        let line = &logits[row * 6..(row + 1) * 6];
        scores.push(digit_expectation_from_slice(line));
    }
    Ok(softened_rubric_mean(&scores))
}

/// Score several stories on one shared rubric suffix.
///
/// Stories are left-padded so the suffix lines up. Pad keys are masked, and
/// each real token keeps the position it would have had on its own.
fn score_rubric_rows(
    model: &SparkModel,
    rows: &[RolloutRow],
    rubric: &KindRubric,
    digits: &[usize; 6],
    indexes: &[usize],
) -> Result<Vec<f32>, String> {
    if indexes.is_empty() {
        return Ok(Vec::new());
    }
    if indexes.len() == 1 {
        return Ok(vec![rubric_score(
            model,
            &rows[indexes[0]].completion,
            rubric,
            digits,
        )?]);
    }
    let stories = indexes
        .iter()
        .map(|index| rows[*index].completion.as_slice())
        .collect::<Vec<_>>();
    if stories.iter().any(|story| story.is_empty()) {
        return Err("cannot score an empty story".into());
    }
    let max_story = stories.iter().map(|story| story.len()).max().unwrap();
    let seq = max_story + rubric.suffix.len();
    let pad_id = model.pad_token_id as i32;
    let mut data = Vec::with_capacity(indexes.len() * seq);
    let mut pads = Vec::with_capacity(indexes.len());
    for story in &stories {
        let pad = max_story - story.len();
        pads.push(pad);
        data.extend(std::iter::repeat_n(pad_id, pad));
        data.extend(story.iter().map(|id| *id as i32));
        data.extend(rubric.suffix.iter().map(|id| *id as i32));
    }
    let positions = rubric
        .positions
        .iter()
        .map(|position| max_story + position)
        .collect::<Vec<_>>();
    let device = model.embed.weight.val().device();
    let input = Tensor::<2, Int>::from_data(TensorData::new(data, [indexes.len(), seq]), &device);
    let logits = host_f32(model.digit_logits_padded(input, &positions, &pads, digits));
    if logits.len() != indexes.len() * positions.len() * 6 {
        return Err("rubric logits were not 6 values per criterion".into());
    }
    let mut scores = Vec::with_capacity(indexes.len());
    for row in 0..indexes.len() {
        let mut parts = Vec::with_capacity(positions.len());
        for criterion in 0..positions.len() {
            let start = (row * positions.len() + criterion) * 6;
            parts.push(digit_expectation_from_slice(&logits[start..start + 6]));
        }
        scores.push(softened_rubric_mean(&parts));
    }
    Ok(scores)
}

/// Score multiple stories together in one batch on the GPU, even when they have
/// different story lengths, formats, and atmosphere criteria.
fn score_rubric_batch(
    model: &SparkModel,
    rows: &[RolloutRow],
    rubrics: &RubricTokens,
    indexes: &[usize],
) -> Result<Vec<f32>, String> {
    if indexes.is_empty() {
        return Ok(Vec::new());
    }
    let mut stories_with_rubric: Vec<(&[u32], &KindRubric)> = Vec::with_capacity(indexes.len());
    for &index in indexes {
        let row = &rows[index];
        if row.completion.is_empty() {
            return Err("cannot score an empty story".into());
        }
        let atm_idx = seeds::INTENTIONAL_ATMOSPHERES
            .iter()
            .position(|&a| a == row.atmosphere)
            .unwrap_or(0);
        let rubric = &rubrics.atmospheres[row.kind.ordinal()][atm_idx];
        stories_with_rubric.push((&row.completion, rubric));
    }

    let unpadded_lens: Vec<usize> = stories_with_rubric
        .iter()
        .map(|(story, rubric)| story.len() + rubric.suffix.len())
        .collect();
    let max_seq = unpadded_lens.iter().copied().max().unwrap();
    let pad_id = model.pad_token_id as i32;

    let mut data = Vec::with_capacity(indexes.len() * max_seq);
    let mut pads = Vec::with_capacity(indexes.len());
    let mut row_positions: Vec<Vec<usize>> = Vec::with_capacity(indexes.len());
    let mut criteria_counts = Vec::with_capacity(indexes.len());

    for (story, rubric) in &stories_with_rubric {
        let unpadded = story.len() + rubric.suffix.len();
        let pad = max_seq - unpadded;
        pads.push(pad);
        data.extend(std::iter::repeat_n(pad_id, pad));
        data.extend(story.iter().map(|id| *id as i32));
        data.extend(rubric.suffix.iter().map(|id| *id as i32));

        let criteria_pos: Vec<usize> = rubric
            .positions
            .iter()
            .map(|&p| pad + story.len() + p)
            .collect();
        criteria_counts.push(criteria_pos.len());
        row_positions.push(criteria_pos);
    }

    let device = model.embed.weight.val().device();
    let input =
        Tensor::<2, Int>::from_data(TensorData::new(data, [indexes.len(), max_seq]), &device);
    let logits =
        host_f32(model.digit_logits_padded_per_row(input, &row_positions, &pads, &rubrics.digits));

    let total_criteria: usize = criteria_counts.iter().sum();
    if logits.len() != total_criteria * 6 {
        return Err(format!(
            "rubric logits length {} does not match expected {}",
            logits.len(),
            total_criteria * 6
        ));
    }

    let mut scores = Vec::with_capacity(indexes.len());
    let mut offset = 0;
    for count in criteria_counts {
        let mut parts = Vec::with_capacity(count);
        for crit in 0..count {
            let start = (offset + crit) * 6;
            parts.push(digit_expectation_from_slice(&logits[start..start + 6]));
        }
        scores.push(softened_rubric_mean(&parts));
        offset += count;
    }
    Ok(scores)
}

fn digit_expectation_from_slice(values: &[f32]) -> f32 {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = [0.0_f32; 6];
    for (slot, value) in values.iter().enumerate() {
        weights[slot] = (*value - max).exp();
    }
    let sum = weights.into_iter().sum::<f32>().max(1e-12);
    let expected = weights
        .iter()
        .enumerate()
        .map(|(digit, weight)| *weight * digit as f32)
        .sum::<f32>()
        / sum;
    expected / 5.0
}

fn batch_from_rows(
    rows: &[RolloutRow],
    advantages: &[f32],
    rewards: &[f32],
    centered: &[f32],
    rubric: &[f32],
    raw_rubric: &[f32],
    progression: &[f32],
    story_lengths: &[usize],
    diversity: &[f32],
    seq_len: usize,
    pad_id: i32,
) -> Result<PolicyBatch, String> {
    if rows.is_empty()
        || rows.len() != advantages.len()
        || rows.len() != rewards.len()
        || rows.len() != centered.len()
        || rows.len() != rubric.len()
        || rows.len() != raw_rubric.len()
        || rows.len() != progression.len()
        || rows.len() != story_lengths.len()
        || rows.len() != diversity.len()
    {
        return Err(
            "each story needs one advantage, reward, centered gap, rubric score, and metadata"
                .into(),
        );
    }
    let fitted: Result<Vec<_>, String> = rows
        .par_iter()
        .zip(advantages)
        .map(|(row, advantage)| {
            let (tokens, mask) = fit_window(&row.prompt, &row.completion, seq_len);
            if !mask.iter().any(|value| *value > 0.0) {
                return Err("sequence length left no story token to train".into());
            }
            if row.sampler_logprobs.len() != row.completion.len() {
                return Err("sampler log probabilities do not match the completion".into());
            }
            let completion_take = mask.iter().filter(|value| **value > 0.0).count();
            let mut old_logprobs = vec![0.0; tokens.len() - completion_take];
            old_logprobs.extend(row.sampler_logprobs.iter().copied().take(completion_take));
            if old_logprobs.len() != tokens.len() {
                return Err("behavior log probabilities do not match the trained window".into());
            }
            Ok((tokens, mask, old_logprobs, *advantage))
        })
        .collect();
    let fitted = fitted?;
    Ok(pack_rows(
        &fitted,
        rewards,
        centered,
        rubric,
        raw_rubric,
        progression,
        story_lengths,
        diversity,
        rows.iter().map(|row| row.text.clone()).collect(),
        pad_id,
    ))
}

fn pack_rows(
    rows: &[(Vec<i32>, Vec<f32>, Vec<f32>, f32)],
    rewards: &[f32],
    centered: &[f32],
    rubric: &[f32],
    raw_rubric: &[f32],
    progression: &[f32],
    story_lengths: &[usize],
    diversity: &[f32],
    stories: Vec<String>,
    pad_id: i32,
) -> PolicyBatch {
    let seq = rows
        .iter()
        .map(|(tokens, _, _, _)| tokens.len())
        .max()
        .unwrap();
    let mut token_data = Vec::new();
    let mut advantage_data = Vec::new();
    let mut mask_data = Vec::new();
    let mut old_data = Vec::new();
    for (tokens, mask, old_logprobs, advantage) in rows {
        let pad = seq - tokens.len();
        token_data.extend_from_slice(tokens);
        token_data.extend(std::iter::repeat_n(pad_id, pad));
        advantage_data.extend(
            mask.iter()
                .map(|value| if *value > 0.0 { *advantage } else { 0.0 }),
        );
        advantage_data.extend(std::iter::repeat_n(0.0, pad));
        mask_data.extend_from_slice(mask);
        mask_data.extend(std::iter::repeat_n(0.0, pad));
        old_data.extend_from_slice(old_logprobs);
        old_data.extend(std::iter::repeat_n(0.0, pad));
    }
    let reference_logprobs = vec![0.0; token_data.len()];
    PolicyBatch {
        tokens: token_data,
        advantages: advantage_data,
        mask: mask_data,
        old_logprobs: old_data,
        batch: rows.len(),
        seq,
        lengths: rows.iter().map(|(tokens, _, _, _)| tokens.len()).collect(),
        reference_logprobs,
        rewards: rewards.to_vec(),
        centered: centered.to_vec(),
        rubric: rubric.to_vec(),
        raw_rubric: raw_rubric.to_vec(),
        progression: progression.to_vec(),
        story_lengths: story_lengths.to_vec(),
        diversity: diversity.to_vec(),
        stories,
    }
}

struct PackedChunk {
    tokens: Vec<i32>,
    advantages: Vec<f32>,
    mask: Vec<f32>,
    old_logprobs: Vec<f32>,
    reference_logprobs: Vec<f32>,
    seq: usize,
}

/// Repack a backward slice to that slice's own longest row.
///
/// Masked padding does not enter the per-story mean, so dropping a short
/// row's trailing pad leaves its gradient unchanged.
fn pack_chunk(batch: &PolicyBatch, row: usize, rows: usize, pad_id: i32) -> PackedChunk {
    let seq = (row..row + rows)
        .map(|index| batch.lengths[index])
        .max()
        .unwrap();
    let mut tokens = Vec::new();
    let mut advantages = Vec::new();
    let mut mask = Vec::new();
    let mut old_logprobs = Vec::new();
    let mut reference_logprobs = Vec::new();
    for index in row..row + rows {
        let length = batch.lengths[index];
        let start = index * batch.seq;
        let pad = seq - length;
        tokens.extend_from_slice(&batch.tokens[start..start + length]);
        tokens.extend(std::iter::repeat_n(pad_id, pad));
        advantages.extend_from_slice(&batch.advantages[start..start + length]);
        advantages.extend(std::iter::repeat_n(0.0, pad));
        mask.extend_from_slice(&batch.mask[start..start + length]);
        mask.extend(std::iter::repeat_n(0.0, pad));
        old_logprobs.extend_from_slice(&batch.old_logprobs[start..start + length]);
        old_logprobs.extend(std::iter::repeat_n(0.0, pad));
        reference_logprobs.extend_from_slice(&batch.reference_logprobs[start..start + length]);
        reference_logprobs.extend(std::iter::repeat_n(0.0, pad));
    }
    PackedChunk {
        tokens,
        advantages,
        mask,
        old_logprobs,
        reference_logprobs,
        seq,
    }
}

/// Frozen log probabilities for every real token, scored a few rows at a time
/// with no tape. Right padding does not change earlier positions, so each
/// group's own width matches a forward of the mini-batch rectangle.
fn precompute_reference(
    reference: &SparkModel,
    batch: &PolicyBatch,
    pad_id: i32,
) -> Result<Vec<f32>, String> {
    let mut flat = vec![0.0_f32; batch.tokens.len()];
    let device = reference.embed.weight.val().device();
    let mut row = 0;
    while row < batch.batch {
        let rows = EVAL_BATCH.min(batch.batch - row);
        let seq = (row..row + rows)
            .map(|index| batch.lengths[index])
            .max()
            .unwrap();
        let mut tokens = Vec::new();
        for index in row..row + rows {
            let length = batch.lengths[index];
            let start = index * batch.seq;
            tokens.extend_from_slice(&batch.tokens[start..start + length]);
            tokens.extend(std::iter::repeat_n(pad_id, seq - length));
        }
        let input = Tensor::<2, Int>::from_data(TensorData::new(tokens, [rows, seq]), &device);
        let logprobs = host_f32(reference.aligned_logprobs_forward(input));
        for (offset, index) in (row..row + rows).enumerate() {
            let length = batch.lengths[index];
            let src = &logprobs[offset * seq..offset * seq + length];
            let dst = index * batch.seq;
            flat[dst..dst + length].copy_from_slice(src);
        }
        row += rows;
    }
    Ok(flat)
}

struct MaskedSums {
    logprob: f64,
    kl: f64,
    tokens: f64,
}

/// Mean completion log probability and k3, in f64, from the f32 scores.
///
/// Prompt and padding positions stay out. This is not the z-score the optimizer
/// multiplies in; that number can fall while the policy's log probability does not.
fn masked_token_sums(new_logprobs: &[f32], reference_logprobs: &[f32], mask: &[f32]) -> MaskedSums {
    let mut sums = MaskedSums {
        logprob: 0.0,
        kl: 0.0,
        tokens: 0.0,
    };
    for i in 0..mask.len() {
        if mask[i] <= 0.0 {
            continue;
        }
        sums.logprob += f64::from(new_logprobs[i]);
        sums.kl += f64::from(k3_kl(reference_logprobs[i], new_logprobs[i]));
        sums.tokens += 1.0;
    }
    sums
}

fn host_f32<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .cast(FloatDType::F32)
        .into_data()
        .as_slice::<f32>()
        .expect("f32 view of a float tensor")
        .to_vec()
}

fn tensor_prefix(tensor: Tensor<2>) -> f32 {
    let cols = tensor.dims()[1].min(16);
    host_f32(tensor.slice([0..1, 0..cols])).iter().sum()
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
    use rollout_engine::ByteTokenizer;

    #[test]
    fn a_cutoff_keeps_the_last_finished_sentence() {
        assert_eq!(
            complete_paragraph("The tools were on the bench. He reached for"),
            "The tools were on the bench."
        );
        assert_eq!(
            complete_paragraph("She stopped at the door."),
            "She stopped at the door."
        );
        assert_eq!(complete_paragraph("No boundary yet"), "No boundary yet");
        assert!(ends_sentence("He asked \"why?\""));
    }

    #[test]
    fn window_trains_completion_tokens_and_keeps_a_prompt_suffix() {
        let tokenizer = ByteTokenizer;
        let batch = policy_batch(
            &tokenizer,
            "abcd",
            &["WXYZ".into(), "wxyz".into()],
            &[1.0, 0.0],
            6,
            2,
        )
        .unwrap();
        assert_eq!(batch.batch, 2);
        assert_eq!(batch.seq, 6);
        assert_eq!(
            &batch.tokens[..6],
            &[
                b'c' as i32,
                b'd' as i32,
                b'W' as i32,
                b'X' as i32,
                b'Y' as i32,
                b'Z' as i32
            ]
        );
        assert_eq!(&batch.mask[..6], &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(batch.advantages[2] > 0.0);
        assert!(batch.advantages[batch.seq + 2] < 0.0);
        assert_eq!(batch.advantages[0], 0.0);
        assert_eq!(batch.stories, ["WXYZ", "wxyz"]);
        assert_eq!(batch.best_story().map(|(_, story)| story), Some("WXYZ"));
    }

    #[test]
    fn a_long_budget_keeps_the_whole_prompt_and_completion() {
        let tokenizer = ByteTokenizer;
        let batch = policy_batch(
            &tokenizer,
            "abcd",
            &["WXYZ".into()],
            &[1.0],
            DEFAULT_SEQ_LEN,
            2,
        )
        .unwrap();
        assert_eq!(batch.seq, 8);
        assert_eq!(batch.mask.iter().filter(|value| **value > 0.0).count(), 4);
        assert_eq!(&batch.mask[..4], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn the_window_holds_a_thousand_token_story() {
        assert!(GENERATED_TOKENS >= 1000);
        assert!(DEFAULT_SEQ_LEN >= GENERATED_TOKENS + SEED_NEW_TOKENS);
        assert_eq!(backward_width(DEFAULT_SEQ_LEN), 1);
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../checkpoints/XHToken/Spark-X2.6-1.7B/tokenizer.json");
        if !path.is_file() {
            return;
        }
        let tokenizer = SparkTokenizer::open(&path).unwrap();
        for kind in SeedKind::ALL {
            let request = tokenizer.try_encode(&seeds::seed_request(kind)).unwrap();
            let story = tokenizer
                .try_encode(&seeds::seed(0, kind, "").prompt)
                .unwrap();
            assert!(
                request.len() + SEED_NEW_TOKENS + GENERATED_TOKENS <= DEFAULT_SEQ_LEN,
                "{kind:?} seed request is {} tokens",
                request.len()
            );
            assert!(
                story.len() + SEED_NEW_TOKENS + GENERATED_TOKENS <= DEFAULT_SEQ_LEN,
                "{kind:?} story prompt shell is {} tokens",
                story.len()
            );
        }
        let rubrics = rubric_tokens(&tokenizer).unwrap();
        assert_eq!(rubrics.kinds.len(), 5);
        assert_eq!(
            rubrics
                .digits
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            6
        );
        for kind in &rubrics.kinds {
            assert_eq!(kind.positions.len(), criteria(SeedKind::Scene).len());
            assert!(kind.positions.windows(2).all(|pair| pair[0] < pair[1]));
        }
        assert_eq!(rubrics.atmospheres.len(), 5);
        for kind_atm in &rubrics.atmospheres {
            assert_eq!(kind_atm.len(), seeds::INTENTIONAL_ATMOSPHERES.len());
            for atm_rubric in kind_atm {
                assert_eq!(
                    atm_rubric.positions.len(),
                    criteria(SeedKind::Scene).len() + 1
                );
            }
        }
    }

    #[test]
    fn best_indices_by_kind_selects_top_story_per_format() {
        let kinds = vec![
            SeedKind::Scene,
            SeedKind::Scene,
            SeedKind::Dialogue,
            SeedKind::Dialogue,
            SeedKind::Monologue,
        ];
        let batch = PolicyBatch {
            batch: 5,
            seq: 10,
            tokens: vec![0; 50],
            advantages: vec![0.0; 50],
            rewards: vec![0.1, 0.4, 0.5, 0.2, 0.3],
            centered: vec![0.0; 5],
            lengths: vec![10; 5],
            reference_logprobs: vec![0.0; 50],
            old_logprobs: vec![0.0; 50],
            mask: vec![1.0; 50],
            rubric: vec![0.0; 5],
            raw_rubric: vec![0.0; 5],
            progression: vec![0.0; 5],
            story_lengths: vec![10; 5],
            diversity: vec![0.5; 5],
            stories: vec![
                "s1".into(),
                "s2".into(),
                "d1".into(),
                "d2".into(),
                "m1".into(),
            ],
        };
        let best = batch.best_indices_by_kind(&kinds);
        assert_eq!(best, vec![1, 2, 4]);
    }

    #[test]
    fn a_batch_story_is_appended_as_prose() {
        let mut buffer = Vec::new();
        append_batch_story(
            &mut buffer,
            2,
            3,
            "dialogue",
            0.5,
            0.8,
            0.85,
            0.7,
            120,
            0.65,
            "{\"prose\":\"Mara opened the ledger.\\u001b[2J\",\"events\":[]}",
        )
        .unwrap();
        let text = String::from_utf8(buffer).unwrap();
        assert!(text.starts_with("step 2 mini-batch 3  dialogue  reward 0.5000"));
        assert!(text.contains("rubric 0.8000"));
        assert!(text.contains("prog 0.7000"));
        assert!(text.contains("Mara opened the ledger."));
        assert!(!text.chars().any(|character| character == '\u{1b}'));
        let path = stories_log_path(Path::new(
            "checkpoints/lora/spark-x2.6-1.7b-attention.safetensors",
        ));
        assert_eq!(
            path,
            PathBuf::from("checkpoints/lora/spark-x2.6-1.7b-attention-stories.txt")
        );
        let metrics_path = metrics_log_path(Path::new(
            "checkpoints/lora/spark-x2.6-1.7b-attention.safetensors",
        ));
        assert_eq!(
            metrics_path,
            PathBuf::from("checkpoints/lora/spark-x2.6-1.7b-attention-metrics.jsonl")
        );
    }

    #[test]
    fn masked_means_ignore_prompt_tokens() {
        let sums = masked_token_sums(&[0.0, -0.2, -0.2], &[0.0, -0.2, -1.2], &[0.0, 1.0, 1.0]);
        assert_eq!(sums.tokens, 2.0);
        assert!((sums.logprob / sums.tokens + 0.2).abs() < 1e-6);
        assert!((sums.kl / sums.tokens) > 0.0);
        let prompted = masked_token_sums(&[10.0, -0.2, -0.2], &[0.0, -0.2, -0.2], &[0.0, 1.0, 1.0]);
        assert!(prompted.kl.abs() < 1e-6, "prompt token leaked into KL");
    }

    #[test]
    fn backward_tape_holds_four_stories() {
        assert_eq!(backward_width(492), 4);
        assert_eq!(backward_width(512), 4);
        assert_eq!(backward_width(1024), 1);
        assert_eq!(backward_width(1536), 1);
        assert_eq!(backward_width(4096), 1);
        assert!(backward_width(64) <= TAPE_STORIES);
    }

    #[test]
    fn highest_reward_is_the_max_and_a_mini_batch_holds_64_stories() {
        assert_eq!(highest_reward(&[0.2, 0.9, 0.4]), Some(0.9));
        assert_eq!(STORIES_PER_MINI_BATCH * ACCUMULATION_STEPS, 1024);
        assert_eq!(SEED_BANK, 512);
        assert_eq!(MAX_SEEDS, 128_000);
        let first_run: Vec<u64> = (0..seeds::kind_quota(SeedKind::Scene))
            .map(|number| situation_draw(SeedKind::Scene, number, 0))
            .collect();
        let next_run = situation_draw(SeedKind::Scene, 0, SEED_BANK);
        assert!(!first_run.contains(&next_run));
    }

    #[test]
    fn saved_seeds_are_appended_and_joined_with_the_new_batch() {
        let dir =
            std::env::temp_dir().join(format!("spark-seeds-{}-{}", std::process::id(), SEED_BANK));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("spark-seeds.jsonl");
        assert!(load_saved_seeds(&path).unwrap().is_empty());

        let first = vec![
            seeds::seed(0, SeedKind::Scene, "one private conflict"),
            seeds::seed(
                1,
                SeedKind::Dialogue,
                "two people who disagree about a debt",
            ),
        ];
        let mut live = open_seed_log(&path).unwrap();
        write_seed(&mut live, &first[0]).unwrap();
        assert_eq!(load_saved_seeds(&path).unwrap(), &first[..1]);
        write_seed(&mut live, &first[1]).unwrap();
        drop(live);
        assert_eq!(load_saved_seeds(&path).unwrap(), first);

        let second = vec![seeds::seed(
            2,
            SeedKind::Monologue,
            "a voice alone at a window",
        )];
        append_seeds(&path, &second).unwrap();
        let saved = load_saved_seeds(&path).unwrap();
        assert_eq!(saved.len(), 3);
        assert_eq!(&saved[..2], &first[..]);
        assert_eq!(saved[2], second[0]);

        let generated = vec![seeds::seed(
            3,
            SeedKind::GroupConversation,
            "three people in a queue",
        )];
        let bank = join_seed_bank(generated.clone(), saved);
        assert_eq!(bank.len(), 4);
        assert_eq!(&bank[..1], &generated[..]);
        assert_eq!(bank[1].kind, SeedKind::Scene);
        assert_eq!(bank[3].kind, SeedKind::Monologue);

        let broken = dir.join("broken.jsonl");
        std::fs::write(
            &broken,
            "{\"index\":0,\"kind\":\"nope\",\"prompt\":\"x\"}\n",
        )
        .unwrap();
        assert!(load_saved_seeds(&broken).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_story_is_plain_prose_without_control_characters() {
        let story = story_text("{\"prose\":\"Mara opened the ledger.\\u001b[2J\",\"events\":[]}");
        let shown = terminal_text(&story);
        assert_eq!(shown, "Mara opened the ledger.[2J");
        assert!(!shown.chars().any(|character| character.is_control()));
        assert_eq!(story_text("plain words"), "plain words");
    }

    #[test]
    fn gradients_stay_put_until_the_accumulated_optimizer_step() {
        let device = burn::tensor::Device::flex().autodiff();
        let model = SparkModel::new(&SparkConfig::tiny(), &device);
        let reference = reference_policy(&model);
        let mut policy = attach_lora(model, LoraSpec::attention_default()).unwrap();
        let before = adapter_abs(&policy);
        let mut accumulator = GradientsAccumulator::new();
        let mut optimizer = Fp32MasterAdam::lora_default();
        for _ in 0..2 {
            let tokens =
                Tensor::<2, Int>::from_data(TensorData::new(vec![1_i32, 2, 3, 4], [1, 4]), &device);
            let old = policy.aligned_logprobs(tokens.clone()).detach();
            let reference_logprobs = reference.aligned_logprobs(tokens.clone()).detach();
            let new = policy.aligned_logprobs(tokens);
            let advantages = Tensor::<2>::from_data(
                TensorData::new(vec![0.0_f32, 1.0, 1.0, 1.0], [1, 4]),
                &device,
            );
            let loss = clipped_policy_loss(
                new,
                old,
                reference_logprobs,
                advantages.clone(),
                advantages,
                0.2,
                0.0,
            )
            .div_scalar(2.0);
            accumulate_grads(&policy, &mut accumulator, loss);
        }
        assert!((adapter_abs(&policy) - before).abs() <= 1e-8);
        policy = optimizer.step(1e-2, policy, accumulator.grads());
        assert!((adapter_abs(&policy) - before).abs() > 1e-8);
    }

    #[test]
    fn a_short_row_is_packed_to_its_own_length() {
        let batch = policy_batch(
            &ByteTokenizer,
            "01",
            &["2345".into(), "6".into()],
            &[1.0, 0.0],
            32,
            2,
        )
        .unwrap();
        assert_eq!(batch.lengths, vec![6, 3]);
        let short = pack_chunk(&batch, 1, 1, 2);
        assert_eq!(short.seq, 3);
        assert_eq!(short.tokens, vec![b'0' as i32, b'1' as i32, b'6' as i32]);
        let both = pack_chunk(&batch, 0, 2, 2);
        assert_eq!(both.seq, 6);
        assert_eq!(
            &both.tokens[6..],
            &[b'0' as i32, b'1' as i32, b'6' as i32, 2, 2, 2]
        );
    }

    #[test]
    fn precomputed_reference_matches_each_row() {
        let device = burn::tensor::Device::flex();
        let model = SparkModel::new(&SparkConfig::tiny(), &device);
        let reference = reference_policy(&model);
        let batch = policy_batch(
            &ByteTokenizer,
            "01",
            &["2345".into(), "67".into()],
            &[1.0, 0.0],
            32,
            2,
        )
        .unwrap();
        let scored = precompute_reference(&reference, &batch, 2).unwrap();
        for (index, length) in batch.lengths.iter().enumerate() {
            let tokens = batch.tokens[index * batch.seq..index * batch.seq + length].to_vec();
            let input = Tensor::<2, Int>::from_data(TensorData::new(tokens, [1, *length]), &device);
            let alone = host_f32(reference.aligned_logprobs(input));
            let stored = &scored[index * batch.seq..index * batch.seq + length];
            for (left, right) in stored.iter().zip(&alone) {
                assert!((left - right).abs() < 1e-5, "{left} vs {right}");
            }
        }
    }

    #[test]
    fn batched_rubric_matches_each_story() {
        let device = burn::tensor::Device::flex();
        let model = SparkModel::new(&SparkConfig::tiny(), &device);
        let rubric = KindRubric {
            suffix: vec![3, 4, 5],
            positions: vec![0, 2],
        };
        let digits = [0_usize, 1, 2, 3, 4, 5];
        let rows = vec![
            RolloutRow {
                prompt: vec![1],
                completion: vec![7, 8],
                sampler_logprobs: vec![0.0, 0.0],
                kind: SeedKind::Scene,
                atmosphere: "energetic and fast-paced".into(),
                text: String::new(),
            },
            RolloutRow {
                prompt: vec![1],
                completion: vec![9, 10, 11],
                sampler_logprobs: vec![0.0; 3],
                kind: SeedKind::Dialogue,
                atmosphere: "warm and comedic".into(),
                text: String::new(),
            },
        ];
        let batched = score_rubric_rows(&model, &rows, &rubric, &digits, &[0, 1]).unwrap();
        let first = rubric_score(&model, &rows[0].completion, &rubric, &digits).unwrap();
        let second = rubric_score(&model, &rows[1].completion, &rubric, &digits).unwrap();
        assert!(
            (batched[0] - first).abs() < 1e-4,
            "{} vs {first}",
            batched[0]
        );
        assert!(
            (batched[1] - second).abs() < 1e-4,
            "{} vs {second}",
            batched[1]
        );
    }
}
