use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use burn::tensor::{DType, FloatDType, Int, Tensor};

use crate::{SplitMix, unit};

/// Candidates kept when `top_k` is off. Nucleus mass at 0.95 sits well inside this.
const CANDIDATE_CAP: usize = 1024;

thread_local! {
    static SCRATCH_SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_ORDER: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Sampler for Spark-X2.6-1.7B.
///
/// The values are the ones in `generation_config.json`: temperature 1,
/// nucleus 0.95, top-k off (`-1`), and a repetition penalty of 1. A penalty
/// of 1 leaves every logit unchanged, and a non-positive top-k does not
/// truncate the distribution. Generation stops when the end-of-sequence
/// token is sampled, or at `max_new_tokens` if it never appears.
#[derive(Debug, Clone, Copy)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub max_new_tokens: usize,
    pub eos_token_id: Option<u32>,
}

impl SamplingConfig {
    /// Published Spark-X2.6-1.7B generation config, with a 1,024-token story.
    pub fn spark_published() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: -1,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            max_new_tokens: 1_024,
            eos_token_id: Some(1),
        }
    }

    pub fn story_default() -> Self {
        Self::spark_published()
    }

    pub fn has_penalties(&self) -> bool {
        (self.repetition_penalty - 1.0).abs() > 1e-6
            || self.presence_penalty.abs() > 1e-8
            || self.frequency_penalty.abs() > 1e-8
    }
}

/// One draw from the sampler, with the log probability of that draw.
///
/// The log probability is under the distribution that was actually sampled:
/// repetition penalty, temperature, top-k, then nucleus. A greedy draw has
/// probability 1, so its log probability is 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledToken {
    pub token: usize,
    pub logprob: f32,
}

pub(crate) fn sample_token(
    logits: &[f32],
    history: &[u32],
    config: &SamplingConfig,
    rng: &mut SplitMix,
) -> SampledToken {
    assert!(!logits.is_empty());

    if !config.has_penalties() {
        if config.temperature <= 0.0 {
            let mut best_i = 0;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            return SampledToken {
                token: best_i,
                logprob: 0.0,
            };
        }

        let temp = config.temperature.max(1e-6);
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }

        if !max_val.is_finite() {
            return SampledToken {
                token: max_idx,
                logprob: 0.0,
            };
        }

        // Keep candidates within 15 nats of the max (relative probability >= exp(-15) ≈ 3e-7)
        let threshold = max_val - 15.0 * temp;
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(128);
        for (i, &v) in logits.iter().enumerate() {
            if v >= threshold {
                candidates.push((i, v));
                if candidates.len() > 1024 {
                    break;
                }
            }
        }

        if candidates.len() <= 1024 && !candidates.is_empty() {
            candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

            if config.top_k > 0 {
                candidates.truncate(config.top_k as usize);
            }

            let max_scaled = max_val / temp;
            let mut sum = 0.0f32;
            let mut probs = Vec::with_capacity(candidates.len());
            for &(_, logit) in &candidates {
                let p = ((logit / temp) - max_scaled).exp();
                probs.push(p);
                sum += p;
            }

            let denom = sum.max(1e-12);
            for p in probs.iter_mut() {
                *p /= denom;
            }

            let top_p = config.top_p.clamp(1e-6, 1.0);
            let mut cumulative = 0.0;
            let mut cutoff = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }

            let mass = cumulative.max(1e-12);
            let draw = unit(rng) * mass;
            let mut running = 0.0;
            for i in 0..cutoff {
                running += probs[i];
                if draw <= running {
                    let prob_norm = (probs[i] / mass).max(1e-12);
                    return SampledToken {
                        token: candidates[i].0,
                        logprob: prob_norm.ln(),
                    };
                }
            }

            let last = cutoff - 1;
            let prob_norm = (probs[last] / mass).max(1e-12);
            return SampledToken {
                token: candidates[last].0,
                logprob: prob_norm.ln(),
            };
        }
    }

    SCRATCH_SCORES.with(|scores_cell| {
        let mut scores = scores_cell.borrow_mut();
        scores.clear();
        scores.extend_from_slice(logits);

        apply_penalties(&mut scores, history, config);
        if config.temperature <= 0.0 {
            return SampledToken {
                token: argmax(&scores),
                logprob: 0.0,
            };
        }
        let temperature = config.temperature.max(1e-6);
        for score in scores.iter_mut() {
            if score.is_finite() {
                *score /= temperature;
            }
        }
        apply_top_k(&mut scores, config.top_k);
        softmax_in_place(&mut scores);
        let kept = nucleus(&scores, config.top_p.clamp(1e-6, 1.0));
        let draw = unit(rng);
        let mut cumulative = 0.0;
        for (index, probability) in &kept {
            cumulative += probability;
            if draw <= cumulative {
                return SampledToken {
                    token: *index,
                    logprob: probability.max(1e-12).ln(),
                };
            }
        }
        let (token, probability) = kept
            .last()
            .copied()
            .unwrap_or_else(|| (argmax(&scores), 1.0));
        SampledToken {
            token,
            logprob: probability.max(1e-12).ln(),
        }
    })
}

/// Sample using an owned generator so callers outside this crate can pass a seed.
pub fn sample_token_from_logits(logits: &[f32], temperature: f32, top_p: f32, seed: u64) -> usize {
    let mut config = SamplingConfig::spark_published();
    config.temperature = temperature;
    config.top_p = top_p;
    config.top_k = -1;
    config.repetition_penalty = 1.0;
    config.presence_penalty = 0.0;
    config.frequency_penalty = 0.0;
    let mut rng = SplitMix::new(seed);
    sample_token(logits, &[], &config, &mut rng).token
}

/// Logits prepared for one draw per row.
///
/// Penalty-free rows never copy the full vocabulary. Greedy sampling is a
/// device argmax. Temperature sampling is a device top-k, and the nucleus
/// draw runs on that short list. Penalties still need every logit on the host.
pub(crate) enum PreparedDraw {
    Full {
        values: Vec<f32>,
        vocab: usize,
    },
    Greedy {
        tokens: Vec<i32>,
    },
    Candidates {
        indices: Vec<i32>,
        values: Vec<f32>,
        k: usize,
    },
}

/// Reduce `[batch, vocab]` logits on the device, then keep only what the draw needs.
pub(crate) fn prepare_draw(logits: Tensor<2>, config: &SamplingConfig) -> PreparedDraw {
    let [batch, vocab] = logits.dims();
    assert!(batch >= 1 && vocab >= 1, "sampling needs a logit row");
    if config.has_penalties() {
        let data = logits.cast(FloatDType::F32).into_data();
        let values = data
            .as_slice::<f32>()
            .expect("penalty logits should be f32")
            .to_vec();
        return PreparedDraw::Full { values, vocab };
    }
    if config.temperature <= 0.0 {
        let tokens = host_indices(logits.cast(FloatDType::F32).argmax(1));
        return PreparedDraw::Greedy { tokens };
    }
    let temperature = config.temperature.max(1e-6);
    let k = candidate_count(config, vocab);
    let scaled = logits.cast(FloatDType::F32) / temperature;
    let (values, indices) = scaled.topk_with_indices(k, 1);
    PreparedDraw::Candidates {
        indices: host_indices(indices),
        values: host_floats(values),
        k,
    }
}

/// Draw row `row` from a prepared batch. `history` is only read when penalties are on.
pub(crate) fn draw_prepared(
    prepared: &PreparedDraw,
    row: usize,
    history: &[u32],
    config: &SamplingConfig,
    rng: &mut SplitMix,
) -> SampledToken {
    match prepared {
        PreparedDraw::Full { values, vocab } => {
            let start = row * vocab;
            sample_token(&values[start..start + vocab], history, config, rng)
        }
        PreparedDraw::Greedy { tokens } => SampledToken {
            token: tokens[row].max(0) as usize,
            logprob: 0.0,
        },
        PreparedDraw::Candidates { indices, values, k } => draw_candidates(
            &indices[row * k..(row + 1) * k],
            &values[row * k..(row + 1) * k],
            config,
            rng,
        ),
    }
}

fn candidate_count(config: &SamplingConfig, vocab: usize) -> usize {
    let requested = if config.top_k > 0 {
        config.top_k as usize
    } else {
        CANDIDATE_CAP
    };
    requested.max(1).min(vocab)
}

fn draw_candidates(
    indices: &[i32],
    values: &[f32],
    config: &SamplingConfig,
    rng: &mut SplitMix,
) -> SampledToken {
    assert!(!indices.is_empty());
    let mut order: Vec<usize> = (0..indices.len()).collect();
    order.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
    let max_scaled = values[order[0]];
    if !max_scaled.is_finite() {
        return SampledToken {
            token: indices[order[0]].max(0) as usize,
            logprob: 0.0,
        };
    }
    let mut sum = 0.0f32;
    let mut probs = Vec::with_capacity(order.len());
    for &index in &order {
        let probability = if values[index].is_finite() {
            (values[index] - max_scaled).exp()
        } else {
            0.0
        };
        probs.push(probability);
        sum += probability;
    }
    let denom = sum.max(1e-12);
    for probability in probs.iter_mut() {
        *probability /= denom;
    }
    let top_p = config.top_p.clamp(1e-6, 1.0);
    let mut cumulative = 0.0;
    let mut cutoff = probs.len();
    for (index, &probability) in probs.iter().enumerate() {
        cumulative += probability;
        if cumulative >= top_p {
            cutoff = index + 1;
            break;
        }
    }
    let mass = cumulative.max(1e-12);
    let draw = unit(rng) * mass;
    let mut running = 0.0;
    for index in 0..cutoff {
        running += probs[index];
        if draw <= running {
            let probability = (probs[index] / mass).max(1e-12);
            return SampledToken {
                token: indices[order[index]].max(0) as usize,
                logprob: probability.ln(),
            };
        }
    }
    let last = cutoff - 1;
    let probability = (probs[last] / mass).max(1e-12);
    SampledToken {
        token: indices[order[last]].max(0) as usize,
        logprob: probability.ln(),
    }
}

fn host_floats(tensor: Tensor<2>) -> Vec<f32> {
    tensor
        .into_data()
        .as_slice::<f32>()
        .expect("candidate logits should be f32")
        .to_vec()
}

fn host_indices(tensor: Tensor<2, Int>) -> Vec<i32> {
    let data = tensor.into_data();
    match data.dtype() {
        DType::I32 => data.as_slice::<i32>().expect("i32 indices").to_vec(),
        DType::I64 => data
            .as_slice::<i64>()
            .expect("i64 indices")
            .iter()
            .map(|index| i32::try_from(*index).unwrap_or(i32::MAX))
            .collect(),
        other => panic!("topk indices have unexpected dtype {other:?}"),
    }
}

/// Read `generation_config.json`. Missing fields keep the published Spark values.
pub fn sampling_from_generation_config(text: &str) -> Result<SamplingConfig, String> {
    let file: GenerationFile =
        serde_json::from_str(text).map_err(|err| format!("generation config: {err}"))?;
    let mut config = SamplingConfig::spark_published();
    assign(&mut config.temperature, file.temperature);
    assign(&mut config.top_p, file.top_p);
    if let Some(top_k) = file.top_k {
        config.top_k = i32::try_from(top_k).unwrap_or(-1);
    }
    assign(&mut config.repetition_penalty, file.repetition_penalty);
    assign(&mut config.presence_penalty, file.presence_penalty);
    assign(&mut config.frequency_penalty, file.frequency_penalty);
    if let Some(eos) = file.eos_token_id {
        config.eos_token_id = u32::try_from(eos).ok();
    }
    Ok(config)
}

/// Sampler used to draft stories.
///
/// Penalties stay off so the next token is a device top-k plus a short host
/// nucleus draw. A penalty would copy the full vocabulary to the host.
pub fn load_story_sampling(
    directory: &Path,
    max_new_tokens: usize,
) -> Result<SamplingConfig, String> {
    let path = directory.join("generation_config.json");
    let mut sampling = if path.is_file() {
        let text = std::fs::read_to_string(&path)
            .map_err(|err| format!("reading {}: {err}", path.display()))?;
        sampling_from_generation_config(&text)?
    } else {
        SamplingConfig::spark_published()
    };
    sampling.max_new_tokens = max_new_tokens.max(1);
    sampling.repetition_penalty = 1.0;
    sampling.presence_penalty = 0.0;
    sampling.frequency_penalty = 0.0;
    Ok(sampling)
}

fn assign<T>(slot: &mut T, value: Option<T>) {
    if let Some(value) = value {
        *slot = value;
    }
}

#[derive(serde::Deserialize)]
struct GenerationFile {
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    eos_token_id: Option<u64>,
}

fn apply_penalties(logits: &mut [f32], history: &[u32], config: &SamplingConfig) {
    let repetition =
        (config.repetition_penalty - 1.0).abs() > 1e-6 && config.repetition_penalty > 0.0;
    let presence = config.presence_penalty.abs() > 1e-8;
    let frequency = config.frequency_penalty.abs() > 1e-8;
    if history.is_empty() || (!repetition && !presence && !frequency) {
        return;
    }
    // The published processor visits each id in the context once. The penalty
    // does not grow with the count; frequency and presence do.
    let mut counts = HashMap::<u32, u32>::new();
    for token in history {
        *counts.entry(*token).or_default() += 1;
    }
    for (token, count) in counts {
        let Some(score) = logits.get_mut(token as usize) else {
            continue;
        };
        if !score.is_finite() {
            continue;
        }
        if repetition {
            penalize(score, config.repetition_penalty);
        }
        if frequency {
            *score -= config.frequency_penalty * count as f32;
        }
        if presence {
            *score -= config.presence_penalty;
        }
    }
}

fn penalize(score: &mut f32, penalty: f32) {
    if *score < 0.0 {
        *score *= penalty;
    } else {
        *score /= penalty;
    }
}

fn apply_top_k(logits: &mut [f32], k: i32) {
    if k <= 0 || k as usize >= logits.len() {
        return;
    }
    let k = k as usize;
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.select_nth_unstable_by(k, |left, right| logits[*right].total_cmp(&logits[*left]));
    let mut keep = vec![false; logits.len()];
    for index in &order[..k] {
        keep[*index] = true;
    }
    for (index, score) in logits.iter_mut().enumerate() {
        if !keep[index] {
            *score = f32::NEG_INFINITY;
        }
    }
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn softmax_in_place(logits: &mut [f32]) {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for val in logits.iter_mut() {
        if val.is_finite() {
            *val = (*val - max).exp();
            sum += *val;
        } else {
            *val = 0.0;
        }
    }
    let denom = sum.max(1e-12);
    for val in logits.iter_mut() {
        *val /= denom;
    }
}

fn nucleus(probabilities: &[f32], top_p: f32) -> Vec<(usize, f32)> {
    let top_p = top_p.clamp(1e-6, 1.0);
    let n = probabilities.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(0, 1.0)];
    }

    if n <= 256 {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&left, &right| probabilities[right].total_cmp(&probabilities[left]));
        return take_nucleus(&order, probabilities, top_p);
    }

    SCRATCH_ORDER.with(|order_cell| {
        let mut order = order_cell.borrow_mut();
        if order.len() != n {
            order.clear();
            order.extend(0..n);
        } else {
            for (i, v) in order.iter_mut().enumerate() {
                *v = i;
            }
        }

        let mut candidate_k = 256.min(n);
        loop {
            if candidate_k < n {
                order.select_nth_unstable_by(candidate_k - 1, |&left, &right| {
                    probabilities[right].total_cmp(&probabilities[left])
                });
            }
            let top_slice = &mut order[..candidate_k];
            top_slice.sort_by(|&left, &right| probabilities[right].total_cmp(&probabilities[left]));

            let mut cumulative = 0.0;
            let mut reached = false;
            for &index in top_slice.iter() {
                if probabilities[index] == 0.0 {
                    break;
                }
                cumulative += probabilities[index];
                if cumulative >= top_p {
                    reached = true;
                    break;
                }
            }

            if reached || candidate_k == n {
                return take_nucleus(top_slice, probabilities, top_p);
            }

            candidate_k = (candidate_k * 4).min(n);
        }
    })
}

fn take_nucleus(order: &[usize], probabilities: &[f32], top_p: f32) -> Vec<(usize, f32)> {
    let mut chosen = Vec::new();
    let mut cumulative = 0.0;
    for &index in order {
        if probabilities[index] == 0.0 && !chosen.is_empty() {
            break;
        }
        chosen.push(index);
        cumulative += probabilities[index];
        if cumulative >= top_p && !chosen.is_empty() {
            break;
        }
    }
    let mass = chosen
        .iter()
        .map(|&index| probabilities[index])
        .sum::<f32>()
        .max(1e-12);
    chosen
        .into_iter()
        .map(|index| (index, probabilities[index] / mass))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_returns_the_largest_logit() {
        assert_eq!(sample_token_from_logits(&[0.1, 2.0, 0.2], 0.0, 0.95, 1), 1);
    }

    #[test]
    fn nucleus_sampling_keeps_a_dominant_token() {
        let token = sample_token_from_logits(&[0.0, 0.0, 8.0], 1.0, 0.5, 99);
        assert_eq!(token, 2);
    }

    #[test]
    fn published_spark_config_is_temperature_one_and_nucleus() {
        let text = r#"{
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "eos_token_id": 1
        }"#;
        let config = sampling_from_generation_config(text).unwrap();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_p, 0.95);
        assert_eq!(config.top_k, -1);
        assert_eq!(config.repetition_penalty, 1.0);
        assert_eq!(config.presence_penalty, 0.0);
        assert_eq!(config.frequency_penalty, 0.0);
        assert_eq!(config.eos_token_id, Some(1));
        let explicit =
            sampling_from_generation_config(r#"{"top_k": 40, "repetition_penalty": 1.2}"#).unwrap();
        assert_eq!(explicit.top_k, 40);
        assert_eq!(explicit.repetition_penalty, 1.2);
    }

    #[test]
    fn story_sampling_keeps_the_published_draw_and_drops_penalties() {
        let dir = std::env::temp_dir().join(format!("story-sampling-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"temperature": 0.5, "top_k": 40, "repetition_penalty": 1.3}"#,
        )
        .unwrap();
        let sampling = load_story_sampling(&dir, 12).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(sampling.temperature, 0.5);
        assert_eq!(sampling.top_k, 40);
        assert_eq!(sampling.repetition_penalty, 1.0);
        assert_eq!(sampling.presence_penalty, 0.0);
        assert_eq!(sampling.frequency_penalty, 0.0);
        assert_eq!(sampling.max_new_tokens, 12);

        let empty =
            std::env::temp_dir().join(format!("story-sampling-empty-{}", std::process::id()));
        std::fs::create_dir_all(&empty).unwrap();
        let fallback = load_story_sampling(&empty, 0).unwrap();
        let _ = std::fs::remove_dir_all(&empty);
        assert_eq!(fallback.max_new_tokens, 1);
        assert_eq!(
            fallback.temperature,
            SamplingConfig::spark_published().temperature
        );
    }

    #[test]
    fn top_k_keeps_only_the_largest_logit() {
        let mut config = SamplingConfig::spark_published();
        config.temperature = 1.0;
        config.top_k = 1;
        config.top_p = 1.0;
        let mut rng = SplitMix::new(7);
        let token = sample_token(&[1.0, 2.0, 10.0, 3.0], &[], &config, &mut rng);
        assert_eq!(token.token, 2);
        assert!(
            token.logprob.abs() < 1e-4,
            "a one-token nucleus has logprob 0"
        );
    }

    #[test]
    fn sampler_logprob_matches_the_renormalized_distribution() {
        let mut config = SamplingConfig::spark_published();
        config.temperature = 1.0;
        config.top_k = -1;
        config.top_p = 1.0;
        config.repetition_penalty = 1.0;
        let mut rng = SplitMix::new(3);
        let drawn = sample_token(&[0.0, 0.0], &[], &config, &mut rng);
        assert!((drawn.logprob - 0.5_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn repetition_penalty_can_move_the_greedy_choice() {
        let mut config = SamplingConfig::spark_published();
        config.temperature = 0.0;
        config.repetition_penalty = 2.0;
        let mut rng = SplitMix::new(1);
        let token = sample_token(&[3.0, 2.5, 0.0], &[0], &config, &mut rng);
        assert_eq!(token.token, 1);
        assert_eq!(token.logprob, 0.0);
    }

    #[test]
    fn repetition_penalty_includes_the_end_marker() {
        let mut config = SamplingConfig::spark_published();
        config.temperature = 0.0;
        config.repetition_penalty = 2.0;
        config.eos_token_id = Some(0);
        let mut rng = SplitMix::new(1);
        // 5 / 2 = 2.5, which is below the untouched 4.
        let token = sample_token(&[5.0, 4.0], &[0], &config, &mut rng);
        assert_eq!(token.token, 1);
    }

    #[test]
    fn frequency_penalty_charges_a_repeated_token() {
        let mut config = SamplingConfig::spark_published();
        config.temperature = 0.0;
        config.frequency_penalty = 1.0;
        let mut rng = SplitMix::new(1);
        let history = vec![1_u32; 10];
        let token = sample_token(&[0.0, 5.0], &history, &config, &mut rng);
        assert_eq!(token.token, 0);
    }

    #[test]
    fn nucleus_sampling_large_vocab_is_fast_and_correct() {
        let mut logits = vec![-10.0_f32; 131_072];
        logits[42] = 12.0;
        logits[1000] = 10.0;
        let mut config = SamplingConfig::spark_published();
        config.temperature = 1.0;
        config.top_p = 0.95;
        let mut rng = SplitMix::new(42);
        let token = sample_token(&logits, &[], &config, &mut rng);
        assert!(token.token == 42 || token.token == 1000);
    }

    #[test]
    fn device_topk_draws_a_peak_a_tie_and_a_single_candidate() {
        use burn::tensor::{Device, TensorData};

        let device = Device::flex();
        let mut peaked = SamplingConfig::spark_published();
        peaked.temperature = 1.0;
        peaked.top_p = 0.5;
        peaked.top_k = -1;
        let peaked_logits =
            Tensor::<2>::from_data(TensorData::new(vec![0.0_f32, 0.0, 8.0], [1, 3]), &device);
        let prepared = prepare_draw(peaked_logits, &peaked);
        let mut rng = SplitMix::new(99);
        let token = draw_prepared(&prepared, 0, &[], &peaked, &mut rng);
        assert_eq!(token.token, 2);

        let mut tied = peaked;
        tied.top_p = 1.0;
        let tied_logits =
            Tensor::<2>::from_data(TensorData::new(vec![0.0_f32, 0.0], [1, 2]), &device);
        let prepared = prepare_draw(tied_logits, &tied);
        let mut rng = SplitMix::new(3);
        let token = draw_prepared(&prepared, 0, &[], &tied, &mut rng);
        assert!((token.logprob - 0.5_f32.ln()).abs() < 1e-5);

        let mut greedy_k = peaked;
        greedy_k.top_k = 1;
        greedy_k.top_p = 1.0;
        let single = Tensor::<2>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 10.0, 3.0], [1, 4]),
            &device,
        );
        let prepared = prepare_draw(single, &greedy_k);
        let mut rng = SplitMix::new(7);
        let token = draw_prepared(&prepared, 0, &[], &greedy_k, &mut rng);
        assert_eq!(token.token, 2);
        assert!(
            token.logprob.abs() < 1e-4,
            "a one-token nucleus has logprob 0"
        );
    }
}
